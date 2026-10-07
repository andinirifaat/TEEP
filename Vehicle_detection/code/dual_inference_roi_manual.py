"""
risk_dual_inference.py  [v4 - ROI manual + perbaikan deteksi parkir]
====================================================================
Pipeline: deteksi kendaraan (YOLO26, tracking BoT-SORT custom)
        + deteksi lampu belakang (YOLOv10, diasosiasikan per kendaraan)
        + risk scoring (TTC looming, manuver lateral, bobot kelas, lampu)
        + deteksi kendaraan parkir (rider-check, background PDiff
          dengan guard koridor-tengah & guard tekstur latar)
        + ROI TRAPESIUM MANUAL (hasil pick_roi.py):
          objek di luar ROI -> skor 0 (bukan ancaman jalur kita)

Cara pakai (dari root project):
    python code/risk_dual_inference.py --source dataset/sample2.mp4 --save
    tambah --show untuk preview live (lebih lambat)
"""

import argparse
import math
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ============================== KONFIGURASI ==============================
VEHICLE_MODEL = r"runs\detect\motorcyclist-hazard\exp26\weights\best.pt"
LIGHT_MODEL = r"runs\detect\taillight\exp1-yolov10\weights\best.pt"
CONF_VEHICLE = 0.4
CONF_LIGHT = 0.30
PERSON_CONF = 0.25     # ambang longgar khusus bukti rider (deteksi 'person')
DEVICE = 0
TRACK_CONF = 0.1       # conf rendah untuk tracker -> anti ID-switch
OUTPUT_DIR = r"risk_output"

WINDOW = 15
HITS_NEEDED = 3

CLASS_RISK_WEIGHT = {
    "person": 0.9, "bicycle": 0.6, "motorcycle": 0.55,
    "rickshaw": 0.5, "rickshaw van": 0.5, "auto rickshaw": 0.5,
    "human hauler": 0.5, "private car": 0.6, "pickup truck": 0.7,
    "micro bus": 0.75, "covered van": 0.75, "bus": 0.85, "truck": 0.9,
    "pathole": 0.8, "hump": 0.4,
    "30": 0.1, "90": 0.1, "green": 0.0, "left": 0.0, "red": 0.2,
    "right": 0.0, "schoolzone": 0.3, "stop": 0.2, "straight": 0.0,
    "yellow": 0.15,
}

HISTORY_LEN = 10
TTC_WARNING_SEC = 3.5
LATERAL_DRIFT_THRESHOLD = 0.015
RISK_LOW_MAX = 40
RISK_MED_MAX = 70
BRAKE_BONUS = 15.0
SIGNAL_AS_MANEUVER = True

# --- Deteksi kendaraan parkir ---
TWO_WHEEL_CLASSES = {"motorcycle", "bicycle"}
FOUR_WHEEL_CLASSES = {"private car", "truck", "pickup truck", "bus",
                      "micro bus", "covered van", "rickshaw",
                      "auto rickshaw", "rickshaw van", "human hauler"}
RIDER_CLASS = "person"
RIDER_OVERLAP_MIN = 0.25
ASPECT_SIDE_4W = 1.5
ASPECT_SIDE_2W = 1.1
PDIFF_THRESHOLD = 28.0     # rata2 |beda piksel|/titik; > ini = bergerak
PDIFF_MIN_HISTORY = 3
TEXTURE_STD_MIN = 12.0     # latar harus bertekstur agar PDiff dipercaya
CENTER_CORRIDOR = (0.35, 0.65)  # koridor tengah: tak pernah divonis parkir
PARKED_SCORE_MULT = 0.2

# --- ROI trapesium MANUAL (hasil pick_roi.py) ---
# urutan: kiri-dekat, kanan-dekat, kanan-jauh, kiri-jauh
ROI_POLYGON = np.array([
    [1, 882],
    [1916, 894],
    [938, 711],
    [725, 706],
], np.int32)

FONT_SCALE_BASE = 1.0

BOTSORT_CUSTOM = """\
tracker_type: botsort
track_high_thresh: 0.4
track_low_thresh: 0.1
new_track_thresh: 0.6
track_buffer: 60
match_thresh: 0.8
fuse_score: True
gmc_method: sparseOptFlow
with_reid: True
model: auto
proximity_thresh: 0.5
appearance_thresh: 0.25
"""
RUNTIME_TRACKER_FILE = "_botsort_runtime.yaml"

COLOR_LEVEL = {"SAFE": (0, 200, 0), "WARNING": (0, 200, 255),
               "DANGER": (0, 0, 255)}
COLOR_PARKED = (180, 180, 180)
COLOR_LIGHT = (255, 200, 0)
COLOR_ROI = (200, 120, 255)

# ============================== STATE ==============================

light_history = defaultdict(lambda: deque(maxlen=WINDOW))
parked_votes = defaultdict(lambda: deque(maxlen=WINDOW))


class TrackHistory:
    def __init__(self, maxlen=HISTORY_LEN):
        self.sizes = defaultdict(lambda: deque(maxlen=maxlen))
        self.centers_x = defaultdict(lambda: deque(maxlen=maxlen))
        self.frame_idx = defaultdict(lambda: deque(maxlen=maxlen))
        self.prev_box = {}

    def update(self, track_id, area, center_x, frame_idx):
        self.sizes[track_id].append(math.sqrt(max(area, 1e-6)))
        self.centers_x[track_id].append(center_x)
        self.frame_idx[track_id].append(frame_idx)

    def estimate_ttc(self, track_id, fps):
        sizes, frames = self.sizes[track_id], self.frame_idx[track_id]
        if len(sizes) < 3:
            return None
        df = frames[-1] - frames[0]
        if df <= 0:
            return None
        growth = (sizes[-1] - sizes[0]) / df
        if growth <= 0:
            return None
        return max((sizes[-1] / growth) / fps, 0.0)

    def detect_lateral_maneuver(self, track_id, frame_center_x, frame_width):
        centers, frames = self.centers_x[track_id], self.frame_idx[track_id]
        if len(centers) < 3:
            return False, ""
        df = frames[-1] - frames[0]
        if df <= 0:
            return False, ""
        drift = (centers[-1] - centers[0]) / df / frame_width
        currently_left = centers[-1] < frame_center_x
        if currently_left and drift > LATERAL_DRIFT_THRESHOLD:
            return True, "menyalip dari kiri"
        if (not currently_left) and drift < -LATERAL_DRIFT_THRESHOLD:
            return True, "menyalip dari kanan"
        return False, ""


# ==================== HEURISTIK KENDARAAN PARKIR ====================

def overlap_fraction(inner, outer):
    ix1 = max(inner[0], outer[0]); iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2]); iy2 = min(inner[3], outer[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inner_area = max((inner[2] - inner[0]) * (inner[3] - inner[1]), 1e-6)
    return (iw * ih) / inner_area


def has_rider(moto_box, person_boxes):
    for pbox in person_boxes:
        if overlap_fraction(pbox, moto_box) >= RIDER_OVERLAP_MIN:
            return True
    return False


def is_side_facing(box, cls_name):
    w = box[2] - box[0]
    h = max(box[3] - box[1], 1e-6)
    thresh = ASPECT_SIDE_2W if cls_name in TWO_WHEEL_CLASSES else ASPECT_SIDE_4W
    return (w / h) > thresh


def sample_points_around(box, W, H):
    x1, y1, x2, y2 = box
    mx = 0.06 * (x2 - x1)
    my = 0.06 * (y2 - y1)
    pts = [
        (x1 - mx, y1 - my), (x2 + mx, y1 - my),
        (x1 - mx, y2 + my), (x2 + mx, y2 + my),
        ((x1 + x2) / 2, y1 - my), ((x1 + x2) / 2, y2 + my),
        (x1 - mx, (y1 + y2) / 2), (x2 + mx, (y1 + y2) / 2),
    ]
    return [(int(min(max(x, 4), W - 5)), int(min(max(y, 4), H - 5)))
            for x, y in pts]


def background_pdiff(gray_now, gray_prev, box_now, box_prev):
    H, W = gray_now.shape[:2]
    pts_now = sample_points_around(box_now, W, H)
    pts_prev = sample_points_around(box_prev, W, H)
    diffs = []
    for (xn, yn), (xp, yp) in zip(pts_now, pts_prev):
        patch_n = gray_now[yn - 2:yn + 3, xn - 2:xn + 3].astype(np.float32)
        patch_p = gray_prev[yp - 2:yp + 3, xp - 2:xp + 3].astype(np.float32)
        if patch_n.size and patch_p.size:
            diffs.append(abs(float(patch_n.mean()) - float(patch_p.mean())))
    return float(np.mean(diffs)) if diffs else None


def judge_parked(o, person_boxes, gray_now, gray_prev, history):
    """Vonis parkir butuh bukti positif gerak-nol dari latar (PDiff) DAN
    latar yang bertekstur; rider-check hanya pendukung untuk roda dua.
    Kendaraan di koridor tengah tidak pernah divonis parkir (kalau pun
    benar berhenti di jalur kita, itu justru ancaman yang harus tetap
    bisa memicu warning). Default = bergerak (gagal ke sisi aman)."""
    tid, cls, box = o["id"], o["cls"], o["box"]

    cx_norm = ((box[0] + box[2]) / 2) / gray_now.shape[1]
    if CENTER_CORRIDOR[0] < cx_norm < CENTER_CORRIDOR[1]:
        vote_parked = False
    else:
        pdiff = None
        texture_ok = False
        if gray_prev is not None and tid in history.prev_box \
                and len(history.frame_idx[tid]) >= PDIFF_MIN_HISTORY:
            pdiff = background_pdiff(gray_now, gray_prev, box,
                                     history.prev_box[tid])
            H, W = gray_now.shape[:2]
            pts = sample_points_around(box, W, H)
            stds = [float(gray_now[y - 4:y + 5, x - 4:x + 5].std())
                    for x, y in pts]
            texture_ok = float(np.mean(stds)) > TEXTURE_STD_MIN

        background_still = (pdiff is not None and texture_ok
                            and pdiff < PDIFF_THRESHOLD)
        if cls in TWO_WHEEL_CLASSES:
            vote_parked = background_still and not has_rider(box, person_boxes)
        else:
            vote_parked = background_still

    if tid >= 0:
        parked_votes[tid].append(vote_parked)
        votes = parked_votes[tid]
        return len(votes) >= 5 and sum(votes) >= 0.7 * len(votes)
    return False


# ============================== LOGIKA ==============================

def resolve_tracker(choice):
    if choice == "custom":
        Path(RUNTIME_TRACKER_FILE).write_text(BOTSORT_CUSTOM, encoding="utf-8")
        return RUNTIME_TRACKER_FILE
    return f"{choice}.yaml"


def center_in_box(inner, outer):
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def foot_in_roi(box):
    """Titik kaki objek (tengah sisi bawah box) harus di dalam trapesium ROI."""
    foot = (float((box[0] + box[2]) / 2), float(box[3]))
    return cv2.pointPolygonTest(ROI_POLYGON, foot, False) >= 0


def compute_risk_score(cls_name, ttc_seconds, is_maneuvering,
                       brake_on, is_parked):
    class_weight = CLASS_RISK_WEIGHT.get(cls_name, 0.3)
    class_multiplier = 0.6 + class_weight * 0.6

    score_ttc = 0.0
    if ttc_seconds is not None:
        score_ttc = 100 * max(0.0, 1 - (ttc_seconds / (TTC_WARNING_SEC + 1.0)))

    score = score_ttc * class_multiplier
    if is_maneuvering and not is_parked:
        score += 25
    if brake_on and not is_parked:
        score += BRAKE_BONUS
    if is_parked:
        score *= PARKED_SCORE_MULT
    return round(min(score, 100.0), 1)


def risk_level(score):
    if score < RISK_LOW_MAX:
        return "SAFE"
    elif score < RISK_MED_MAX:
        return "WARNING"
    return "DANGER"


def process_frame(frame, gray_now, gray_prev, m_vehicle, m_light, tracker_cfg,
                  history, frame_idx, fps, frame_center_x, frame_width):
    rv = m_vehicle.track(frame, conf=TRACK_CONF, device=DEVICE,
                         persist=True, tracker=tracker_cfg, verbose=False)[0]
    rl = m_light.predict(frame, conf=CONF_LIGHT, device=DEVICE, verbose=False)[0]

    objects = []
    person_boxes = []
    for b in rv.boxes:
        conf = float(b.conf)
        cls = m_vehicle.names[int(b.cls)]
        # bukti rider dikumpulkan dgn ambang longgar, terpisah dari filter tampilan
        if cls == RIDER_CLASS and conf >= PERSON_CONF:
            person_boxes.append(tuple(b.xyxy[0].tolist()))
        if conf < CONF_VEHICLE:
            continue
        tid = int(b.id) if b.id is not None else -1
        box = tuple(b.xyxy[0].tolist())
        objects.append({"id": tid, "box": box, "cls": cls, "conf": conf,
                        "lights": [], "in_roi": foot_in_roi(box)})

    unmatched = []
    for b in rl.boxes:
        lbox = tuple(b.xyxy[0].tolist())
        lname = m_light.names[int(b.cls)]
        candidates = [o for o in objects if center_in_box(lbox, o["box"])]
        if candidates:
            smallest = min(candidates, key=lambda o: (o["box"][2] - o["box"][0])
                                                     * (o["box"][3] - o["box"][1]))
            smallest["lights"].append((lname, float(b.conf), lbox))
        else:
            unmatched.append((lname, float(b.conf), lbox))

    for o in objects:
        names_now = {l[0] for l in o["lights"]}
        if o["id"] >= 0:
            light_history[o["id"]].append(names_now)
            hist = light_history[o["id"]]
            o["stable"] = {n: sum(n in h for h in hist) >= HITS_NEEDED
                           for n in ("brake", "left", "right")}
        else:
            o["stable"] = {n: (n in names_now) for n in ("brake", "left", "right")}

        x1, y1, x2, y2 = o["box"]
        area = (x2 - x1) * (y2 - y1)
        cx = x1 + (x2 - x1) / 2

        o["parked"] = (judge_parked(o, person_boxes, gray_now, gray_prev,
                                    history)
                       if o["cls"] in TWO_WHEEL_CLASSES | FOUR_WHEEL_CLASSES
                       else False)
        o["side"] = is_side_facing(o["box"], o["cls"])

        if o["id"] >= 0:
            history.update(o["id"], area, cx, frame_idx)
            ttc = history.estimate_ttc(o["id"], fps)
            maneuvering, mdesc = history.detect_lateral_maneuver(
                o["id"], frame_center_x, frame_width)
            history.prev_box[o["id"]] = o["box"]
        else:
            ttc, maneuvering, mdesc = None, False, ""

        signal_on = o["stable"]["left"] or o["stable"]["right"]
        if SIGNAL_AS_MANEUVER and signal_on and not maneuvering:
            maneuvering = True
            mdesc = "sein " + ("kiri" if o["stable"]["left"] else "kanan")

        if maneuvering and o["side"] and not o["parked"]:
            mdesc = (mdesc + " [encroach]").strip()

        brake_on = o["stable"]["brake"]
        score = compute_risk_score(o["cls"], ttc, maneuvering,
                                   brake_on, o["parked"])
        if not o["in_roi"]:
            score = 0.0

        o.update({"ttc": ttc, "maneuver": maneuvering, "mdesc": mdesc,
                  "score": score, "level": risk_level(score)})

    order = {"SAFE": 0, "WARNING": 1, "DANGER": 2}
    frame_status = max((o["level"] for o in objects),
                       key=lambda l: order[l], default="SAFE")
    return objects, unmatched, frame_status


def draw(frame, objects, unmatched, frame_status):
    s = (frame.shape[0] / 720.0) * FONT_SCALE_BASE
    font_scale = 0.7 * s
    font_thick = max(2, int(round(2 * s)))
    box_thick = max(2, int(round(2 * s)))

    cv2.polylines(frame, [ROI_POLYGON], True, COLOR_ROI, box_thick)

    for o in objects:
        x1, y1, x2, y2 = map(int, o["box"])
        color = COLOR_PARKED if o.get("parked") else COLOR_LEVEL[o["level"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color,
                      box_thick + (1 if o["level"] == "DANGER" else 0))

        ttc_str = f"{o['ttc']:.1f}s" if o["ttc"] is not None else "-"
        active = [n.upper() for n, on in o["stable"].items() if on]
        label = f"#{o['id']} {o['cls']}"
        if o.get("parked"):
            label += " [PARKIR]"
        label += f" | TTC:{ttc_str} | {o['level']} ({o['score']:.0f})"
        if active:
            label += " | " + ",".join(active)
        if o["maneuver"] and o["mdesc"]:
            label += f" [{o['mdesc']}]"
        if not o["in_roi"]:
            label += " (luar ROI)"

        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        ty = y1 - 8 if y1 - th - 12 > 0 else y2 + th + 8
        cv2.rectangle(frame, (x1, ty - th - baseline),
                      (x1 + tw + 6, ty + baseline), color, -1)
        cv2.putText(frame, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

        for lname, lconf, lbox in o["lights"]:
            lx1, ly1, lx2, ly2 = map(int, lbox)
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), COLOR_LIGHT,
                          max(1, box_thick - 1))

    for lname, lconf, lbox in unmatched:
        lx1, ly1, lx2, ly2 = map(int, lbox)
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), COLOR_LIGHT,
                      max(1, box_thick - 1))

    bw, bh = int(360 * s), int(56 * s)
    cv2.rectangle(frame, (0, 0), (bw, bh), COLOR_LEVEL[frame_status], -1)
    cv2.putText(frame, f"STATUS: {frame_status}", (int(12 * s), int(40 * s)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0 * s, (255, 255, 255),
                max(2, int(3 * s)), cv2.LINE_AA)
    return frame


# ============================== MAIN ==============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--tracker", default="custom",
                    choices=["custom", "botsort", "bytetrack"])
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--show", action="store_true",
                    help="tampilkan preview live (lebih lambat)")
    args = ap.parse_args()

    tracker_cfg = resolve_tracker(args.tracker)
    print(f"Tracker : {args.tracker} ({tracker_cfg})")

    m_vehicle = YOLO(VEHICLE_MODEL)
    m_light = YOLO(LIGHT_MODEL)
    history = TrackHistory()

    cap = cv2.VideoCapture(str(Path(args.source)))
    if not cap.isOpened():
        print(f"Gagal membuka: {args.source}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frame_center_x = width / 2.0

    # ROI dikalibrasi pada frame 1920-lebar; skalakan bila resolusi beda
    global ROI_POLYGON
    if width != 1920:
        sx, sy = width / 1920.0, height / 1080.0
        ROI_POLYGON = (ROI_POLYGON.astype(np.float32)
                       * np.array([sx, sy], np.float32)).astype(np.int32)
        print(f"ROI diskalakan ke {width}x{height}")

    writer, out_path = None, None
    if args.save:
        out_path = args.out or str(
            Path(OUTPUT_DIR) / (Path(args.source).stem + "_risk.mp4"))
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        print(f"Output  : {Path(out_path).resolve()}")
    print(f"Video   : {width}x{height} @ {fps:.0f}fps, {total} frame")
    print("Mulai memproses... (Ctrl+C untuk membatalkan)")

    frame_idx = 0
    danger_frames = 0
    gray_prev = None
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray_now = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            objects, unmatched, frame_status = process_frame(
                frame, gray_now, gray_prev, m_vehicle, m_light, tracker_cfg,
                history, frame_idx, fps, frame_center_x, width)
            frame = draw(frame, objects, unmatched, frame_status)
            gray_prev = gray_now
            if frame_status == "DANGER":
                danger_frames += 1
                # ==== titik pemicu warning fisik (buzzer/lampu) ====

            if writer:
                writer.write(frame)
            if args.show:
                cv2.imshow("risk dual inference", frame)
                if cv2.waitKey(1) == 27:
                    print("\nDihentikan (Esc).")
                    break
            frame_idx += 1

            if frame_idx % 30 == 0:
                elapsed = time.time() - t0
                speed = frame_idx / elapsed
                if total:
                    pct = 100 * frame_idx / total
                    eta = (total - frame_idx) / max(speed, 1e-6)
                    print(f"\r  {pct:5.1f}%  ({frame_idx}/{total})  "
                          f"{speed:4.1f} fps  sisa ~{eta/60:4.1f} mnt   ",
                          end="", flush=True)
                else:
                    print(f"\r  {frame_idx} frame  {speed:4.1f} fps   ",
                          end="", flush=True)
    except KeyboardInterrupt:
        print("\nDibatalkan (Ctrl+C). Hasil parsial tetap tersimpan.")

    cap.release()
    if writer:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"\nSelesai. {frame_idx} frame diproses, "
          f"{danger_frames} frame DANGER, "
          f"{len(light_history)} kendaraan unik ter-track.")
    if out_path:
        print(f"Video hasil: {Path(out_path).resolve()}")


if __name__ == "__main__":
    main()