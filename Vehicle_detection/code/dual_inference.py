"""
Dual inference + tracking  [v2]

  Model 1 (kendaraan/bahaya)  -> model.track()  dengan BoT-SORT atau ByteTrack
  Model 2 (lampu belakang)    -> model.predict() lalu diasosiasikan ke kendaraan

Fitur:
  - ID persisten per kendaraan antar frame
  - Smoothing temporal status lampu (brake/left/right) per ID
    -> sein yang berkedip tetap terbaca "aktif" secara stabil
  - Pilihan tracker lewat argumen: --tracker botsort | bytetrack
  - Slot filter ROI trapesium (opsional, matikan/nyalakan lewat konfigurasi)

Contoh pemakaian (dari root project):
    python code/dual_inference.py --source sample.mp4
    python code/dual_inference.py --source sample.mp4 --tracker bytetrack --save
"""

import argparse
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ====== Konfigurasi tracker (ditulis otomatis ke file saat runtime) ======
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

def write_tracker_cfg(path="_botsort_runtime.yaml"):
    Path(path).write_text(BOTSORT_CUSTOM, encoding="utf-8")
    return path

# ================== KONFIGURASI ==================
VEHICLE_MODEL = r"runs\detect\motorcyclist-hazard\exp26\weights\best.pt"  # yolo26
LIGHT_MODEL = r"runs\detect\taillight\exp1-yolov10\weights\best.pt"       # yolov10
CONF_VEHICLE = 0.4
CONF_LIGHT = 0.30
DEVICE = 0

# Smoothing status lampu: dianggap AKTIF jika terdeteksi minimal
# HITS_NEEDED kali dalam WINDOW frame terakhir (untuk ID yang sama)
WINDOW = 15
HITS_NEEDED = 3
IMPORTANT_LIGHTS = ("brake", "left", "right")

# ROI trapesium (piksel). None = filter ROI nonaktif (semua kendaraan diproses).
# Isi dengan 4 titik [x, y] hasil kalibrasi homography kamu, urut:
# kiri-dekat, kanan-dekat, kanan-jauh, kiri-jauh. Contoh:
# ROI_POLYGON = np.array([[420, 700], [860, 700], [700, 430], [580, 430]], np.int32)
ROI_POLYGON = None
# =================================================

COLOR_VEHICLE = (80, 200, 80)
COLOR_ALERT = (0, 80, 255)
COLOR_LIGHT = (255, 200, 0)
COLOR_ROI = (200, 120, 255)
FONT_SCALE_BASE = 1.0   # perbesar/perkecil semua teks dari sini (mis. 1.3)

light_history = defaultdict(lambda: deque(maxlen=WINDOW))


def center_in_box(inner, outer):
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def foot_in_roi(box):
    """True jika titik kaki box ada di dalam ROI (atau ROI nonaktif)."""
    if ROI_POLYGON is None:
        return True
    foot = ((box[0] + box[2]) / 2, box[3])
    return cv2.pointPolygonTest(ROI_POLYGON, foot, False) >= 0


def process_frame(frame, m_vehicle, m_light, tracker_cfg):
    rv = m_vehicle.track(frame, conf=CONF_VEHICLE, device=DEVICE,
                         persist=True, tracker=tracker_cfg, verbose=False)[0]
    rl = m_light.predict(frame, conf=CONF_LIGHT, device=DEVICE, verbose=False)[0]

    vehicles = []
    for b in rv.boxes:
        tid = int(b.id) if b.id is not None else -1
        vehicles.append({
            "id": tid,
            "box": tuple(b.xyxy[0].tolist()),
            "cls": m_vehicle.names[int(b.cls)],
            "conf": float(b.conf),
            "lights": [],
            "in_roi": foot_in_roi(tuple(b.xyxy[0].tolist())),
        })

    unmatched = []
    for b in rl.boxes:
        lbox = tuple(b.xyxy[0].tolist())
        lname = m_light.names[int(b.cls)]
        lconf = float(b.conf)
        candidates = [v for v in vehicles if center_in_box(lbox, v["box"])]
        if candidates:
            smallest = min(candidates, key=lambda v: (v["box"][2] - v["box"][0])
                                                     * (v["box"][3] - v["box"][1]))
            smallest["lights"].append((lname, lconf, lbox))
        else:
            unmatched.append((lname, lconf, lbox))

    # ---- smoothing temporal per track-id ----
    for v in vehicles:
        names_now = {l[0] for l in v["lights"]}
        if v["id"] >= 0:
            light_history[v["id"]].append(names_now)
            hist = light_history[v["id"]]
            v["stable"] = {name: sum(name in h for h in hist) >= HITS_NEEDED
                           for name in IMPORTANT_LIGHTS}
        else:  # box tanpa ID (jarang): pakai deteksi frame ini apa adanya
            v["stable"] = {name: (name in names_now) for name in IMPORTANT_LIGHTS}

    return vehicles, unmatched


def draw(frame, vehicles, unmatched):
    # skala otomatis relatif tinggi frame (acuan: 720p)
    s = (frame.shape[0] / 720.0) * FONT_SCALE_BASE
    font_scale = 0.7 * s
    font_thick = max(2, int(round(2 * s)))
    box_thick = max(2, int(round(2 * s)))

    if ROI_POLYGON is not None:
        cv2.polylines(frame, [ROI_POLYGON], True, COLOR_ROI, box_thick)

    for v in vehicles:
        x1, y1, x2, y2 = map(int, v["box"])
        active = [n for n, on in v["stable"].items() if on]
        alert = bool(active) and v["in_roi"]
        color = COLOR_ALERT if alert else COLOR_VEHICLE

        label = f"#{v['id']} {v['cls']}"
        if active:
            label += " | " + ",".join(n.upper() for n in active)
        if ROI_POLYGON is not None and not v["in_roi"]:
            label += " (luar ROI)"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color,
                      box_thick + (1 if alert else 0))

        # teks dengan latar solid supaya terbaca di background apa pun
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        ty = y1 - 8 if y1 - th - 12 > 0 else y2 + th + 8
        cv2.rectangle(frame, (x1, ty - th - baseline),
                      (x1 + tw + 6, ty + baseline), color, -1)
        cv2.putText(frame, label, (x1 + 3, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), font_thick)

        for lname, lconf, lbox in v["lights"]:
            lx1, ly1, lx2, ly2 = map(int, lbox)
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), COLOR_LIGHT,
                          max(1, box_thick - 1))

    for lname, lconf, lbox in unmatched:
        lx1, ly1, lx2, ly2 = map(int, lbox)
        cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), COLOR_LIGHT,
                      max(1, box_thick - 1))
        cv2.putText(frame, f"{lname}?", (lx1, max(ly1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8,
                    COLOR_LIGHT, max(1, font_thick - 1))
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="file video")
    ap.add_argument("--tracker", default="botsort",
                    choices=["botsort", "bytetrack"],
                    help="algoritma tracking (default: botsort)")
    ap.add_argument("--save", action="store_true", help="simpan video hasil")
    ap.add_argument("--out", default="dual_track_output.mp4")
    args = ap.parse_args()

    tracker_cfg = f"{args.tracker}.yaml"
    print(f"Tracker: {args.tracker}")

    m_vehicle = YOLO(VEHICLE_MODEL)
    m_light = YOLO(LIGHT_MODEL)

    cap = cv2.VideoCapture(str(Path(args.source)))
    if not cap.isOpened():
        print(f"Gagal membuka: {args.source}")
        return

    writer = None
    if args.save:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (w, h))

    n_frame = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n_frame += 1
        vehicles, unmatched = process_frame(frame, m_vehicle, m_light, tracker_cfg)
        frame = draw(frame, vehicles, unmatched)

        # ===== TITIK INTEGRASI risk_scoring =====
        # `vehicles`: list per kendaraan berisi
        #   id      -> ID persisten antar frame
        #   box     -> posisi (x1,y1,x2,y2)
        #   cls     -> kelas dari model kendaraan
        #   stable  -> {'brake':bool,'left':bool,'right':bool} SUDAH di-smoothing
        #   in_roi  -> True jika di dalam koridor ROI
        # Contoh aturan: warning jika v["in_roi"] and v["stable"]["brake"]
        # ========================================

        if writer:
            writer.write(frame)
        cv2.imshow("dual inference + tracking", frame)
        if cv2.waitKey(1) == 27:   # Esc
            break

    cap.release()
    if writer:
        writer.release()
        print(f"Video tersimpan: {args.out}")
    cv2.destroyAllWindows()
    print(f"Selesai. {n_frame} frame diproses, "
          f"{len(light_history)} kendaraan unik ter-track.")


if __name__ == "__main__":
    main()