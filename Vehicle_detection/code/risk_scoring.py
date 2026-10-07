"""
risk_scoring.py
================
Lapisan risk-scoring untuk sistem pencegahan bahaya pengendara motor.

Alur:
    Video -> YOLO detection + ByteTrack tracking (per frame)
          -> Hitung skor bahaya per objek yang di-track (berbasis proxy visual:
             laju pembesaran bounding box, luas relatif, posisi lateral, bobot kelas)
          -> Overlay warna + label peringatan ke video output

Asumsi:
    - Kamera menghadap DEPAN, terpasang di motor/dashboard (mis. dashcam/helm cam).
    - Tidak ada data jarak nyata (LiDAR/radar/depth) -- semua estimasi bahaya
      adalah PROXY dari bounding box 2D, bukan pengukuran fisik akurat.
    - Class index HARUS SAMA PERSIS dengan urutan di merged_dataset/data.yaml
      (25 kelas: 0-12 RSUD20K, 13-24 Roboflow).

Cara pakai:
    python risk_scoring.py --model motorcyclist-hazard/exp1/weights/best.pt --source video.mp4

Output:
    Video baru dengan overlay bounding box berwarna sesuai level risiko,
    tersimpan di folder risk_output/.
"""

import argparse
import math
from collections import defaultdict, deque

import cv2
from ultralytics import YOLO

# ============================== PATH CONFIG ==============================

MODEL_PATH = r"D:\vehicle_detection\runs\detect\motorcyclist-hazard\exp1\weights\best.pt"
SOURCE_PATH = r"D:\vehicle_detection\dataset\sample.mp4"
OUTPUT_PATH = r"D:\vehicle_detection\risk_output\hasil.mp4"

CONFIDENCE = 0.4

# ============================== CONFIG ==============================


# Urutan class HARUS sama persis dengan merged_dataset/data.yaml
CLASS_NAMES = [
    "person", "rickshaw", "rickshaw van", "auto rickshaw", "truck",
    "pickup truck", "private car", "motorcycle", "bicycle", "bus",
    "micro bus", "covered van", "human hauler",  # 0-12 (RSUD20K)
    "30", "90", "green", "hump", "left", "pathole", "red", "right",
    "schoolzone", "stop", "straight", "yellow",  # 13-24 (Roboflow)
]

# Bobot risiko dasar per kelas (0.0 - 1.0). Kelas yang tidak relevan sebagai
# "ancaman fisik langsung" (rambu/sinyal statis) diberi bobot rendah/0,
# kecuali yang memang berarti bahaya jalan (pathole, hump).
CLASS_RISK_WEIGHT = {
    "person": 0.9,          # pejalan kaki -- rentan, prioritas tinggi
    "bicycle": 0.6,
    "motorcycle": 0.55,     # sesama motor -- risiko tabrakan sejajar
    "rickshaw": 0.5,
    "rickshaw van": 0.5,
    "auto rickshaw": 0.5,
    "human hauler": 0.5,
    "private car": 0.6,
    "pickup truck": 0.7,
    "micro bus": 0.75,
    "covered van": 0.75,
    "bus": 0.85,            # kendaraan besar -- dampak parah kalau tabrakan
    "truck": 0.9,
    "pathole": 0.8,         # bahaya jalan langsung, terutama untuk motor
    "hump": 0.4,            # perlu perlambatan tapi tidak sefatal pothole
    # rambu & sinyal -- bukan ancaman fisik langsung, bobot rendah/nol
    "30": 0.1, "90": 0.1, "green": 0.0, "left": 0.0, "red": 0.2,
    "right": 0.0, "schoolzone": 0.3,  # sekolah -> tandai kewaspadaan ekstra
    "stop": 0.2, "straight": 0.0, "yellow": 0.15,
}

HISTORY_LEN = 10        # jumlah frame riwayat per track untuk estimasi TTC & lateral drift

# --- Threshold Time-to-Collision (detik) ---
TTC_DANGER_SEC = 1.5    # TTC < 1.5s   -> BAHAYA (mendesak)
TTC_WARNING_SEC = 3.5   # 1.5-3.5s     -> WASPADA
                         # >3.5s / tidak mendekat -> AMAN (dari sisi TTC)

# --- Threshold deteksi maneuver lateral (belok/menyalip ke jalur kita) ---
LATERAL_DRIFT_THRESHOLD = 0.015   # pergeseran horizontal min. per frame (fraksi lebar frame)
                                    # untuk dianggap "bergerak menuju jalur kita"

RISK_LOW_MAX = 40
RISK_MED_MAX = 70

COLOR_LOW = (0, 200, 0)      # hijau (BGR)
COLOR_MED = (0, 200, 255)    # kuning
COLOR_HIGH = (0, 0, 255)     # merah

# ============================== INTI LOGIKA ==============================


class TrackHistory:
    """Menyimpan riwayat ukuran & posisi lateral tiap track ID, dipakai untuk
    estimasi TTC (time-to-collision) dan deteksi tren pergerakan lateral."""

    def __init__(self, maxlen=HISTORY_LEN):
        self.sizes = defaultdict(lambda: deque(maxlen=maxlen))     # sqrt(area), proxy skala linear
        self.centers_x = defaultdict(lambda: deque(maxlen=maxlen))
        self.frame_idx = defaultdict(lambda: deque(maxlen=maxlen))

    def update(self, track_id, area, center_x, frame_idx):
        self.sizes[track_id].append(math.sqrt(max(area, 1e-6)))
        self.centers_x[track_id].append(center_x)
        self.frame_idx[track_id].append(frame_idx)

    def estimate_ttc(self, track_id, fps):
        """Estimasi Time-to-Collision (detik) berbasis laju pembesaran objek
        di citra (prinsip 'looming' / tau function).
        Mengembalikan None kalau objek tidak mendekat (aman dari sisi TTC)."""
        sizes = self.sizes[track_id]
        frames = self.frame_idx[track_id]
        if len(sizes) < 3:
            return None  # riwayat belum cukup untuk estimasi andal

        size_old, size_now = sizes[0], sizes[-1]
        frame_old, frame_now = frames[0], frames[-1]
        delta_frames = frame_now - frame_old
        if delta_frames <= 0:
            return None

        growth_per_frame = (size_now - size_old) / delta_frames
        if growth_per_frame <= 0:
            return None  # tidak membesar -> tidak mendekat -> aman

        ttc_frames = size_now / growth_per_frame
        ttc_seconds = ttc_frames / fps
        return max(ttc_seconds, 0.0)

    def detect_lateral_maneuver(self, track_id, frame_center_x, frame_width):
        """Deteksi apakah objek konsisten bergerak MENDEKATI garis tengah
        frame (indikasi berbelok/menyalip masuk ke jalur kita).
        Return: (is_maneuvering: bool, direction: str)"""
        centers = self.centers_x[track_id]
        frames = self.frame_idx[track_id]
        if len(centers) < 3:
            return False, ""

        x_old, x_now = centers[0], centers[-1]
        delta_frames = frames[-1] - frames[0]
        if delta_frames <= 0:
            return False, ""

        drift_per_frame_norm = (x_now - x_old) / delta_frames / frame_width

        # Apakah objek sedang berada di sisi kiri atau kanan frame relatif ke tengah?
        currently_left = x_now < frame_center_x
        moving_right = drift_per_frame_norm > LATERAL_DRIFT_THRESHOLD
        moving_left = drift_per_frame_norm < -LATERAL_DRIFT_THRESHOLD

        # "Mendekati tengah" = objek di kiri bergerak ke kanan, ATAU objek di
        # kanan bergerak ke kiri -> konvergen ke jalur kita.
        if currently_left and moving_right:
            return True, "berbelok/menyalip dari kiri"
        if (not currently_left) and moving_left:
            return True, "berbelok/menyalip dari kanan"
        return False, ""


def compute_risk_score(cls_name, ttc_seconds, is_maneuvering):
    """Menggabungkan TTC + indikasi maneuver + bobot kelas menjadi skor 0-100.

    Logika:
      - TTC adalah faktor UTAMA (semakin kecil TTC, semakin tinggi skor dasar).
      - Bobot kelas bertindak sebagai PENGALI (kendaraan besar = dampak lebih
        parah pada TTC yang sama).
      - Indikasi maneuver (belok/menyalip ke jalur kita) menambah skor secara
        tetap, karena ini menandakan risiko akan meningkat meski TTC saat ini
        masih belum mendesak.
    """
    class_weight = CLASS_RISK_WEIGHT.get(cls_name, 0.3)
    class_multiplier = 0.6 + class_weight * 0.6  # rentang kira-kira 0.6 - 1.2

    if ttc_seconds is None:
        # Tidak mendekat (dari sisi ukuran) -> skor dasar TTC = 0
        score_ttc = 0.0
    else:
        # TTC=0 -> skor 100; TTC>=TTC_WARNING_SEC dianggap sudah tidak mendesak
        score_ttc = 100 * max(0.0, 1 - (ttc_seconds / (TTC_WARNING_SEC + 1.0)))

    score = score_ttc * class_multiplier

    if is_maneuvering:
        score += 25  # bonus tetap: objek yang menyalip/berbelok ke jalur kita

    return round(min(score, 100.0), 1)


def risk_level(score):
    if score < RISK_LOW_MAX:
        return "SAFE", COLOR_LOW
    elif score < RISK_MED_MAX:
        return "WARNING", COLOR_MED
    else:
        return "DANGER", COLOR_HIGH


# ============================== MAIN LOOP ==============================

def main():

    model_path = MODEL_PATH
    source_path = SOURCE_PATH
    output_path = OUTPUT_PATH
    conf = CONFIDENCE

    model = YOLO(model_path)
    history = TrackHistory()

    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Buka video untuk ambil info dimensi/fps sebelum mulai stream tracking
    cap = cv2.VideoCapture(source_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    frame_area = float(width * height)
    frame_center_x = width / 2.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # stream=True + persist=True -> tracking ID konsisten antar frame dalam 1 video
    results_gen = model.track(
        source=source_path,
        tracker="bytetrack.yaml",
        conf=conf,
        persist=True,
        stream=True,
        verbose=False,
    )

    frame_idx = 0
    for result in results_gen:
        frame = result.orig_img.copy()

        boxes = result.boxes
        if boxes is not None and boxes.id is not None:
            for box, track_id, cls_idx, conf_score in zip(
                boxes.xyxy.cpu().numpy(),
                boxes.id.cpu().numpy(),
                boxes.cls.cpu().numpy(),
                boxes.conf.cpu().numpy(),
            ):
                x1, y1, x2, y2 = box
                track_id = int(track_id)
                cls_idx = int(cls_idx)

                cls_name = (
                    CLASS_NAMES[cls_idx]
                    if cls_idx < len(CLASS_NAMES)
                    else f"cls{cls_idx}"
                )

                w, h = x2 - x1, y2 - y1
                area = float(w * h)
                center_x = float(x1 + w / 2)

                history.update(track_id, area, center_x, frame_idx)

                ttc = history.estimate_ttc(track_id, fps)

                is_maneuvering, maneuver_desc = history.detect_lateral_maneuver(
                    track_id,
                    frame_center_x,
                    width,
                )

                score = compute_risk_score(
                    cls_name,
                    ttc,
                    is_maneuvering,
                )

                level, color = risk_level(score)

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    3,
                )

                ttc_str = f"{ttc:.1f}s" if ttc is not None else "-"

                label = (
                    f"id:{track_id} "
                    f"{cls_name} | "
                    f"TTC:{ttc_str} | "
                    f"{level} ({score:.0f})"
                )

                if is_maneuvering:
                    label += f" [{maneuver_desc}]"

                FONT = cv2.FONT_HERSHEY_SIMPLEX
                FONT_SCALE = 0.8      # sebelumnya 0.5
                FONT_THICKNESS = 2    # sebelumnya 1

                (tw, th), baseline = cv2.getTextSize(
                    label,
                    FONT,
                    FONT_SCALE,
                    FONT_THICKNESS,
                )

                padding = 8

                cv2.rectangle(
                    frame,
                    (int(x1), int(y1) - th - baseline - padding * 2),
                    (int(x1) + tw + padding * 2, int(y1)),
                    color,
                    -1,
                )

                cv2.putText(
                    frame,
                    label,
                    (int(x1) + padding, int(y1) - padding),
                    FONT,
                    FONT_SCALE,
                    (255, 255, 255),
                    FONT_THICKNESS,
                    cv2.LINE_AA,
                )

        writer.write(frame)
        frame_idx += 1

    writer.release()

    print(f"Selesai. {frame_idx} frame diproses.")
    print(f"Video hasil tersimpan di: {output_path}")


if __name__ == "__main__":
    main()