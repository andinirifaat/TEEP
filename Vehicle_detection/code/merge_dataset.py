"""
merge_datasets.py
==================
Menggabungkan dataset RSUD20K (13 kelas: kendaraan/orang) dengan dataset
Roboflow "autonomous-vehicle-qvcsy" (kelas: rambu/sinyal/kondisi jalan)
menjadi SATU dataset YOLO siap training (mis. untuk YOLOv8 / YOLO26).

Struktur input yang diharapkan:

datasets/
├── rsud20k/
│   ├── images/{train,val,test}/*.jpg
│   └── labels/{train,val,test}/*.txt
└── roboflow/
    ├── train/{images,labels}/
    ├── valid/{images,labels}/   (akan dipetakan ke "val")
    └── test/{images,labels}/

Struktur output:

merged_dataset/
├── images/{train,val,test}/
├── labels/{train,val,test}/
└── data.yaml

Cara pakai:
    python merge_datasets.py

Kalau path dataset kamu beda, ubah variabel RSUD20K_DIR dan ROBOFLOW_DIR
di bagian CONFIG di bawah.
"""

import os
import shutil
import sys
from pathlib import Path

# =========================== CONFIG ===========================

RSUD20K_DIR = Path("dataset/RSUD20K")
ROBOFLOW_DIR = Path("dataset/traffic_sign")
OUTPUT_DIR = Path("merged_dataset")

# 13 kelas asli RSUD20K, urutan HARUS sama persis dengan classes.txt / data.yaml
# resmi mereka (index 0-12). Kalau kamu cek data.yaml asli RSUD20K dan urutannya
# beda, sesuaikan list ini.
RSUD20K_CLASSES = [
    "person",
    "rickshaw",
    "rickshaw van",
    "auto rickshaw",
    "truck",
    "pickup truck",
    "private car",
    "motorcycle",
    "bicycle",
    "bus",
    "micro bus",
    "covered van",
    "human hauler",
]

# Kelas dataset Roboflow (autonomous-vehicle-qvcsy, v2), diambil PERSIS dari
# data.yaml hasil download (urutan alfabetis, 12 kelas). Jangan diubah urutannya
# kecuali kamu download versi lain dan urutannya beda.
# Catatan: "pathole" (bukan "pothole") adalah ejaan asli di dataset sumbernya --
# sengaja dipertahankan apa adanya supaya konsisten dengan nama kelas asli.
ROBOFLOW_CLASSES = [
    "30",
    "90",
    "green",
    "hump",
    "left",
    "pathole",
    "red",
    "right",
    "schoolzone",
    "stop",
    "straight",
    "yellow",
]

# Mapping nama folder split. RSUD20K pakai "val", Roboflow biasanya pakai
# "valid" -- kita satukan semua jadi "train"/"val"/"test".
SPLIT_MAP_ROBOFLOW = {
    "train": "train",
    "valid": "val",
    "val": "val",
    "test": "test",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ================================================================


def offset_label_file(src_path: Path, dst_path: Path, offset: int):
    """Baca file label YOLO (.txt), geser class_id sebesar `offset`,
    lalu simpan ke lokasi tujuan."""
    with open(src_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    new_lines = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        try:
            cls_id = int(parts[0])
        except ValueError:
            print(f"  [WARNING] Baris label aneh dilewati: {src_path} -> {line}")
            continue
        parts[0] = str(cls_id + offset)
        new_lines.append(" ".join(parts))

    with open(dst_path, "w") as f:
        f.write("\n".join(new_lines))
        if new_lines:
            f.write("\n")


def copy_split(
    images_src: Path,
    labels_src: Path,
    images_dst: Path,
    labels_dst: Path,
    prefix: str,
    class_offset: int = 0,
):
    """Copy pasangan gambar+label dari satu sumber ke folder merged_dataset,
    dengan prefix nama file (biar tidak collision) dan offset class_id
    (biar tidak tabrakan antar dataset)."""

    if not images_src.exists():
        print(f"  [SKIP] Folder tidak ditemukan: {images_src}")
        return 0

    images_dst.mkdir(parents=True, exist_ok=True)
    labels_dst.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in sorted(images_src.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        label_path = labels_src / (img_path.stem + ".txt")
        new_stem = f"{prefix}_{img_path.stem}"
        new_img_path = images_dst / (new_stem + img_path.suffix.lower())

        shutil.copy2(img_path, new_img_path)

        new_label_path = labels_dst / (new_stem + ".txt")
        if label_path.exists():
            if class_offset:
                offset_label_file(label_path, new_label_path, class_offset)
            else:
                shutil.copy2(label_path, new_label_path)
        else:
            # Gambar tanpa objek (background negatif) -> buat label kosong
            new_label_path.touch()

        count += 1

    return count


def main():
    if not RSUD20K_DIR.exists():
        print(f"[ERROR] Folder RSUD20K tidak ditemukan: {RSUD20K_DIR.resolve()}")
        sys.exit(1)
    if not ROBOFLOW_DIR.exists():
        print(f"[ERROR] Folder Roboflow tidak ditemukan: {ROBOFLOW_DIR.resolve()}")
        sys.exit(1)

    all_classes = RSUD20K_CLASSES + ROBOFLOW_CLASSES
    roboflow_offset = len(RSUD20K_CLASSES)

    print("=" * 60)
    print(f"Total kelas gabungan : {len(all_classes)}")
    print(f"  RSUD20K   : 0 - {len(RSUD20K_CLASSES) - 1} ({len(RSUD20K_CLASSES)} kelas)")
    print(
        f"  Roboflow  : {roboflow_offset} - {len(all_classes) - 1} "
        f"({len(ROBOFLOW_CLASSES)} kelas, offset +{roboflow_offset})"
    )
    print("=" * 60)

    # ---------- RSUD20K ----------
    print("\n[1/2] Menyalin dataset RSUD20K (tanpa offset)...")
    for split in ["train", "val", "test"]:
        n = copy_split(
            images_src=RSUD20K_DIR / "images" / split,
            labels_src=RSUD20K_DIR / "labels" / split,
            images_dst=OUTPUT_DIR / "images" / split,
            labels_dst=OUTPUT_DIR / "labels" / split,
            prefix="rsud20k",
            class_offset=0,
        )
        print(f"  {split}: {n} gambar disalin")

    # ---------- Roboflow ----------
    print("\n[2/2] Menyalin dataset Roboflow (dengan offset class_id +%d)..." % roboflow_offset)
    for rf_split, merged_split in SPLIT_MAP_ROBOFLOW.items():
        rf_images = ROBOFLOW_DIR / rf_split / "images"
        rf_labels = ROBOFLOW_DIR / rf_split / "labels"
        if not rf_images.exists():
            continue
        n = copy_split(
            images_src=rf_images,
            labels_src=rf_labels,
            images_dst=OUTPUT_DIR / "images" / merged_split,
            labels_dst=OUTPUT_DIR / "labels" / merged_split,
            prefix="roboflow",
            class_offset=roboflow_offset,
        )
        print(f"  {rf_split} -> {merged_split}: {n} gambar disalin")

    # ---------- data.yaml ----------
    print("\nMembuat data.yaml gabungan...")
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUTPUT_DIR.resolve()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        f.write(f"nc: {len(all_classes)}\n")
        f.write("names:\n")
        for i, name in enumerate(all_classes):
            f.write(f"  {i}: {name}\n")

    print(f"  data.yaml disimpan di: {yaml_path.resolve()}")

    # ---------- Ringkasan ----------
    print("\n" + "=" * 60)
    print("SELESAI! Dataset gabungan siap di:", OUTPUT_DIR.resolve())
    print("=" * 60)
    print("\nLangkah training selanjutnya:")
    print(
        f"  yolo train model=yolov8l.pt data={yaml_path} "
        "epochs=100 imgsz=640 batch=64 device=0"
    )


if __name__ == "__main__":
    main()