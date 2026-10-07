"""
Konversi TLD-LOKI (COCO JSON per scenario) -> format YOLO.  [v3]

PERBAIKAN KUNCI dari v2:
Id kategori TIDAK konsisten antar scenario (tiap JSON punya pemetaan id->nama
sendiri). Maka pemetaan dilakukan PER SCENARIO berdasarkan NAMA kelas,
lalu nama dipetakan ke indeks kelas global yang konsisten.

Kelas dummy (supercategory 'none', mis. 'car'/'cars'/'cars_1') dibuang.

Jalankan dari root project:
    python code/convert_tld.py
"""

import json
import random
import shutil
from collections import Counter
from pathlib import Path

# ================== KONFIGURASI ==================
SRC_ROOT = Path(r"dataset/TLD-LOKI")
OUT_ROOT = Path(r"TLD_yolo")
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42
COPY_MODE = "copy"   # "copy" atau "move"
# Opsional: batasi kelas yang dipakai, mis. ["brake", "go"] untuk rem saja.
# Kosongkan (None) untuk memakai semua kelas sungguhan yang ditemukan.
KEEP_CLASSES = ["brake", "go", "left", "right"]    # sebelumnya: None
# =================================================


def find_scenarios(root: Path):
    return sorted(ann.parent for ann in root.rglob("_annotations.coco.json"))


def real_categories(coco):
    """Kategori sungguhan di satu JSON: buang dummy (supercategory 'none')."""
    return {c["id"]: c["name"].strip().lower()
            for c in coco.get("categories", [])
            if c.get("supercategory", "").lower() != "none"}


def main():
    random.seed(SEED)
    scenarios = find_scenarios(SRC_ROOT)
    if not scenarios:
        print(f"Tidak ada scenario ditemukan di {SRC_ROOT.resolve()}")
        return
    print(f"Ditemukan {len(scenarios)} scenario.")

    # ---- Pass 1: kumpulkan semua NAMA kelas sungguhan + hitung instance ----
    print("Pass 1: scan nama kelas dari semua JSON...")
    name_count = Counter()
    for sc in scenarios:
        with open(sc / "_annotations.coco.json", encoding="utf-8") as f:
            coco = json.load(f)
        id2name = real_categories(coco)
        for a in coco.get("annotations", []):
            name = id2name.get(a["category_id"])
            if name:
                name_count[name] += 1

    print("\nSemua kelas sungguhan (berdasar nama) dan jumlah instansinya:")
    for name, cnt in name_count.most_common():
        print(f"  {name}: {cnt}")

    if KEEP_CLASSES:
        class_names = [c.strip().lower() for c in KEEP_CLASSES]
        print(f"\nKEEP_CLASSES aktif -> hanya memakai: {class_names}")
    else:
        class_names = sorted(name_count.keys())
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    print(f"Kelas final untuk YOLO: {dict(enumerate(class_names))}")

    # ---- Split per scenario ----
    shuffled = scenarios[:]
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * SPLIT["train"])
    n_val = int(n * SPLIT["val"])
    assign = {sc: ("train" if i < n_train
                   else "val" if i < n_train + n_val
                   else "test")
              for i, sc in enumerate(shuffled)}

    for split in SPLIT:
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    stats = Counter()
    written_count = Counter()

    # ---- Pass 2: konversi dengan pemetaan per-scenario ----
    for k, sc in enumerate(scenarios, 1):
        split = assign[sc]
        prefix = "_".join(sc.relative_to(SRC_ROOT).parts)

        with open(sc / "_annotations.coco.json", encoding="utf-8") as f:
            coco = json.load(f)

        id2name = real_categories(coco)   # pemetaan LOKAL scenario ini

        anns_by_img = {}
        for a in coco.get("annotations", []):
            anns_by_img.setdefault(a["image_id"], []).append(a)

        for img in coco.get("images", []):
            src_img = sc / img["file_name"]
            if not src_img.exists():
                stats["gambar_hilang"] += 1
                continue
            W, H = img["width"], img["height"]
            new_name = f"{prefix}_{img['file_name']}"
            dst_img = OUT_ROOT / "images" / split / new_name

            if COPY_MODE == "move":
                shutil.move(str(src_img), str(dst_img))
            else:
                shutil.copy2(str(src_img), str(dst_img))

            lines = []
            for a in anns_by_img.get(img["id"], []):
                name = id2name.get(a["category_id"])
                if name is None:                 # dummy -> lewati
                    stats["anotasi_dummy_dilewati"] += 1
                    continue
                if name not in name_to_idx:      # di luar KEEP_CLASSES
                    stats["anotasi_kelas_difilter"] += 1
                    continue
                x, y, w, h = a["bbox"]
                cx = min(max((x + w / 2) / W, 0), 1)
                cy = min(max((y + h / 2) / H, 0), 1)
                nw = min(max(w / W, 0), 1)
                nh = min(max(h / H, 0), 1)
                cls = name_to_idx[name]
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                written_count[name] += 1

            label_path = OUT_ROOT / "labels" / split / (Path(new_name).stem + ".txt")
            label_path.write_text("\n".join(lines), encoding="utf-8")
            stats[f"gambar_{split}"] += 1

        if k % 50 == 0:
            print(f"  ...{k}/{len(scenarios)} scenario diproses")

    # ---- data.yaml ----
    yaml_lines = [
        f"path: {OUT_ROOT.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for i, name in enumerate(class_names):
        yaml_lines.append(f"  {i}: {name}")
    (OUT_ROOT / "data.yaml").write_text("\n".join(yaml_lines), encoding="utf-8")

    print("\n===== SELESAI =====")
    for key, v in sorted(stats.items()):
        print(f"{key}: {v}")
    print("\nInstance tertulis per kelas:")
    for name, cnt in written_count.most_common():
        print(f"  {name}: {cnt}")
    print(f"\ndata.yaml tersimpan di: {(OUT_ROOT / 'data.yaml').resolve()}")


if __name__ == "__main__":
    main()