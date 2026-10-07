"""
Perbandingan YOLOv8 vs YOLO26 pada test set.
Menghitung: Mean IoU, Precision, Recall, mAP@50, mAP@50-95
lalu membuat grouped bar chart perbandingan.

Cara pakai (dari root project VEHICLE_DETECTION):
    python compare_models.py
Sesuaikan path MODELS dan DATA_YAML di bawah bila perlu.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

# ================== KONFIGURASI (sesuaikan) ==================
MODELS = {
    "YOLOv8l": r"runs/detect/motorcyclist-hazard/expv8/weights/best.pt",
    "YOLO26l": r"runs/detect/motorcyclist-hazard/exp26/weights/best.pt",
}
DATA_YAML = r"merged_dataset/data.yaml"
TEST_IMAGES_DIR = r"merged_dataset/images/test"
TEST_LABELS_DIR = r"merged_dataset/labels/test"
IMGSZ = 640
DEVICE = 0
CONF_THRES = 0.25   # confidence minimum saat menghitung mean IoU
IOU_MATCH = 0.5     # threshold IoU agar prediksi dianggap match dengan GT
# =============================================================


def box_iou(box1, box2):
    """IoU antara dua set box format xyxy. box1: (N,4), box2: (M,4) -> (N,M)"""
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    lt = np.maximum(box1[:, None, :2], box2[None, :, :2])
    rb = np.minimum(box1[:, None, 2:], box2[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def load_gt_boxes(label_path, img_w, img_h):
    """Baca label YOLO (class cx cy w h, ternormalisasi) -> xyxy piksel."""
    boxes = []
    if not os.path.exists(label_path):
        return np.zeros((0, 4))
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = map(float, parts[:5])
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([x1, y1, x2, y2])
    return np.array(boxes) if boxes else np.zeros((0, 4))


def compute_mean_iou(model):
    """Mean IoU dari pasangan prediksi-GT yang match (greedy, IoU >= IOU_MATCH)."""
    image_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        image_paths += glob.glob(os.path.join(TEST_IMAGES_DIR, ext))
    ious = []
    for img_path in image_paths:
        r = model.predict(img_path, imgsz=IMGSZ, conf=CONF_THRES,
                          device=DEVICE, verbose=False)[0]
        pred = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        h, w = r.orig_shape
        stem = os.path.splitext(os.path.basename(img_path))[0]
        gt = load_gt_boxes(os.path.join(TEST_LABELS_DIR, stem + ".txt"), w, h)
        if len(pred) == 0 or len(gt) == 0:
            continue
        iou_mat = box_iou(pred, gt)
        # greedy matching: pasangan IoU tertinggi dulu
        while iou_mat.size and iou_mat.max() >= IOU_MATCH:
            i, j = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
            ious.append(iou_mat[i, j])
            iou_mat = np.delete(np.delete(iou_mat, i, axis=0), j, axis=1)
    return float(np.mean(ious)) if ious else 0.0


def main():
    metric_names = ["Mean IoU", "Precision", "Recall", "mAP@50", "mAP@50-95"]
    results = {}

    for name, weights in MODELS.items():
        print(f"\n===== Evaluasi {name} =====")
        model = YOLO(weights)
        m = model.val(data=DATA_YAML, split="test", imgsz=IMGSZ,
                      batch=16, device=DEVICE, verbose=False)
        print("Menghitung mean IoU (prediksi per gambar)...")
        miou = compute_mean_iou(model)
        results[name] = [miou, m.box.mp, m.box.mr, m.box.map50, m.box.map]
        print(f"{name}: IoU={miou:.4f}  P={m.box.mp:.4f}  R={m.box.mr:.4f}  "
              f"mAP@50={m.box.map50:.4f}  mAP@50-95={m.box.map:.4f}")

    # ------------------ Bar chart ------------------
    labels = list(results.keys())
    x = np.arange(len(metric_names))
    width = 0.35
    colors = ["#4C72B0", "#DD8452"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(labels):
        offset = (i - (len(labels) - 1) / 2) * width
        bars = ax.bar(x + offset, results[name], width, label=name,
                      color=colors[i % len(colors)])
        ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=2)

    ax.set_ylabel("Score")
    ax.set_title("Perbandingan Metrik: YOLOv8 vs YOLO26 (Test Set)")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=200)
    print("\nGrafik tersimpan: model_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()