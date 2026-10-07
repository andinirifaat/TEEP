import matplotlib
matplotlib.use("QtAgg")        # backend Qt, tidak butuh Tcl/Tk
import matplotlib.pyplot as plt
import numpy as np

metric_names = ["Mean IoU", "Precision", "Recall", "mAP@50", "mAP@50-95"]
results = {
    "YOLOv8l": [0.9065, 0.8008, 0.8461, 0.8497, 0.6491],
    "YOLO26l": [0.9064, 0.7263, 0.8705, 0.8789, 0.6506],
}

labels = list(results.keys())
x = np.arange(len(metric_names))
width = 0.35
colors = ["#4C72B0", "#DD8452"]

fig, ax = plt.subplots(figsize=(10, 6))
for i, name in enumerate(labels):
    offset = (i - (len(labels) - 1) / 2) * width
    bars = ax.bar(x + offset, results[name], width, label=name, color=colors[i])
    ax.bar_label(bars, fmt="%.3f", fontsize=9, padding=2)

ax.set_ylabel("Score")
ax.set_title("Perbandingan Metrik: YOLOv8l vs YOLO26l (Test Set, 1082 gambar)")
ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=200)   # tetap simpan file juga
plt.show()                                      # tampilkan jendela grafik