# Vehicle Detection & Motorcyclist Risk Scoring

Pipeline deteksi kendaraan berbasis YOLO untuk sistem peringatan bahaya bagi pengendara motor (kamera depan/dashcam), dilengkapi tracking, deteksi lampu belakang, ROI jalur dinamis (YOLOP), dan risk scoring.

> Catatan: repo ini berisi **kode sumber + 3 model terlatih terbaik (best.pt)**. Dataset penuh, checkpoint training lain (`last.pt`, log, gambar val/train batch), hasil video (`risk_output/`), dan virtual environment (`yolo-env/`) sengaja **tidak** disertakan karena ukurannya besar (ratusan MB–puluhan GB) dan tidak cocok disimpan di Git. Simpan/host secara terpisah (mis. Google Drive, Kaggle, Hugging Face) bila perlu dibagikan.

## Model terlatih (included)

| Path | Model | Ukuran | Dipakai di |
|---|---|---|---|
| `runs/detect/motorcyclist-hazard/exp26/weights/best.pt` | YOLO26l — deteksi kendaraan/bahaya (model utama) | ~51 MB | `risk_dual_inference.py`, `dual_inference_YOLOP*.py` (`VEHICLE_MODEL`) |
| `runs/detect/motorcyclist-hazard/expv8/weights/best.pt` | YOLOv8l — deteksi kendaraan (pembanding) | ~84 MB | `model_comparison.py` |
| `runs/detect/taillight/exp1-yolov10/weights/best.pt` | YOLOv10l — deteksi lampu belakang (brake/sein) | ~50 MB | `risk_dual_inference.py`, `dual_inference*.py` (`LIGHT_MODEL`) |

Struktur foldernya sengaja disamakan dengan path yang sudah ditulis di dalam skrip (`runs/detect/...`), jadi tidak perlu mengubah path apa pun di kode — cukup jalankan skrip dari dalam folder repo ini.

## Isi kode

| File | Fungsi |
|---|---|
| `merge_dataset.py` | Menggabungkan dataset RSUD20K + Roboflow "autonomous-vehicle-qvcsy" menjadi satu dataset YOLO (`merged_dataset/`) siap training. |
| `convert_tld.py` | Konversi dataset TLD-LOKI (COCO JSON per skenario, lampu belakang: brake/go/left/right) ke format YOLO, dengan pemetaan kelas per-skenario. |
| `model_comparison.py` | Evaluasi & perbandingan YOLOv8 vs YOLO26 pada test set (Mean IoU, Precision, Recall, mAP@50, mAP@50-95). |
| `graph_model_comparison.py` | Membuat grouped bar chart dari hasil `model_comparison.py`. |
| `roi_manual.py` | Utilitas klik 4 titik untuk menentukan ROI trapesium manual pada frame video. |
| `dual_inference.py` | Dual inference + tracking (kendaraan via BoT-SORT/ByteTrack, lampu belakang diasosiasikan per kendaraan) dengan smoothing status lampu. |
| `dual_inference_roi_manual.py` | Varian dual inference dengan filter ROI trapesium manual. |
| `dual_inference_YOLOP.py` / `dual_inference_YOLOP2.py` | Varian dual inference dengan ROI dinamis dari segmentasi drivable-area YOLOP (Wu dkk., 2021). |
| `risk_scoring.py` | Layer risk-scoring berbasis proxy visual 2D (laju pembesaran bbox, luas relatif, posisi lateral, bobot kelas) — tanpa data jarak nyata (LiDAR/radar). |
| `risk_dual_inference.py` | Pipeline utama (versi terbaru): deteksi kendaraan (YOLO26 + BoT-SORT) + deteksi lampu belakang (YOLOv10) + risk scoring (TTC looming, manuver lateral, bobot kelas, lampu) + deteksi kendaraan parkir + ROI dinamis (YOLOP) dengan fallback ke ROI manual. |
| `requirements.txt` | Daftar dependency Python. |
| `_botsort_runtime.yaml` | Konfigurasi custom tracker BoT-SORT yang dipakai saat runtime. |

## Setup

Gunakan Python 3.11 atau 3.12 (bukan 3.14 — PyTorch belum menyediakan wheel CUDA untuk 3.14 di Windows per pertengahan 2026).

```bash
# 1. Install torch + CUDA TERPISAH dulu (contoh untuk CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. Baru install sisanya
pip install -r requirements.txt
```

## Pemakaian

Jalankan dari dalam folder repo ini (model sudah ada di `runs/detect/...`, tinggal sediakan video sumbernya sendiri):

```bash
python risk_dual_inference.py --source path/ke/video.mp4 --save
python risk_dual_inference.py --source path/ke/video.mp4 --show --no-yolop
```

Untuk YOLOP (segmentasi drivable-area, dipakai untuk ROI dinamis) dibutuhkan file `yolop_end2end.pth` (~91 MB, model pretrained pihak ketiga — Wu dkk., 2021, arXiv:2108.11250) yang **tidak** ada di repo ini; skrip otomatis fallback ke ROI trapesium manual bila file ini tidak ditemukan.
