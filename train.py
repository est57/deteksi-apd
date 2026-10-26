# Training model di laptop sendiri (alternatif dari Google Colab)
# Jalankan: python train.py

from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="data/data.yaml",
    epochs=30,
    imgsz=640,
    name="apd",
    # device="mps",  # hapus tanda # jika memakai Mac M1/M2/M3/M4/M5 agar lebih cepat
)

# Hasil terbaik tersimpan di: runs/detect/apd/weights/best.pt
# Salin file tersebut ke folder models/
