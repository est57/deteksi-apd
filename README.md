# Computer Vision for Construction Safety

Web untuk mendeteksi pemakaian APD di area konstruksi, dibuat dengan YOLO11 dan Streamlit. Bahan workshop "Computer Vision for Construction Safety".

Panduan lengkap langkah demi langkah ada di `guidebook/Guidebook_CV_Construction_Safety.pdf`.

Model mengenali 10 kelas: Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest, Mask, NO-Mask, Person, Safety Cone, machinery, dan vehicle. Kelas berawalan NO- ditandai kotak merah.

## Dataset

[Construction Site Safety Image Dataset](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow) dari Kaggle. Jumlahnya 2.801 gambar: 2.605 train, 114 valid, 82 test.

## Isi folder

| File | Kegunaan |
| --- | --- |
| `app/app.py` | Web deteksi |
| `notebooks/training_yolo_colab.ipynb` | Training di Google Colab |
| `download_data.py` | Download dataset dari Kaggle |
| `train.py` | Training di laptop |
| `data/data.yaml` | Konfigurasi dataset |
| `models/best.pt` | Model hasil training |
| `requirements.txt` | Daftar library |

## Menjalankan web

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/app.py
```

Pengguna Windows mengganti baris kedua dengan `.venv\Scripts\activate`. Web terbuka di http://localhost:8501.

## Training ulang

Cara paling mudah lewat Google Colab: upload `notebooks/training_yolo_colab.ipynb`, pilih GPU T4, lalu jalankan semua sel.

Kalau ingin training di laptop, jalankan `python download_data.py` lalu `python train.py`. Salin `best.pt` hasilnya ke folder `models/`.
