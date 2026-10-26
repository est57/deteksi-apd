# Web deteksi APD. Cara menjalankan: streamlit run app/app.py

import os
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from inference_sdk import InferenceHTTPClient

MODEL_PATH = "models/best.pt"
PELANGGARAN = ["NO-Hardhat", "NO-Mask", "NO-Safety Vest"]


@st.cache_resource
def muat_model():
    return YOLO(MODEL_PATH)


def deteksi_lokal(gambar, conf):
    hasil = muat_model()(gambar, conf=conf, verbose=False)[0]
    deteksi = []
    for box in hasil.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        nama = hasil.names[int(box.cls)]
        deteksi.append((nama, float(box.conf), x1, y1, x2, y2))
    return deteksi


def deteksi_roboflow(gambar, conf, api_key, model_id):
    client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=api_key)
    hasil = client.infer(Image.fromarray(gambar), model_id=model_id)
    deteksi = []
    for p in hasil["predictions"]:
        if p["confidence"] < conf:
            continue
        x1, y1 = p["x"] - p["width"] / 2, p["y"] - p["height"] / 2
        x2, y2 = p["x"] + p["width"] / 2, p["y"] + p["height"] / 2
        deteksi.append((p["class"], p["confidence"], x1, y1, x2, y2))
    return deteksi


def gambar_kotak(gambar, deteksi):
    gambar = gambar.copy()
    for nama, skor, x1, y1, x2, y2 in deteksi:
        warna = (255, 0, 0) if nama in PELANGGARAN else (0, 200, 0)
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        teks = f"{nama} {skor:.2f}"
        (w, h), _ = cv2.getTextSize(teks, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(gambar, p1, p2, warna, 2)
        cv2.rectangle(gambar, (p1[0], p1[1] - h - 6), (p1[0] + w, p1[1]), warna, -1)
        cv2.putText(gambar, teks, (p1[0], p1[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return gambar


def tampilkan_ringkasan(deteksi):
    jumlah = {}
    for d in deteksi:
        jumlah[d[0]] = jumlah.get(d[0], 0) + 1

    pelanggaran = {k: v for k, v in jumlah.items() if k in PELANGGARAN}
    if pelanggaran:
        for nama, n in pelanggaran.items():
            st.error(f"{n} pelanggaran: {nama}")
    else:
        st.success("Tidak ada pelanggaran APD")

    st.subheader("Objek terdeteksi")
    for nama, n in jumlah.items():
        st.write(f"- **{nama}**: {n}")


# Tampilan web

st.set_page_config(page_title="Deteksi APD")
st.title("Deteksi APD di Area Konstruksi")

st.sidebar.header("Pengaturan")
sumber = st.sidebar.radio("Sumber model", ["Model sendiri (best.pt)", "Roboflow API"])
conf = st.sidebar.slider("Confidence minimum", 0.1, 1.0, 0.4)

if sumber == "Roboflow API":
    api_key = st.sidebar.text_input("Roboflow API Key", type="password")
    model_id = st.sidebar.text_input("Model ID", placeholder="nama-project/1")
    if not api_key or not model_id:
        st.info("Isi API Key dan Model ID Roboflow di sidebar.")
        st.stop()
elif not os.path.exists(MODEL_PATH):
    st.error(f"File {MODEL_PATH} belum ada. Salin hasil training ke folder models/.")
    st.stop()


def deteksi(gambar):
    if sumber == "Roboflow API":
        return deteksi_roboflow(gambar, conf, api_key, model_id)
    return deteksi_lokal(gambar, conf)


tab_gambar, tab_kamera, tab_video = st.tabs(["Gambar", "Kamera", "Video"])

with tab_gambar:
    file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])
    if file:
        gambar = np.array(Image.open(file).convert("RGB"))
        hasil = deteksi(gambar)
        st.image(gambar_kotak(gambar, hasil))
        tampilkan_ringkasan(hasil)

with tab_kamera:
    foto = None
    if st.toggle("Nyalakan kamera"):
        foto = st.camera_input("Ambil foto")
    if foto:
        gambar = np.array(Image.open(foto).convert("RGB"))
        hasil = deteksi(gambar)
        st.image(gambar_kotak(gambar, hasil))
        tampilkan_ringkasan(hasil)

with tab_video:
    if sumber == "Roboflow API":
        st.info("Deteksi video hanya untuk model sendiri (best.pt).")
    else:
        video = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])
        if video:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp.write(video.read())
            cap = cv2.VideoCapture(tmp.name)
            layar = st.empty()
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                layar.image(gambar_kotak(frame, deteksi(frame)))
            cap.release()
