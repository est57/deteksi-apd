# Download dataset dari Kaggle ke folder data/css-data
# Jalankan: python download_data.py

import shutil
import kagglehub

folder = kagglehub.dataset_download("snehilsanyal/construction-site-safety-image-dataset-roboflow")
shutil.copytree(folder + "/css-data", "data/css-data", dirs_exist_ok=True)

print("Selesai! Dataset ada di data/css-data")
