import os
import pandas as pd
from zipfile import ZipFile

def load_data():
    data_dir = "Data"
    zip_path = os.path.join(data_dir, "imdb-dataset-of-50k-movie-reviews.zip")
    csv_path = os.path.join(data_dir, "IMDB Dataset.csv")

    # تأكد إن الفولدر موجود
    os.makedirs(data_dir, exist_ok=True)

    # لو الـ zip مش موجود → نزّله
    if not os.path.exists(zip_path):
        os.system(
            "kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews "
            f"-p {data_dir}"
        )

    # لو الـ CSV مش موجود → فك الضغط
    if not os.path.exists(csv_path):
        with ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(data_dir)

    # اقرأ الداتا
    df = pd.read_csv(csv_path)

    return df
