# 🎬 Sentiment Analysis using LSTM

A Deep Learning project for performing sentiment analysis on the IMDB movie reviews dataset using an LSTM neural network built with TensorFlow/Keras.

---

## 📌 Project Overview

This project builds and trains a Long Short-Term Memory (LSTM) model to classify movie reviews as:

- ✅ Positive  
- ❌ Negative  

The dataset used is the **IMDB 50K Movie Reviews Dataset**.

---

## 🧠 Model Architecture

- Text Preprocessing (Tokenization & Padding)
- Embedding Layer
- LSTM Layer
- Dense Layer (Sigmoid Activation)

---

## 📂 Project Structure
```
├── Data/               # ملفات البيانات (IMDB Dataset Zip)
├── models/             # The Best Model
├── models_metrics/     # نتائج تقييم النماذج (History & Logs)
├── src/                # الكود المصدري الأساسي
│   ├── data_loader.py  # قراءة وتحميل البيانات
│   ├── preprocessing.py# تنظيف ومعالجة النصوص
│   ├── model.py        # تعريف هيكل النموذج
│   ├── train.py        # كود التدريب الأساسي
│   └── prediction.py   # كود التوقع على بيانات جديدة
├── main.py             # نقطة الانطلاق لتشغيل المشروع
├──LICENSE              # Apache version 2.0
├──README.md            # توضيح محتويات المشروع
└── requirment.txt      # المكتبات المطلوبة للتشغيل

```
## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/omarelnabawi/Sentiment_analysis-LSTM-.git
cd Sentiment_analysis-LSTM-
```
### 2️⃣ Create virtual environment
```bash
python -m venv nlp_env
```
### 3️⃣ Install dependencies
```bash
pip install -r requirement.txt
```
##  Training the Model

```bash
python src/train.py
```
### The model will:

- Train on the dataset

- Save the trained model inside models/

- Save metrics and logs
---
### 🚀 Running the Project via Streamlit

**To launch the app and allow others to test your model:**
```bash
streamlit run main.py
```

- This will open a local web interface where you can input movie reviews and see predicted sentiment (Positive / Negative).


## 📊 Model Performance

### 7 epochs model output:

- Validation Accuracy: ~90%

- Loss: ~0.28

**Notes 1:** you can use our Default model with `7 epochs` or choose the number of epochs you want but be carful `⚠️ it run in your local machine. `

**Notes 2:** Results vary depending on epochs and hyperparameters.

## 🛠 Technologies Used

- Python

- TensorFlow / Keras

- NumPy

- Pandas

- Scikit-learn
---
## 📈 Future Improvements

- Add EarlyStopping

- Add ModelCheckpoint

- Deploy using Streamlit (coming in version 2 )

- Add TensorBoard visualization

- Convert to Transformer-based model (BERT)

## 👨‍💻 Author

**Omar Nabawi**
- AI Engineer | Machine Learning Engineer
