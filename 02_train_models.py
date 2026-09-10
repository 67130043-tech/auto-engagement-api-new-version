# -*- coding: utf-8 -*-
from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

print("STEP 3: Load cleaned datasets")
crm = pd.read_csv(OUT / "crm_cleaned.csv", encoding="utf-8-sig")
wongnai = pd.read_csv(OUT / "wongnai_cleaned.csv", encoding="utf-8-sig")

print("STEP 4: Train Sentiment Model from wongnai.csv")
# ใช้ sample แบบ stratified เพื่อให้รันเร็วใน Colab และยังรักษาสัดส่วน label
#
# FIX (พบระหว่างทำ 07_evaluate_baseline.py): บรรทัดเดิม
#   wongnai.groupby("sentiment", group_keys=False).apply(lambda x: x.sample(...))
# ใน pandas เวอร์ชันที่ใช้รันจริงตอนนี้ (2.x) การ apply คืนค่าที่ "column ที่ใช้
# groupby" (sentiment) หายไปจาก DataFrame ผลลัพธ์ ทำให้บรรทัดถัดไปที่อ้าง
# wongnai["sentiment"] พัง (KeyError) ถ้ารันสคริปต์นี้ซ้ำในเครื่องนี้/เวอร์ชัน
# pandas นี้ แก้โดย sample ทีละกลุ่มเองแทน (ได้ผลลัพธ์เทียบเท่ากันทุกประการ เพราะ
# ใช้ random_state=42 เดียวกัน แค่เปลี่ยนวิธีเขียนโค้ดให้ไม่พึ่ง behavior ที่เปลี่ยนไป)
_parts = [grp.sample(min(len(grp), 2500), random_state=42) for _, grp in wongnai.groupby("sentiment")]
wongnai = pd.concat(_parts)
X = wongnai["clean_text"].astype(str)
y = wongnai["sentiment"].astype(str)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

# ---------------------------------------------------------------------------
# FIX (พบระหว่างพัฒนาความแม่นยำของ sentiment_confidence/category_confidence):
# LinearSVC ไม่มี predict_proba() ในตัว (มีแค่ decision_function ซึ่งเป็นระยะห่าง
# จาก decision boundary ดิบๆ ไม่ใช่ความน่าจะเป็น) เดิม app/engine.py เลี่ยงปัญหานี้
# ด้วยการคำนวณ confidence เองแบบ sigmoid/softmax ทับ decision_function ซึ่งเป็นแค่
# heuristic ที่ "ดูเหมือน" ความน่าจะเป็น แต่ไม่ได้ผ่านการสอบเทียบ (calibration) กับ
# ข้อมูลจริงเลยว่าเวลาโมเดลบอกว่า "มั่นใจ 90%" มันถูกจริง ~90% ของเวลาไหม
#
# แก้โดยห่อ Pipeline เดิมด้วย CalibratedClassifierCV (Platt scaling, method=
# "sigmoid" — เหมาะกับ SVM โดยเฉพาะ เป็นวิธีที่ Platt ออกแบบมาให้ SVM ตั้งแต่แรก)
# ซึ่งจะ:
#   1. แบ่งข้อมูล train เป็น 5 folds (cv=5)
#   2. เทรน Pipeline (tfidf+SVM) ซ้ำ 5 รอบ แต่ละรอบ fold ที่เหลือใช้ fit sigmoid
#      function แปลง decision_function -> ความน่าจะเป็นที่สอบเทียบแล้วจริงๆ
#   3. ตอนพยากรณ์จริง เฉลี่ยผลจากทั้ง 5 sigmoid function เข้าด้วยกัน
# ผลคือโมเดลที่ได้มี .predict_proba() ใช้งานได้จริง (ไม่ใช่ของปลอมแบบเดิม) —
# ดู app/engine.py::predict_with_confidence() ที่แก้คู่กันให้ใช้ predict_proba()
# นี้โดยตรงเมื่อมี แทนสูตร sigmoid/softmax เดิม (ซึ่งยังเก็บไว้เป็น fallback เผื่อ
# โหลด model.joblib รุ่นเก่าที่ยังไม่ได้ calibrate)
# ---------------------------------------------------------------------------
sentiment_base = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5), min_df=2, max_features=20000)),
    ("clf", LinearSVC(class_weight="balanced"))
])
sentiment_model = CalibratedClassifierCV(sentiment_base, method="sigmoid", cv=5)
sentiment_model.fit(X_train, y_train)
sent_pred = sentiment_model.predict(X_test)
sent_acc = accuracy_score(y_test, sent_pred)
sent_report = classification_report(y_test, sent_pred, output_dict=True, zero_division=0)

print("Sentiment accuracy (calibrated):", round(sent_acc, 4))
joblib.dump(sentiment_model, MODELS / "sentiment_model.joblib")

print("\nSTEP 5: Train Category Model from CRM")
X = crm["clean_text"].astype(str)
y = crm["Category"].astype(str)
# category อาจบางคลาสน้อย ใช้ stratify เฉพาะถ้าทำได้
stratify = y if y.value_counts().min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=stratify)

category_base = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5), min_df=1, max_features=20000)),
    ("clf", LinearSVC(class_weight="balanced"))
])
category_model = CalibratedClassifierCV(category_base, method="sigmoid", cv=5)
category_model.fit(X_train, y_train)
cat_pred = category_model.predict(X_test)
cat_acc = accuracy_score(y_test, cat_pred)
cat_report = classification_report(y_test, cat_pred, output_dict=True, zero_division=0)

print("Category accuracy (calibrated):", round(cat_acc, 4))
joblib.dump(category_model, MODELS / "category_model.joblib")

metrics = {
    "sentiment_accuracy": float(sent_acc),
    "category_accuracy": float(cat_acc),
    "sentiment_report": sent_report,
    "category_report": cat_report,
}
with open(OUT / "model_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("Saved models to:", MODELS)
print("Saved metrics to:", OUT / "model_metrics.json")
