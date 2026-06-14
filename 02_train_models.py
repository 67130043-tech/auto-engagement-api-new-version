# -*- coding: utf-8 -*-
from pathlib import Path
import json
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
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
wongnai = wongnai.groupby("sentiment", group_keys=False).apply(lambda x: x.sample(min(len(x), 2500), random_state=42))
X = wongnai["clean_text"].astype(str)
y = wongnai["sentiment"].astype(str)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

sentiment_model = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5), min_df=2, max_features=20000)),
    ("clf", LinearSVC(class_weight="balanced"))
])
sentiment_model.fit(X_train, y_train)
sent_pred = sentiment_model.predict(X_test)
sent_acc = accuracy_score(y_test, sent_pred)
sent_report = classification_report(y_test, sent_pred, output_dict=True, zero_division=0)

print("Sentiment accuracy:", round(sent_acc, 4))
joblib.dump(sentiment_model, MODELS / "sentiment_model.joblib")

print("\nSTEP 5: Train Category Model from CRM")
X = crm["clean_text"].astype(str)
y = crm["Category"].astype(str)
# category อาจบางคลาสน้อย ใช้ stratify เฉพาะถ้าทำได้
stratify = y if y.value_counts().min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=stratify)

category_model = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5), min_df=1, max_features=20000)),
    ("clf", LinearSVC(class_weight="balanced"))
])
category_model.fit(X_train, y_train)
cat_pred = category_model.predict(X_test)
cat_acc = accuracy_score(y_test, cat_pred)
cat_report = classification_report(y_test, cat_pred, output_dict=True, zero_division=0)

print("Category accuracy:", round(cat_acc, 4))
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
