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
from sklearn.metrics import accuracy_score, classification_report, f1_score

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

print("STEP 3: Load cleaned datasets")
# ---------------------------------------------------------------------------
# FIX (พบ 11 ก.ย.: Dataset_Restaurant_CRM_4000_Rows.xlsx เดิม 4,000 แถวมีข้อความ
# ไม่ซ้ำกันจริงแค่ 85 ประโยค ทำให้ train/test split แบ่งประโยคเดียวกันเข้าทั้งสองฝั่ง
# category_model จึง "จำ" แทนที่จะ "generalize" — accuracy 100% ที่วัดได้เดิมเป็น
# ตัวเลขหลอก) เปลี่ยนมาใช้ data/category_dataset_v2.csv แทน ซึ่งสร้างจากรีวิว
# Wongnai จริง (ไม่ซ้ำกัน) ผ่านการติด label ด้วยระบบ keyword ที่ปรับแต่งไว้แล้ว
# (ดู 09_build_category_dataset_from_wongnai.py สำหรับรายละเอียดวิธีสร้าง) —
# fallback ไปที่ crm_cleaned.csv เดิมถ้ายังไม่เคยรันสคริปต์สร้าง dataset ใหม่
#
# FIX (ต.ค. 2569): เดิมบรรทัดนี้ชี้ไปที่ outputs/category_dataset_v2.csv ซึ่งไม่ได้
# commit ไว้ใน repo (มีแค่ data/category_dataset_v2.csv) — clone ใหม่แล้วรันสคริปต์นี้
# เงื่อนไข .exists() ด้านล่างจะเป็น False แล้ว fallback ไปใช้ crm_cleaned.csv เดิมที่มี
# ปัญหา data leakage กลับมาแบบเงียบๆ จึงชี้ไปที่ data/ แทน
# ---------------------------------------------------------------------------
category_dataset_v2_path = BASE / "data" / "category_dataset_v2.csv"
if category_dataset_v2_path.exists():
    crm = pd.read_csv(category_dataset_v2_path, encoding="utf-8-sig")
    print(f"  ใช้ category dataset ใหม่: {category_dataset_v2_path.name} ({len(crm)} แถว, ไม่ซ้ำกัน)")
else:
    crm = pd.read_csv(OUT / "crm_cleaned.csv", encoding="utf-8-sig")
    print(f"  [คำเตือน] ไม่พบ {category_dataset_v2_path.name} — ใช้ crm_cleaned.csv เดิม (มีปัญหาข้อความซ้ำ)")

print("STEP 4: Train Sentiment Model")
# ---------------------------------------------------------------------------
# FIX (ต.ค. 2569): ใช้ data/sentiment_dataset_v2.csv (Wisesight ที่คนติดป้าย + รีวิว
# Wongnai ป้ายจากดาว — ดู 10_build_sentiment_dataset.py) แทนการสุ่มจาก Wongnai อย่าง
# เดียว และใช้คอลัมน์ split ในไฟล์ ทุกโมเดล (TF-IDF+SVM, WangchanBERTa ใน
# 11_finetune_wangchanberta.ipynb) จึงวัดผลบนแถวทดสอบชุดเดียวกัน
# ถ้ายังไม่มีไฟล์ จะ fallback ไปใช้วิธีเดิม (สุ่ม Wongnai 2,500 รีวิวต่อป้าย)
# ---------------------------------------------------------------------------
sentiment_dataset_path = BASE / "data" / "sentiment_dataset_v2.csv"
if sentiment_dataset_path.exists():
    sent_df = pd.read_csv(sentiment_dataset_path, encoding="utf-8-sig")
    print(f"  ใช้ sentiment dataset ใหม่: {sentiment_dataset_path.name} ({len(sent_df)} แถว)")
    tr, te = sent_df[sent_df["split"] == "train"], sent_df[sent_df["split"] == "test"]
    X_train, y_train = tr["clean_text"].astype(str), tr["sentiment"].astype(str)
    X_test, y_test = te["clean_text"].astype(str), te["sentiment"].astype(str)
    sent_src_test = te["source"].astype(str)
else:
    print(f"  [คำเตือน] ไม่พบ {sentiment_dataset_path.name} — ใช้รีวิว Wongnai อย่างเดียว")
    wongnai = pd.read_csv(OUT / "wongnai_cleaned.csv", encoding="utf-8-sig")
    _parts = [grp.sample(min(len(grp), 2500), random_state=42) for _, grp in wongnai.groupby("sentiment")]
    wongnai = pd.concat(_parts)
    X = wongnai["clean_text"].astype(str)
    y = wongnai["sentiment"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    sent_src_test = pd.Series(["wongnai"] * len(X_test), index=X_test.index)

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

sent_macro_f1 = f1_score(y_test, sent_pred, average="macro", zero_division=0)
print("Sentiment accuracy (calibrated):", round(sent_acc, 4), " macro-F1:", round(sent_macro_f1, 4))
sent_by_source = {}
for source in sorted(sent_src_test.unique()):
    m = (sent_src_test == source).values
    sent_by_source[source] = {
        "n_test": int(m.sum()),
        "accuracy": float(accuracy_score(y_test[m], sent_pred[m])),
        "macro_f1": float(f1_score(y_test[m], sent_pred[m], average="macro", zero_division=0)),
    }
    print(f"  source={source:<10} n={m.sum():5d}  accuracy={sent_by_source[source]['accuracy']:.4f}"
          f"  macro-F1={sent_by_source[source]['macro_f1']:.4f}")
joblib.dump(sentiment_model, MODELS / "sentiment_model.joblib")

print("\nSTEP 5: Train Category Model")
X = crm["clean_text"].astype(str)
y = crm["Category"].astype(str)
src = crm["source"].astype(str) if "source" in crm.columns else pd.Series(["unknown"] * len(crm), index=crm.index)
if "split" in crm.columns:
    # ใช้ split ที่กำหนดไว้ในไฟล์ (ดู 09_build_category_dataset_from_wongnai.py)
    is_test = (crm["split"] == "test").values
    X_train, X_test = X[~is_test], X[is_test]
    y_train, y_test = y[~is_test], y[is_test]
    src_train, src_test = src[~is_test], src[is_test]
else:
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test, src_train, src_test = train_test_split(
        X, y, src, test_size=0.20, random_state=42, stratify=stratify)

category_base = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5), min_df=1, max_features=20000)),
    ("clf", LinearSVC(class_weight="balanced"))
])
category_model = CalibratedClassifierCV(category_base, method="sigmoid", cv=5)
category_model.fit(X_train, y_train)
cat_pred = category_model.predict(X_test)
cat_acc = accuracy_score(y_test, cat_pred)
cat_report = classification_report(y_test, cat_pred, output_dict=True, zero_division=0)

cat_macro_f1 = f1_score(y_test, cat_pred, average="macro", zero_division=0)
print("Category accuracy (calibrated):", round(cat_acc, 4))
print("Category macro-F1:", round(cat_macro_f1, 4))

# ---------------------------------------------------------------------------
# รายงานผลแยกตามที่มาของป้ายกำกับ (คอลัมน์ source ใน category_dataset_v2.csv):
#   wongnai_real           = ติดป้ายด้วยกฎ keyword ของระบบ
#   wongnai_pattern        = ติดป้ายด้วย pattern เฉพาะหมวด (data/wongnai.csv)
#   wongnai_corpus_pattern = ติดป้ายด้วย pattern เฉพาะหมวด (Wongnai corpus ฉบับเต็ม)
# ทุกแถวเป็นรีวิวจริง ไม่มีประโยคสังเคราะห์ แต่ป้ายกำกับยังมาจากกฎ (weak supervision)
# ตัวเลขนี้จึงวัด "ความสอดคล้องกับกฎ" ไม่ใช่ความถูกต้องเทียบกับคน — ตัวเลขหลักของงานวิจัย
# ต้องมาจากชุดทดสอบมาตรฐานที่คนติดป้าย (Gold Standard)
#
# วิธีจัดการความไม่สมดุลที่เหลือ (สัดส่วนหมวดใหญ่สุด:เล็กสุด ~1.5:1) ใช้
# class_weight="balanced" ตามเดิม เพราะทดสอบ 5-fold CV แล้ว oversampling / isotonic
# ไม่ได้ดีขึ้นจริง (macro-F1 ต่างกันไม่เกิน 0.001)
# ---------------------------------------------------------------------------
print("Per-class F1:")
for label in sorted(y_test.unique()):
    st = cat_report.get(label, {})
    print(f"  {label:<24} f1={st.get('f1-score', 0):.3f}  (test n={int(st.get('support', 0))})")
cat_by_source = {}
for source in sorted(src_test.unique()):
    m = (src_test == source).values
    cat_by_source[source] = {
        "n_test": int(m.sum()),
        "accuracy": float(accuracy_score(y_test[m], cat_pred[m])),
    }
    print(f"  source={source:<24} n={m.sum():5d}  accuracy={cat_by_source[source]['accuracy']:.4f}")
joblib.dump(category_model, MODELS / "category_model.joblib")

metrics = {
    "sentiment_accuracy": float(sent_acc),
    "sentiment_macro_f1": float(sent_macro_f1),
    "sentiment_by_source": sent_by_source,
    "category_accuracy": float(cat_acc),
    "category_macro_f1": float(cat_macro_f1),
    "category_accuracy_by_source": cat_by_source,
    "sentiment_report": sent_report,
    "category_report": cat_report,
}
with open(OUT / "model_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("Saved models to:", MODELS)
print("Saved metrics to:", OUT / "model_metrics.json")
