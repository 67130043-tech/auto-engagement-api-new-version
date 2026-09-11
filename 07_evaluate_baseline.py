# -*- coding: utf-8 -*-
"""
ประเมินผล baseline ของระบบปัจจุบัน (ก่อนแก้ไขใดๆ) แยก 2 ระดับ:
  1. ML model ดิบๆ (ไม่ผ่าน keyword override) - เพื่อดูว่าโมเดลเองแม่นแค่ไหน
  2. Full pipeline (ผ่าน keyword_sentiment_override / keyword_category_override
     เหมือนที่ระบบจริงใช้ใน engine.py) - เพื่อดูความแม่นยำที่ลูกค้าจะเจอจริง

ใช้ train_test_split แบบเดียวกับ 02_train_models.py เป๊ะ (random_state=42,
test_size=0.20, stratify เดียวกัน) เพื่อให้ชุด test เป็นแถวที่ "โมเดลไม่เคยเห็นตอนเทรน"
จริงๆ ไม่ใช่เอาข้อมูลที่ใช้เทรนไปวัดซ้ำ (data leakage)
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.engine import keyword_sentiment_override, keyword_category_override, map_to_ml_category

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
MODELS = BASE / "models"

sentiment_model = joblib.load(MODELS / "sentiment_model.joblib")
category_model = joblib.load(MODELS / "category_model.joblib")

results = {}

# ---------------------------------------------------------------------------
# 1) SENTIMENT (wongnai.csv) — held-out test split เดียวกับตอนเทรน
# ---------------------------------------------------------------------------
print("=" * 70)
print("SENTIMENT — held-out test set (wongnai.csv, เดียวกับตอนเทรน)")
print("=" * 70)

wongnai = pd.read_csv(OUT / "wongnai_cleaned.csv", encoding="utf-8-sig")
# หมายเหตุ: เดิม 02_train_models.py ใช้
#   wongnai.groupby("sentiment", group_keys=False).apply(lambda x: x.sample(...))
# ซึ่งใน pandas เวอร์ชันปัจจุบัน (ที่รันในเครื่องนี้) มี behavior เปลี่ยนไป ทำให้
# คอลัมน์ที่ใช้ groupby ("sentiment") หายไปจากผลลัพธ์ (KeyError ตอนอ้างอิงทีหลัง)
# แก้โดย sample ทีละกลุ่มเองแทน ได้ผลลัพธ์เทียบเท่ากันทุกประการ (random_state=42 เดียวกัน)
_parts = [grp.sample(min(len(grp), 2500), random_state=42) for _, grp in wongnai.groupby("sentiment")]
wongnai = pd.concat(_parts)
X = wongnai["clean_text"].astype(str)
y = wongnai["sentiment"].astype(str)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ระดับ 1: ML ดิบ
pred_ml = sentiment_model.predict(X_test)
acc_ml = accuracy_score(y_test, pred_ml)
f1_ml = f1_score(y_test, pred_ml, average="macro", zero_division=0)
print(f"\n[ML ดิบ]      accuracy={acc_ml:.4f}  macro-F1={f1_ml:.4f}")

# ระดับ 2: full pipeline (ผ่าน keyword override เหมือนของจริง)
pred_full = [keyword_sentiment_override(t, m) for t, m in zip(X_test, pred_ml)]
acc_full = accuracy_score(y_test, pred_full)
f1_full = f1_score(y_test, pred_full, average="macro", zero_division=0)
print(f"[Full system] accuracy={acc_full:.4f}  macro-F1={f1_full:.4f}")

print("\nclassification_report (Full system):")
print(classification_report(y_test, pred_full, zero_division=0))

results["sentiment"] = {
    "n_test": len(y_test),
    "ml_only": {"accuracy": acc_ml, "macro_f1": f1_ml},
    "full_system": {"accuracy": acc_full, "macro_f1": f1_full},
    "full_system_report": classification_report(y_test, pred_full, zero_division=0, output_dict=True),
    "confusion_matrix_labels": sorted(y_test.unique().tolist()),
    "confusion_matrix": confusion_matrix(y_test, pred_full, labels=sorted(y_test.unique().tolist())).tolist(),
}

# ---------------------------------------------------------------------------
# 2) CATEGORY (CRM dataset) — held-out test split เดียวกับตอนเทรน
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("CATEGORY — held-out test set (Dataset_Restaurant_CRM_4000_Rows, เดียวกับตอนเทรน)")
print("=" * 70)

# FIX (11 ก.ย.): ใช้ category_dataset_v2.csv (สร้างจากรีวิว Wongnai จริง ไม่ซ้ำกัน)
# แทน crm_cleaned.csv เดิม (ซ้ำกันแค่ 85 ประโยคจาก 4,000 แถว — ดูรายละเอียดใน
# 09_build_category_dataset_from_wongnai.py) เพื่อให้ตัวเลข accuracy ที่วัดได้
# สะท้อนความสามารถ generalize จริง ไม่ใช่การจำข้อความที่เคยเห็นตอนเทรนซ้ำ
_category_dataset_path = OUT / "category_dataset_v2.csv"
if _category_dataset_path.exists():
    crm = pd.read_csv(_category_dataset_path, encoding="utf-8-sig")
else:
    crm = pd.read_csv(OUT / "crm_cleaned.csv", encoding="utf-8-sig")
X = crm["clean_text"].astype(str)
y = crm["Category"].astype(str)
stratify = y if y.value_counts().min() >= 2 else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=stratify
)

# เช็ค data leakage แบบตรงๆ: มีแถวไหนใน test ที่ข้อความซ้ำกับใน train เป๊ะไหม
overlap = set(X_test) & set(X_train)
print(f"\n[เช็ค leakage] จำนวนข้อความใน test ที่ text ซ้ำกับใน train เป๊ะๆ: {len(overlap)} จาก {len(X_test)}")

pred_ml = category_model.predict(X_test)
acc_ml = accuracy_score(y_test, pred_ml)
f1_ml = f1_score(y_test, pred_ml, average="macro", zero_division=0)
print(f"\n[ML ดิบ]      accuracy={acc_ml:.4f}  macro-F1={f1_ml:.4f}")

pred_detail = [keyword_category_override(t, m) for t, m in zip(X_test, pred_ml)]
pred_full = [map_to_ml_category(d, m) for d, m in zip(pred_detail, pred_ml)]
acc_full = accuracy_score(y_test, pred_full)
f1_full = f1_score(y_test, pred_full, average="macro", zero_division=0)
print(f"[Full system] accuracy={acc_full:.4f}  macro-F1={f1_full:.4f}")

print("\nclassification_report (Full system):")
print(classification_report(y_test, pred_full, zero_division=0))

results["category"] = {
    "n_test": len(y_test),
    "n_exact_text_overlap_with_train": len(overlap),
    "ml_only": {"accuracy": acc_ml, "macro_f1": f1_ml},
    "full_system": {"accuracy": acc_full, "macro_f1": f1_full},
    "full_system_report": classification_report(y_test, pred_full, zero_division=0, output_dict=True),
}

with open(OUT / "baseline_evaluation.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\nบันทึกผลละเอียดไว้ที่:", OUT / "baseline_evaluation.json")
