# -*- coding: utf-8 -*-
"""
เช็คว่า "ค่าความมั่นใจ" (confidence) ที่โมเดลรายงานออกมา เชื่อถือได้แค่ไหนจริงๆ —
ไม่ใช่แค่ accuracy ของ label (วัดไปแล้วใน 07_evaluate_baseline.py) แต่เป็นคำถามที่
ต่างกัน: "ตอนโมเดลบอกว่ามั่นใจ 90% แม่นจริงประมาณ 90% ของเวลาไหม หรือมั่นใจมั่ว?"

เทียบ 2 วิธีคำนวณ confidence บนโมเดลชุดเดียวกัน (held-out test split เดิม):
  OLD = สูตร sigmoid/softmax ทับ decision_function() ของโมเดลรุ่นเก่า (ไม่ได้ calibrate)
  NEW = predict_proba() ของโมเดลที่ห่อด้วย CalibratedClassifierCV (Platt scaling)

วัด 2 ตัวชี้วัดมาตรฐานสำหรับ calibration quality:
  1. Brier score (ยิ่งน้อยยิ่งดี) — เฉลี่ยของ (confidence/100 - ถูก/ผิด)^2
  2. Expected Calibration Error (ECE, ยิ่งน้อยยิ่งดี) — แบ่ง confidence เป็น bin
     ละ 10% แล้ววัดส่วนต่างเฉลี่ย (ถ่วงน้ำหนักตามขนาด bin) ระหว่าง "confidence เฉลี่ย
     ที่โมเดลบอก" กับ "accuracy จริงในกลุ่มนั้น" — ถ้า ECE ต่ำ แปลว่าเลข confidence
     ที่ระบบโชว์ให้ผู้ใช้ดู เอาไปเชื่อได้จริง (เช่น กรองว่า confidence<70% ต้องมีคน
     รีวิวก่อนส่ง ก็จะกรองได้แม่นยำสมเหตุสมผลจริง)
"""
from pathlib import Path
import json
import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
MODELS = BASE / "models"
MODELS_OLD = BASE / "models_backup_uncalibrated"


def old_style_confidence(model, text: str):
    """สูตรเดิมของ app/engine.py ก่อนแก้ (sigmoid/softmax ทับ decision_function)"""
    pred = str(model.predict([text])[0])
    scores = np.atleast_1d(model.decision_function([text])[0])
    if scores.shape[0] == 1:
        confidence = float(1 / (1 + np.exp(-abs(scores[0])))) * 100
    else:
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        classes = list(model.named_steps["clf"].classes_) if hasattr(model, "named_steps") else list(model.classes_)
        idx = classes.index(pred)
        confidence = float(probs[idx]) * 100
    return pred, confidence


def new_style_confidence(model, text: str):
    pred = str(model.predict([text])[0])
    probs = model.predict_proba([text])[0]
    classes = list(model.classes_)
    idx = classes.index(pred)
    confidence = float(probs[idx]) * 100
    return pred, confidence


def brier_and_ece(y_true, preds, confidences, n_bins=10):
    correct = np.array([int(p == t) for p, t in zip(preds, y_true)])
    conf = np.array(confidences) / 100.0
    brier = float(np.mean((conf - correct) ** 2))

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(conf)
    bin_rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = conf[mask].mean()
        weight = mask.sum() / n
        ece += weight * abs(bin_acc - bin_conf)
        bin_rows.append({
            "range": f"{int(lo*100)}-{int(hi*100)}%",
            "n": int(mask.sum()),
            "avg_confidence_%": round(bin_conf * 100, 1),
            "actual_accuracy_%": round(bin_acc * 100, 1),
        })
    return brier, ece, bin_rows


def run_task(name, X_test, y_test, model_old, model_new):
    print("=" * 70)
    print(name)
    print("=" * 70)

    old_preds, old_confs = [], []
    new_preds, new_confs = [], []
    for t in X_test:
        p, c = old_style_confidence(model_old, t)
        old_preds.append(p); old_confs.append(c)
        p2, c2 = new_style_confidence(model_new, t)
        new_preds.append(p2); new_confs.append(c2)

    old_brier, old_ece, old_bins = brier_and_ece(y_test, old_preds, old_confs)
    new_brier, new_ece, new_bins = brier_and_ece(y_test, new_preds, new_confs)

    print(f"\n[OLD - uncalibrated sigmoid/softmax]  Brier={old_brier:.4f}  ECE={old_ece:.4f}")
    for r in old_bins:
        print(f"   confidence {r['range']:>9}  n={r['n']:4d}  avg_conf={r['avg_confidence_%']:5.1f}%  actual_acc={r['actual_accuracy_%']:5.1f}%")

    print(f"\n[NEW - CalibratedClassifierCV]        Brier={new_brier:.4f}  ECE={new_ece:.4f}")
    for r in new_bins:
        print(f"   confidence {r['range']:>9}  n={r['n']:4d}  avg_conf={r['avg_confidence_%']:5.1f}%  actual_acc={r['actual_accuracy_%']:5.1f}%")

    improvement_pct = (1 - new_ece / old_ece) * 100 if old_ece > 0 else 0.0
    print(f"\n=> ECE ลดลง {improvement_pct:.1f}% (ยิ่งลดมากยิ่งดี — แปลว่า confidence ที่รายงานเชื่อถือได้ขึ้น)")

    return {
        "old": {"brier": old_brier, "ece": old_ece, "bins": old_bins},
        "new": {"brier": new_brier, "ece": new_ece, "bins": new_bins},
        "ece_improvement_pct": improvement_pct,
    }


results = {}

# --- SENTIMENT ---
wongnai = pd.read_csv(OUT / "wongnai_cleaned.csv", encoding="utf-8-sig")
_parts = [grp.sample(min(len(grp), 2500), random_state=42) for _, grp in wongnai.groupby("sentiment")]
wongnai = pd.concat(_parts)
X = wongnai["clean_text"].astype(str)
y = wongnai["sentiment"].astype(str)
_, X_test_s, _, y_test_s = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

sent_old = joblib.load(MODELS_OLD / "sentiment_model.joblib")
sent_new = joblib.load(MODELS / "sentiment_model.joblib")
results["sentiment"] = run_task("SENTIMENT — ML ล้วนๆ (ไม่ผ่าน keyword override)", X_test_s.tolist(), y_test_s.tolist(), sent_old, sent_new)

# --- CATEGORY ---
crm = pd.read_csv(OUT / "crm_cleaned.csv", encoding="utf-8-sig")
X = crm["clean_text"].astype(str)
y = crm["Category"].astype(str)
stratify = y if y.value_counts().min() >= 2 else None
_, X_test_c, _, y_test_c = train_test_split(X, y, test_size=0.20, random_state=42, stratify=stratify)

cat_old = joblib.load(MODELS_OLD / "category_model.joblib")
cat_new = joblib.load(MODELS / "category_model.joblib")
results["category"] = run_task("CATEGORY — ML ล้วนๆ (ไม่ผ่าน keyword override)", X_test_c.tolist(), y_test_c.tolist(), cat_old, cat_new)

with open(OUT / "calibration_check.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\nบันทึกผลละเอียดไว้ที่:", OUT / "calibration_check.json")
