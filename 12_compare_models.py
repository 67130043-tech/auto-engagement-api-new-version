# -*- coding: utf-8 -*-
"""
12_compare_models.py
====================
เปรียบเทียบตัวจำแนก sentiment และ category บนชุดทดสอบเดียวกัน (ใช้ในบทที่ 4)

ระบบที่เทียบ:
  tfidf_svm          TF-IDF + LinearSVC (models/*.joblib จาก 02_train_models.py)
  tfidf_svm+keyword  แบบเดิมของระบบ: keyword override ทับผล TF-IDF (โหมด hybrid)
  wangchanberta      WangchanBERTa fine-tune (models/wangchanberta_*/ จาก 11_finetune_wangchanberta.py)
  llm_zero_shot      OpenAI zero-shot (เฉพาะเมื่อใส่ --llm และตั้ง OPENAI_API_KEY; สุ่ม --llm-n แถว)

ชุดทดสอบ:
  test   แถว split=="test" ใน data/sentiment_dataset_v2.csv / data/category_dataset_v2.csv
         หมายเหตุ: ป้าย category ชุดนี้สร้างจากกฎ keyword (weak supervision) ระบบ
         tfidf_svm+keyword จึงได้เปรียบโดยโครงสร้าง ไม่ควรใช้ตัดสินว่าระบบไหนดีกว่า
  gold   data/gold_standard.csv (คนติดป้าย — ดู data/gold_standard_template.csv)
         **ตัวเลขหลักของงานวิจัยควรมาจากชุดนี้**

ตัวชี้วัด: accuracy, macro-F1 และ ECE (ความสอดคล้องของค่าความมั่นใจ) เมื่อโมเดลให้ความน่าจะเป็น

ผลลัพธ์: outputs/model_comparison.csv และ outputs/model_comparison.json
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from app.preprocess import clean_text
from app.engine import keyword_sentiment_override, keyword_category_override, map_to_ml_category
from app.transformer_model import load_transformer_model

DATA, OUT, MODELS = BASE / "data", BASE / "outputs", BASE / "models"
LABEL_COL = {"sentiment": "sentiment", "category": "Category"}


def ece(y_true, pred, conf, n_bins=10):
    correct = (np.asarray(pred) == np.asarray(y_true)).astype(float)
    conf = np.asarray(conf)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & ((conf <= bins[i + 1]) if i == n_bins - 1 else (conf < bins[i + 1]))
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def score(system, task, eval_set, y_true, pred, conf=None):
    row = {"task": task, "system": system, "eval_set": eval_set, "n": len(y_true),
           "accuracy": round(float(accuracy_score(y_true, pred)), 4),
           "macro_f1": round(float(f1_score(y_true, pred, average="macro", zero_division=0)), 4),
           "ece": round(ece(y_true, pred, conf), 4) if conf is not None else None}
    print(f"  {task:<9} {eval_set:<5} {system:<18} n={row['n']:5d}  acc={row['accuracy']:.4f}  "
          f"macro-F1={row['macro_f1']:.4f}" + (f"  ECE={row['ece']:.4f}" if row["ece"] is not None else ""))
    return row


def proba_pred(model, texts):
    probs = model.predict_proba(texts)
    classes = np.asarray(model.classes_)
    return classes[probs.argmax(axis=1)], probs.max(axis=1)


def keyword_pipeline(task, texts, ml_pred):
    if task == "sentiment":
        return [keyword_sentiment_override(t, p) for t, p in zip(texts, ml_pred)]
    return [map_to_ml_category(keyword_category_override(t, p), p) for t, p in zip(texts, ml_pred)]


def llm_predict(task, texts):
    from app.openai_verifier import verify_category, verify_sentiment
    out = []
    for t in texts:
        if task == "sentiment":
            label, _ = verify_sentiment(t)
        else:
            detail, _ = verify_category(t)
            label = map_to_ml_category(detail, None) if detail else None
        out.append(label or "__failed__")
    return out


def load_eval_sets(task):
    sets = {}
    csv = DATA / f"{task}_dataset_v2.csv"
    df = pd.read_csv(csv, encoding="utf-8-sig")
    test = df[df["split"] == "test"]
    sets["test"] = (test["clean_text"].astype(str).tolist(), test[LABEL_COL[task]].astype(str).tolist())
    gold_path = DATA / "gold_standard.csv"
    if gold_path.exists():
        gold = pd.read_csv(gold_path, encoding="utf-8-sig").dropna(subset=["message", task])
        gold = gold[gold[task].astype(str).str.strip() != ""]
        if len(gold):
            sets["gold"] = ([clean_text(m) for m in gold["message"]], gold[task].astype(str).str.strip().tolist())
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="รวม OpenAI zero-shot (เสียค่า API)")
    ap.add_argument("--llm-n", type=int, default=300, help="จำนวนแถวที่สุ่มให้ LLM ต่อชุดทดสอบ")
    args = ap.parse_args()

    rows = []
    for task in ("sentiment", "category"):
        print(f"\n=== {task} ===")
        tfidf = joblib.load(MODELS / f"{task}_model.joblib")
        wangchan = load_transformer_model(task)
        if wangchan is None:
            print("  (ยังไม่มี WangchanBERTa สำหรับ task นี้ — ข้าม; รัน 11_finetune_wangchanberta.ipynb ก่อน)")
        for eval_set, (texts, y_true) in load_eval_sets(task).items():
            pred, conf = proba_pred(tfidf, texts)
            rows.append(score("tfidf_svm", task, eval_set, y_true, pred, conf))
            rows.append(score("tfidf_svm+keyword", task, eval_set, y_true, keyword_pipeline(task, texts, pred)))
            if wangchan is not None:
                wpred, wconf = proba_pred(wangchan, texts)
                rows.append(score("wangchanberta", task, eval_set, y_true, wpred, wconf))
            if args.llm:
                idx = np.random.RandomState(42).permutation(len(texts))[:args.llm_n]
                sub_t, sub_y = [texts[i] for i in idx], [y_true[i] for i in idx]
                rows.append(score("llm_zero_shot", task, eval_set, sub_y, llm_predict(task, sub_t)))

    OUT.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "model_comparison.csv", index=False, encoding="utf-8-sig")
    (OUT / "model_comparison.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nบันทึกไว้ที่: {OUT / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
