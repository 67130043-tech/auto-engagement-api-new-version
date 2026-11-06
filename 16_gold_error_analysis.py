# -*- coding: utf-8 -*-
"""
16_gold_error_analysis.py
=========================
ทำนาย gold standard ด้วยทุกระบบ แล้วบันทึกผลรายข้อ + confusion matrix ของ category/sentiment

ผลลัพธ์:
  outputs/gold_predictions.csv          gold_id, source, ป้ายจริง, ผลทำนายของแต่ละระบบ (+ความมั่นใจ)
  outputs/gold_confusion_<task>_<system>.csv   แถว = ป้ายจริง, คอลัมน์ = ที่ทำนาย
  outputs/gold_per_class_<task>.csv     precision/recall/F1 ต่อหมวด ของแต่ละระบบ
--print-compact  พิมพ์ผลรายข้อแบบย่อ (ใช้คัดลอกออกจาก Colab)
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from app.preprocess import clean_text
from app.engine import keyword_sentiment_override, keyword_category_override, map_to_ml_category
from app.transformer_model import load_transformer_model

OUT = BASE / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--print-compact", action="store_true")
    args = ap.parse_args()
    gold = pd.read_csv(BASE / "data" / "gold_standard.csv", encoding="utf-8-sig").fillna("")
    msgs = gold["message"].astype(str).tolist()
    clean = [clean_text(m) for m in msgs]
    out = gold[["gold_id", "source", "sentiment", "category"]].copy()
    systems = []
    for task, gcol in (("sentiment", "sentiment"), ("category", "category")):
        tfidf = joblib.load(BASE / "models" / f"{task}_model.joblib")
        p = tfidf.predict_proba(clean); cls = np.asarray(tfidf.classes_)
        pred = cls[p.argmax(1)]
        out[f"{task}_tfidf"] = pred
        if task == "sentiment":
            out[f"{task}_tfidf_kw"] = [keyword_sentiment_override(t, x) for t, x in zip(clean, pred)]
        else:
            out[f"{task}_tfidf_kw"] = [map_to_ml_category(keyword_category_override(t, x), x) for t, x in zip(clean, pred)]
        w = load_transformer_model(task)
        if w is not None:
            pw = w.predict_proba(msgs); out[f"{task}_wangchan"] = np.asarray(w.classes_)[pw.argmax(1)]
            out[f"{task}_wangchan_conf"] = pw.max(1).round(3)
    OUT.mkdir(exist_ok=True)
    out.to_csv(OUT / "gold_predictions.csv", index=False, encoding="utf-8-sig")

    for task in ("sentiment", "category"):
        labels = sorted(set(out[task]))
        rows = []
        for sysname in ("tfidf", "tfidf_kw", "wangchan"):
            col = f"{task}_{sysname}"
            if col not in out:
                continue
            labs = sorted(set(labels) | set(out[col]))
            cm = pd.DataFrame(confusion_matrix(out[task], out[col], labels=labs), index=labs, columns=labs)
            cm.to_csv(OUT / f"gold_confusion_{task}_{sysname}.csv", encoding="utf-8-sig")
            rep = classification_report(out[task], out[col], labels=labels, output_dict=True, zero_division=0)
            for lab in labels:
                rows.append({"system": sysname, "label": lab, **{k: round(v, 4) for k, v in rep[lab].items()}})
        pd.DataFrame(rows).to_csv(OUT / f"gold_per_class_{task}.csv", index=False, encoding="utf-8-sig")
    print(f"saved {OUT / 'gold_predictions.csv'} ({len(out)} rows)")
    if args.print_compact:
        cols = [c for c in out.columns if c not in ("source", "sentiment", "category")]
        print("BEGIN_PRED"); print(out[cols].to_csv(index=False)); print("END_PRED")


if __name__ == "__main__":
    main()
