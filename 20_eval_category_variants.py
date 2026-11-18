"""เทียบโมเดลหมวดหมู่ WangchanBERTa หลายรุ่นบนชุดทดสอบมาตรฐาน (gold standard 209 ข้อ)

    python 20_eval_category_variants.py models/wangchanberta_category models/wc_cat_mix3 models/wc_cat_human

รายงาน accuracy / macro-F1 พร้อม 95% bootstrap CI ทั้งโมเดลเดี่ยวและเมื่อรวมกฎคำสำคัญ
ผลแยกตามแหล่งที่มา และจำนวนครั้งที่ทำนาย "ปัญหาออเดอร์/จัดส่ง" -> outputs/category_variants_gold.json
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from app.engine import V2_TO_V3_CATEGORY  # noqa: E402
from app.preprocess import clean_text  # noqa: E402
from app.transformer_model import OnnxTextClassifier, prepare_text  # noqa: E402

spec = importlib.util.spec_from_file_location("c12", BASE / "12_compare_models.py")
c12 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c12)


def ci(y, p, rng, B=2000):
    y, p = np.asarray(y), np.asarray(p)
    a, f = [], []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        a.append((y[i] == p[i]).mean())
        f.append(f1_score(y[i], p[i], average="macro"))
    return [round(float(x), 3) for x in np.percentile(a, [2.5, 97.5])], [round(float(x), 3) for x in np.percentile(f, [2.5, 97.5])]


def main():
    dirs = sys.argv[1:]
    g = pd.read_csv(BASE / "data/gold_standard.csv", encoding="utf-8-sig")
    y = g["category"].tolist()
    raw = [prepare_text(m) for m in g["message"].astype(str)]
    clean = [clean_text(m) for m in g["message"].astype(str)]
    out = {}
    for d in dirs:
        m = OnnxTextClassifier(d)
        pred = [V2_TO_V3_CATEGORY.get(x, x) for x in m.predict(raw)]
        hyb = c12.keyword_pipeline("category", clean, pred)
        for name, p in [(Path(d).name, pred), (Path(d).name + "+keyword", hyb)]:
            rng = np.random.default_rng(42)
            acc, mf1 = accuracy_score(y, p), f1_score(y, p, average="macro")
            ca, cf = ci(y, p, rng)
            by = {s: round(accuracy_score(g.category[g.source == s], np.asarray(p)[(g.source == s).values]), 3)
                  for s in ["wisesight", "own_page"]}
            out[name] = {"accuracy": round(acc, 4), "macro_f1": round(mf1, 4), "acc_ci": ca, "f1_ci": cf,
                         "by_source": by, "pred_order_problem": int(sum(x == "ปัญหาออเดอร์/จัดส่ง" for x in p))}
            print(f"{name:40s} acc={acc:.3f} {ca} mF1={mf1:.3f} {cf} {by} order={out[name]['pred_order_problem']}")
    (BASE / "outputs").mkdir(exist_ok=True)
    json.dump(out, open(BASE / "outputs/category_variants_gold.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
