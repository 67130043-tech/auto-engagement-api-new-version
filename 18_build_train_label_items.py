"""เลือกคอมเมนต์จริงจาก Wisesight ส่วน train (ไม่ทับกับ gold standard ซึ่งมาจากส่วน test)
เพื่อให้คนติดป้ายหมวดหมู่ 8 คลาส แล้วใช้เป็นข้อมูลฝึกที่มนุษย์ติดป้าย

    python 18_build_train_label_items.py  -> data/train_label_items.csv
"""
import importlib.util
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("g14", BASE / "14_build_gold_candidates.py")
g14 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g14)

PER_LABEL = {"neg": 300, "neu": 300, "pos": 200}


def main():
    s = pd.read_csv(BASE / "data/sentiment_dataset_v2.csv", encoding="utf-8-sig")
    w = s[(s.source == "wisesight") & (s.split == "train")].copy()
    w["message"] = w["text"].astype(str).str.strip()
    w = w[w.message.str.len().between(8, 300)].drop_duplicates("clean_text")
    w = w[g14.strict_ok(w.message)]
    gold = pd.read_csv(BASE / "data/gold_candidates.csv", encoding="utf-8-sig")
    w = w[~w.message.isin(set(gold.message.astype(str).str.strip()))]
    parts = [w[w.orig_label == k].sample(min(n, (w.orig_label == k).sum()), random_state=42) for k, n in PER_LABEL.items()]
    out = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
    out.insert(0, "item_id", ["T%04d" % (i + 1) for i in range(len(out))])
    out = out[["item_id", "message", "orig_label", "sentiment"]]
    out.to_csv(BASE / "data/train_label_items.csv", index=False, encoding="utf-8-sig")
    print(len(out), out.orig_label.value_counts().to_dict())


if __name__ == "__main__":
    main()
