# -*- coding: utf-8 -*-
"""
10_build_sentiment_dataset.py
=============================
สร้าง data/sentiment_dataset_v2.csv สำหรับฝึก sentiment model จาก 2 แหล่ง:

1. Wisesight Sentiment Corpus (data/external/wisesight-sentiment, CC0)
   ข้อความโซเชียลมีเดียไทยจริง รวมคอมเมนต์ใต้โพสต์เพจ Facebook สาธารณะ
   **ติดป้ายโดยคน** 4 แบบ: pos / neg / neu / q
   - q (คำถาม) แปลงเป็น neutral ตามนิยามของระบบนี้ (neutral = คำถามล้วน หรือไม่แสดง
     ความรู้สึกชัดเจน) เก็บป้ายเดิมไว้ในคอลัมน์ orig_label
2. รีวิว Wongnai (data/wongnai.csv) แปลงคะแนนดาวเป็นป้าย (1-2 = negative,
   3 = neutral, 4-5 = positive) สุ่มไม่เกิน 2,500 รีวิวต่อป้ายเหมือนเดิม

เหตุผลที่เพิ่ม Wisesight: เดิมใช้แค่ป้ายจากดาวรีวิว Wongnai ซึ่งเป็นตัวแทนความรู้สึกแบบ
ประมาณ และรีวิวยาวกว่าคอมเมนต์ Facebook มาก ส่วน Wisesight เป็นข้อความสั้นจากโซเชียลที่
คนติดป้ายเอง ใกล้กับข้อมูลที่ระบบเจอจริงกว่า

คอลัมน์ split (train/test) กำหนดไว้ในไฟล์เลย เพื่อให้ทุกโมเดล (TF-IDF+SVM,
WangchanBERTa, LLM) วัดผลบนแถวทดสอบชุดเดียวกันเป๊ะ — แบ่ง 80/20 แบบ stratified ตาม
(source, sentiment) ด้วย random_state=42

ผลลัพธ์: data/sentiment_dataset_v2.csv
        คอลัมน์ clean_text, sentiment, source, orig_label, split, text (ข้อความดิบ)
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from app.preprocess import clean_text, star_to_sentiment

DATA = BASE / "data"
WISESIGHT = DATA / "external" / "wisesight-sentiment"
WISESIGHT_LABELS = {"pos": "positive", "neg": "negative", "neu": "neutral", "q": "neutral"}
WONGNAI_MAX_PER_LABEL = 2500


def load_wisesight() -> pd.DataFrame:
    rows = []
    for orig, label in WISESIGHT_LABELS.items():
        with open(WISESIGHT / f"{orig}.txt", encoding="utf-8") as f:
            for line in f:
                rows.append({"text": line.strip(), "sentiment": label,
                             "source": "wisesight", "orig_label": orig})
    return pd.DataFrame(rows)


def load_wongnai() -> pd.DataFrame:
    df = pd.read_csv(DATA / "wongnai.csv", encoding="cp874", on_bad_lines="skip")
    df["sentiment"] = df["stars"].apply(star_to_sentiment)
    df = df.dropna(subset=["sentiment"])
    df["clean"] = df["review_body"].apply(clean_text)
    df = df[df["clean"].str.len() > 0].sort_values("clean")
    parts = [g.sample(min(len(g), WONGNAI_MAX_PER_LABEL), random_state=42)
             for _, g in df.groupby("sentiment")]
    df = pd.concat(parts)
    return pd.DataFrame({"text": df["review_body"], "sentiment": df["sentiment"],
                         "source": "wongnai", "orig_label": df["stars"].astype(int).astype(str) + "_stars"})


def main():
    df = pd.concat([load_wisesight(), load_wongnai()], ignore_index=True)
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0]
    # ข้อความเดียวกันที่ถูกติดป้ายต่างกัน (มีใน Wisesight บ้าง) ตัดทิ้งทั้งหมด — กำกวม
    conflict = df.groupby("clean_text")["sentiment"].nunique()
    df = df[~df["clean_text"].isin(conflict[conflict > 1].index)]
    df = df.drop_duplicates(subset=["clean_text"]).sort_values(["source", "sentiment", "clean_text"])
    df = df.reset_index(drop=True)

    strata = df["source"] + "|" + df["sentiment"]
    train_idx, test_idx = train_test_split(df.index, test_size=0.20, random_state=42, stratify=strata)
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"

    # text = ข้อความดิบ (มีอีโมจิ/วรรคตอน) สำหรับ WangchanBERTa — TF-IDF ใช้ clean_text
    df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    out = df[["clean_text", "sentiment", "source", "orig_label", "split", "text"]]
    out.to_csv(DATA / "sentiment_dataset_v2.csv", index=False, encoding="utf-8-sig")
    print(pd.crosstab([out["source"], out["sentiment"]], out["split"], margins=True, margins_name="รวม").to_string())
    print(f"\nบันทึกไว้ที่: {DATA / 'sentiment_dataset_v2.csv'}  (รวม {len(out)} แถว)")


if __name__ == "__main__":
    main()
