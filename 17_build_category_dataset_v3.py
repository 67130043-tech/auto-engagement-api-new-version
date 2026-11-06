# -*- coding: utf-8 -*-
"""
17_build_category_dataset_v3.py
===============================
แปลง data/category_dataset_v2.csv (10 หมวด) เป็น taxonomy v3 (8 หมวดที่ไม่ทับกัน) ->
data/category_dataset_v3.csv

เหตุผล: วิเคราะห์ gold standard (16_gold_error_analysis.py) พบว่า "ติชมอาหารและบริการ"
เป็นหมวดรวมที่ทับกับ "บริการไม่ดี" และ "ชมรสชาติอาหาร" จนผู้ติดป้ายเองยังเห็นไม่ตรงกัน
และ "ชมสถานที่/บรรยากาศ" มีตัวอย่างจริงน้อยจนเรียนไม่ได้ (F1 0.15)

หมวด v3 (ดู app/engine.py ML_CATEGORY_CLASSES):
  ชมอาหาร | ชมบริการ/บรรยากาศ | ติอาหาร | ติบริการ/ร้าน | ปัญหาออเดอร์/จัดส่ง |
  สอบถามข้อมูลร้าน | สอบถามเมนูอาหาร | สอบถามโปรโมชั่น

วิธีแปลง (ไม่มีการสังเคราะห์ข้อความ — ใช้แถวเดิมทั้งหมด แค่ติดป้ายใหม่):
- แถวจาก keyword (wongnai_real): หาหมวดละเอียดด้วย keyword ชุดเดิมแล้ว map ด้วย
  KEYWORD_CATEGORY_TO_ML_CATEGORY (v3) เช่น ร้องเรียนคุณภาพอาหาร/ราคา -> ติอาหาร,
  ร้องเรียนแอร์/ห้องน้ำ/ความสะอาด -> ติบริการ/ร้าน, สอบถามราคา -> สอบถามเมนูอาหาร
  หมวดละเอียด "รีวิว/ให้คะแนน", "ข้อเสนอแนะทั่วไป" (ชมหรือติก็ได้) -> ตัดทิ้ง
- แถวจาก pattern "อาหารได้ไม่ตรง/ช้า": ได้ผิด/ไม่ครบ -> ปัญหาออเดอร์/จัดส่ง,
  ช้าอย่างเดียว (รออาหารนานในร้าน) -> ติบริการ/ร้าน
- แถว "การจัดส่ง (Delivery)": คำชมการจัดส่ง (ส่งไว แพ็คดี) -> ชมบริการ/บรรยากาศ
  นอกนั้น -> ปัญหาออเดอร์/จัดส่ง (หัวข้อการสั่ง/ส่ง)
- จำกัดไม่เกิน 1,200 แถว/หมวด แล้วแบ่ง train/test ใหม่ 80/20 stratified (random_state=42)
"""
import importlib.util
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from app.engine import KEYWORD_CATEGORY_TO_ML_CATEGORY, ML_CATEGORY_CLASSES
from app.preprocess import clean_text, tokenize_boundary

_spec = importlib.util.spec_from_file_location("b09", BASE / "09_build_category_dataset_from_wongnai.py")
b09 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b09)

DATA = BASE / "data"
MAX_PER_CATEGORY = 1200
SLOW_RE = re.compile(b09._ORDER_SLOW)
WRONG_RE = re.compile(b09._ORDER_WRONG)
DELIVERY_PRAISE_RE = re.compile(r"ส่งไว|ส่งเร็ว|ไวมาก|เร็วมาก|แพ็ค(มา)?(อย่าง)?ดี|แพค(มา)?(อย่าง)?ดี|ยังร้อน|ร้อนๆ|เรียบร้อยดี")
DELIVERY_COMPLAINT_RE = re.compile(r"ช้า|นาน|หก|เลอะ|ชืด|ไม่|ผิด|ขาด|แย่|หาย")


def detail_of(text: str):
    tb = tokenize_boundary(clean_text(text))
    return b09.keyword_category_and_match(tb) or b09.keyword_category_best_match(tb)


def to_v3(row) -> str | None:
    cat, src, text = row["Category"], row["source"], str(row["text"])
    if cat == "อาหารได้ไม่ตรง/ช้า":
        if WRONG_RE.search(text) or src == "wongnai_real":   # wongnai_real ในหมวดนี้ = ยกเลิก/คืนเงิน
            return "ปัญหาออเดอร์/จัดส่ง"
        return "ติบริการ/ร้าน" if SLOW_RE.search(text) else None
    if cat == "การจัดส่ง (Delivery)":
        if DELIVERY_PRAISE_RE.search(text) and not DELIVERY_COMPLAINT_RE.search(text):
            return "ชมบริการ/บรรยากาศ"
        return "ปัญหาออเดอร์/จัดส่ง"
    if src != "wongnai_real":  # pattern ชมการบริการ / สอบถามโปรโมชั่น
        return {"ชมการบริการ": "ชมบริการ/บรรยากาศ", "สอบถามโปรโมชั่น": "สอบถามโปรโมชั่น"}.get(cat)
    d = detail_of(text)
    if d in ML_CATEGORY_CLASSES:
        return d
    return KEYWORD_CATEGORY_TO_ML_CATEGORY.get(d)


def main():
    v2 = pd.read_csv(DATA / "category_dataset_v2.csv", encoding="utf-8-sig")
    v2["Category_v3"] = v2.apply(to_v3, axis=1)
    print("v2 -> v3:\n" + pd.crosstab(v2["Category"], v2["Category_v3"].fillna("(ตัดทิ้ง)")).to_string())
    df = v2.dropna(subset=["Category_v3"]).copy()
    df["Category_v2"] = df["Category"]
    df["Category"] = df["Category_v3"]
    parts = [g if len(g) <= MAX_PER_CATEGORY else g.sample(MAX_PER_CATEGORY, random_state=42)
             for _, g in df.groupby("Category")]
    df = pd.concat(parts).sort_values(["Category", "source", "clean_text"]).reset_index(drop=True)
    _, test_idx = train_test_split(df.index, test_size=0.20, random_state=42, stratify=df["Category"])
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"
    out = df[["clean_text", "Category", "source", "text", "split", "Category_v2"]]
    out.to_csv(DATA / "category_dataset_v3.csv", index=False, encoding="utf-8-sig")
    counts = out["Category"].value_counts()
    print("\n" + pd.crosstab(out["Category"], out["split"], margins=True).to_string())
    print(f"\nอัตราส่วนใหญ่สุด:เล็กสุด = {counts.max() / counts.min():.2f}:1")
    print(f"บันทึกไว้ที่: {DATA / 'category_dataset_v3.csv'} ({len(out)} แถว)")


if __name__ == "__main__":
    main()
