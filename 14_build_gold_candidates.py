# -*- coding: utf-8 -*-
"""
14_build_gold_candidates.py
===========================
สร้างชุดข้อความผู้สมัครสำหรับ gold standard (ให้คน 2 คนติดป้ายแยกกัน) -> data/gold_candidates.csv

แหล่งข้อความ (ข้อความจริงทั้งหมด ไม่สังเคราะห์):
1. wisesight  คอมเมนต์/โพสต์โซเชียลจริงจาก Wisesight Sentiment Corpus เฉพาะแถวในชุด test
              (split=="test" ใน sentiment_dataset_v2.csv จึงไม่เคยใช้ฝึก sentiment model และ
              category model ฝึกจาก Wongnai อย่างเดียว) คัดเฉพาะที่มีคำเกี่ยวกับร้านอาหาร
              แล้วสุ่มแบบแบ่งชั้นตามป้ายเดิมของ Wisesight (ใช้แค่ให้สัดส่วนอารมณ์ไม่เอียง —
              ผู้ติดป้ายไม่เห็นป้ายนี้)
2. own_page   คอมเมนต์บนเพจทดสอบของระบบ (ชีต AI_Log จาก Make.com) ตัดซ้ำ ปิดชื่อบุคคล
              หมายเหตุ: ส่วนใหญ่ผู้วิจัย/คนรู้จักพิมพ์ทดสอบ ควรรายงานผลแยกแหล่งในบทที่ 4

ข้อความที่ไม่เกี่ยวกับร้านอาหาร ผู้ติดป้ายเลือก "ไม่เกี่ยวข้อง" แล้วจะถูกตัดออกจากการวัดผล
"""
import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
SEED = 42
WISESIGHT_PER_LABEL = {"negative": 110, "neutral": 110, "positive": 80}

RESTAURANT_RE = re.compile(
    r"ร้าน|อร่อย|เมนู|อาหาร|รสชาติ|สั่ง|เดลิเวอรี่|delivery|ไลน์แมน|grab|พนักงาน|บริการ|โปร|ส่วนลด|"
    r"จองโต๊ะ|สาขา|เปิดกี่โมง|ก๋วยเตี๋ยว|ชาบู|หมูกระทะ|บุฟเฟ่ต์|ส้มตำ|กาแฟ|ชานม|ขนม|เค้ก|พิซซ่า|"
    r"ไก่ทอด|ข้าว|ปิ้งย่าง|ซูชิ|ราเมง", re.IGNORECASE)


def main():
    s = pd.read_csv(DATA / "sentiment_dataset_v2.csv", encoding="utf-8-sig")
    ws = s[(s["source"] == "wisesight") & (s["split"] == "test")].copy()
    ws["message"] = ws["text"].astype(str).str.strip()
    ws = ws[ws["message"].str.contains(RESTAURANT_RE) & ws["message"].str.len().between(8, 300)]
    ws = pd.concat([g.sample(min(len(g), WISESIGHT_PER_LABEL[lab]), random_state=SEED)
                    for lab, g in ws.groupby("sentiment")])
    ws = ws.assign(source="wisesight", ref_label=ws["orig_label"])[["message", "source", "ref_label"]]

    own = pd.read_csv(DATA / "external" / "own_page_comments.csv", encoding="utf-8-sig")
    own = own.assign(source="own_page", ref_label="")[["message", "source", "ref_label"]]

    df = pd.concat([ws, own]).drop_duplicates(subset="message")
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.insert(0, "gold_id", [f"G{i + 1:03d}" for i in range(len(df))])
    df.to_csv(DATA / "gold_candidates.csv", index=False, encoding="utf-8-sig")
    print(df["source"].value_counts().to_string())
    print(f"บันทึกไว้ที่: {DATA / 'gold_candidates.csv'}  (รวม {len(df)} ข้อความ)")


if __name__ == "__main__":
    main()
