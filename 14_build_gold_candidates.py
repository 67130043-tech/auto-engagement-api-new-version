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

--refine (ใช้หลังเริ่มติดป้ายแล้ว): ตัวกรองคำรอบแรกกว้างเกินไป ("โปร" ไปจับ "สิงคโปร์/โปรดักส์",
"สั่ง/บริการ/ข้าว" ไปจับเรื่องรถ เครื่องสำอาง ฯลฯ) ผู้ติดป้ายเจอข้อความไม่เกี่ยวข้อง ~1 ใน 3
โหมดนี้ใช้ตัวกรองเข้มขึ้น (คำเฉพาะอาหาร/ชื่อร้าน + ตัดคำนอกเรื่อง) แล้ว
  - ข้อ Wisesight ที่ยังไม่มีใครติดป้าย (gold_id > --labeled-upto) และไม่ผ่านตัวกรองใหม่ -> status=dropped
  - เติมข้อใหม่ที่ผ่านตัวกรองจาก Wisesight ชุด test จำนวนเท่ากัน ใช้ gold_id ต่อท้าย (ไม่ใช้ id ซ้ำ
    เพื่อไม่ให้ป้ายที่ติดไว้แล้วชี้ผิดข้อความ)
ข้อที่ติดป้ายไปแล้วไม่ถูกแตะต้อง
"""
import argparse
import re
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
SEED = 42
WISESIGHT_PER_LABEL = {"negative": 110, "neutral": 110, "positive": 80}

STRICT_RE = re.compile(
    r"อร่อย|รสชาติ|เมนู|ร้านอาหาร|ชาบู|หมูกระทะ|บุฟเฟ่ต์|บุฟเฟต์|ส้มตำ|ก๋วยเตี๋ยว|พิซซ่า|pizza|ไก่ทอด|ปิ้งย่าง|ซูชิ|"
    r"ราเมน|ราเมง|สุกี้|กาแฟ|ชานม|ชาไข่มุก|เค้ก|ไอศครีม|ไอติม|เดลิเวอรี่|delivery|ไลน์แมน|lineman|grab ?food|"
    r"foodpanda|น้ำจิ้ม|เสิร์ฟ|ไปกิน|กินข้าว|มื้อ|เอ็มเค|\bmk\b|บาร์บีคิว|bar ?b ?q|\bkfc\b|เคเอฟซี|แมคโด|"
    r"mcdonald|starbucks|สตาร์บัค|เชสเตอร์|ฟูจิ|โอบองแปง|s&p|after you|แดรี่ควีน|สเวนเซ่น|swensen|ยาโยอิ|"
    r"โออิชิ|ชาบูชิ|ซิซซ์เล่อร์|sizzler|อาหาร|ร้าน.{0,15}(?:กิน|อร่อย)", re.IGNORECASE)
OFFTOPIC_RE = re.compile(
    r"รถ|ครีม|เครื่องสำอาง|หุ้น|ประกัน|มือถือ|โทรศัพท์|ไอโฟน|iphone|ซัมซุง|\bais\b|dtac|ทรูมูฟ|ธนาคาร|"
    r"บัตรเครดิต|คอนโด|สิว|ผิว|ลิป|เซรั่ม|แชมพู|ผงซักฟอก|อาหารเสริม|อาหารแมว|อาหารหมา|อาหารสุนัข|นม ?ผง",
    re.IGNORECASE)


def strict_ok(text: pd.Series) -> pd.Series:
    return text.str.contains(STRICT_RE) & ~text.str.contains(OFFTOPIC_RE)


RESTAURANT_RE = re.compile(
    r"ร้าน|อร่อย|เมนู|อาหาร|รสชาติ|สั่ง|เดลิเวอรี่|delivery|ไลน์แมน|grab|พนักงาน|บริการ|โปร|ส่วนลด|"
    r"จองโต๊ะ|สาขา|เปิดกี่โมง|ก๋วยเตี๋ยว|ชาบู|หมูกระทะ|บุฟเฟ่ต์|ส้มตำ|กาแฟ|ชานม|ขนม|เค้ก|พิซซ่า|"
    r"ไก่ทอด|ข้าว|ปิ้งย่าง|ซูชิ|ราเมง", re.IGNORECASE)


def wisesight_test() -> pd.DataFrame:
    s = pd.read_csv(DATA / "sentiment_dataset_v2.csv", encoding="utf-8-sig")
    ws = s[(s["source"] == "wisesight") & (s["split"] == "test")].copy()
    ws["message"] = ws["text"].astype(str).str.strip()
    return ws[ws["message"].str.len().between(8, 300)]


def refine(labeled_upto: int):
    cand = pd.read_csv(DATA / "gold_candidates.csv", encoding="utf-8-sig")
    if "status" not in cand.columns:
        cand["status"] = "active"
    num = cand["gold_id"].str[1:].astype(int)
    drop = (cand["source"] == "wisesight") & (num > labeled_upto) & (cand["status"] == "active") & ~strict_ok(cand["message"])
    cand.loc[drop, "status"] = "dropped"
    n_new = int(drop.sum())

    ws = wisesight_test()
    ws = ws[strict_ok(ws["message"]) & ~ws["message"].isin(set(cand["message"]))]
    share = cand[(cand["source"] == "wisesight") & drop]["ref_label"].value_counts(normalize=True)
    per = {lab: round(n_new * share.get(orig, 0)) for lab, orig in [("negative", "neg"), ("neutral", "neu"), ("positive", "pos")]}
    per["neutral"] += n_new - sum(per.values())
    picks = []
    for lab, g in ws.groupby("sentiment"):
        picks.append(g.sample(min(len(g), per.get(lab, 0)), random_state=SEED))
    new = pd.concat(picks).sample(frac=1, random_state=SEED)
    start = int(num.max()) + 1
    new = pd.DataFrame({"gold_id": [f"G{start + i:03d}" for i in range(len(new))], "message": new["message"].values,
                        "source": "wisesight", "ref_label": new["orig_label"].values, "status": "active"})
    out = pd.concat([cand, new], ignore_index=True)
    out.to_csv(DATA / "gold_candidates.csv", index=False, encoding="utf-8-sig")
    print(f"dropped {n_new}, added {len(new)}; active = {(out.status == 'active').sum()}")
    print(out[out.status == "active"]["source"].value_counts().to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--labeled-upto", type=int, default=28, help="เลขข้อสุดท้ายที่เริ่มติดป้ายไปแล้ว (ไม่แตะต้อง)")
    args = ap.parse_args()
    if args.refine:
        return refine(args.labeled_upto)
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
    df["status"] = "active"
    df.to_csv(DATA / "gold_candidates.csv", index=False, encoding="utf-8-sig")
    print(df["source"].value_counts().to_string())
    print(f"บันทึกไว้ที่: {DATA / 'gold_candidates.csv'}  (รวม {len(df)} ข้อความ)")


if __name__ == "__main__":
    main()
