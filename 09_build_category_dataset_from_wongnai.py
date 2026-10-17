# -*- coding: utf-8 -*-
"""
09_build_category_dataset_from_wongnai.py
===========================================
สร้าง training dataset ใหม่สำหรับ category_model ทดแทน
Dataset_Restaurant_CRM_4000_Rows.xlsx เดิม

ปัญหาของ dataset เดิม (พบจากการตรวจสอบละเอียด 11 ก.ย.):
  4,000 แถว แต่มีข้อความไม่ซ้ำกันจริงแค่ 85 ประโยค (ซ้ำเฉลี่ย ~47 ครั้ง/ประโยค)
  ทำให้ train_test_split แบ่งประโยคเดียวกันเข้าทั้ง train และ test ได้ (แค่คนละ "ก็อปปี้")
  โมเดล "จำ" ข้อความที่เคยเห็นตอนเทรนได้ตรงๆ ไม่ใช่การ generalize จริง (accuracy 100%
  ที่วัดได้ก่อนหน้านี้จึงเป็นตัวเลขหลอก ไม่ใช่ความสามารถจริงของโมเดล)

วิธีแก้ในสคริปต์นี้ (weak supervision / distant supervision — แนวทางมาตรฐานเมื่อไม่มี
labeled dataset ที่ตรงโดเมนพอ ใช้กันทั่วไปในงานวิจัย NLP เมื่อไม่มีข้อมูลติดป้ายพร้อมใช้):
  1. ใช้รีวิวร้านอาหารไทยจริงจาก Wongnai corpus (data/wongnai.csv, ~19,458 รีวิว จริง
     ไม่ซ้ำกันเลย) แทนข้อความสังเคราะห์/ก็อปปี้เดิม
  2. ตัดรีวิวยาวๆ ออกเป็นประโยคย่อย (รีวิวจริงมักพูดถึงหลายแง่มุมปนกันในย่อหน้าเดียว
     เช่น ชมรสชาติ แล้วบ่นเรื่องที่จอดรถ) ให้ใกล้เคียงคอมเมนต์ Facebook สั้นๆ มากขึ้น
  3. ใช้ระบบ keyword_category_and_match() / keyword_category_best_match() ที่ทีมวิจัย
     ปรับแต่ง/ทดสอบมาแล้วตลอดทั้งโปรเจกต์ เป็นตัว "ติดป้าย" (label) ประโยคจริงเหล่านี้
     แทนคน -> เก็บเฉพาะประโยคที่ keyword จับ category ได้จริง (ทิ้งประโยคที่ไม่มี
     keyword ไหน match เลย ไม่บังคับติดป้ายมั่ว)
  4. หมวดที่เป็น "คำถาม" ล้วนๆ (สอบถามข้อมูลร้าน/สอบถามโปรโมชั่น/สอบถามเมนูอาหาร) แทบ
     ไม่มีทางเจอในรีวิวจริงเลย (รีวิวคือการเล่าประสบการณ์ ไม่ใช่การถามคำถามร้าน) จึง
     เสริมด้วยตัวอย่างคำถามที่เขียนขึ้นเองอย่างหลากหลาย (paraphrase) แทนการปล่อยให้
     หมวดเหล่านี้มีข้อมูลน้อยเกินไปจนโมเดลเรียนรู้ไม่ได้เลย — ระบุ source="synthetic_faq"
     ไว้ชัดเจน แยกจาก source="wongnai_real" เพื่อความโปร่งใสในรายงานวิทยานิพนธ์

  5. หมวด "อาหารได้ไม่ตรง/ช้า" ขาดแคลนหนัก (31 แถว) จึงขุดประโยคจริงจาก Wongnai ด้วย
     pattern เฉพาะเจาะจง (source="wongnai_pattern") ดู is_order_issue() ด้านล่าง
  6. หมวดอื่นที่ยังไม่ถึง 1,200 (Delivery, ชมการบริการ, สอบถามโปรโมชั่น) เติมด้วยประโยค
     จริงที่ pattern เฉพาะหมวดจับได้ ดู TOPUP_PATTERNS ด้านล่าง

ผลลัพธ์: outputs/category_dataset_v2.csv (คอลัมน์ clean_text, Category, source)
        และคัดลอกไปที่ data/category_dataset_v2.csv (สำเนาที่ commit ไว้ใน repo)
"""
import re
from pathlib import Path
import pandas as pd

import sys
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from app.preprocess import clean_text, tokenize_boundary
from app.keywords_data import (
    keyword_category_and_match,
    keyword_category_best_match,
)
from app.engine import map_to_ml_category, ML_CATEGORY_CLASSES

OUT = BASE / "outputs"

# ---------------------------------------------------------------------------
# 1) โหลดรีวิวจริงจาก Wongnai แล้วตัดเป็นประโยคย่อย
# ---------------------------------------------------------------------------
def split_into_chunks(text: str):
    """ตัดรีวิวยาวๆ เป็นประโยค/ท่อนสั้นๆ โดยใช้เครื่องหมายวรรคตอน/ตัวแบ่งที่มีจริงใน
    ข้อความ (ขึ้นบรรทัดใหม่, ฯลฯ, จุด, !, ?) ภาษาไทยไม่มีเครื่องหมายจบประโยคมาตรฐาน
    จึงใช้เท่าที่มีเป็นตัวช่วยแบ่งคร่าวๆ พอ ไม่ต้องสมบูรณ์แบบ เพราะ keyword matching
    ที่ใช้ label ต่อไปทำงานได้แม้ท่อนจะยาว/สั้นไม่เท่ากันบ้าง"""
    text = str(text)
    parts = re.split(r"[\n\r]+|(?<=[.!?])\s+|ๆ\s+|\s{2,}", text)
    chunks = [p.strip() for p in parts if p and len(p.strip()) >= 6]
    return chunks


def load_wongnai_chunks():
    df = pd.read_csv(BASE / "data" / "wongnai.csv", encoding="cp874", on_bad_lines="skip")
    all_chunks = set()
    for review in df["review_body"].astype(str):
        for chunk in split_into_chunks(review):
            all_chunks.add(chunk)
    # FIX (class imbalance): เดิมคืน list(all_chunks) ตรงๆ — ลำดับของ set เปลี่ยนทุกครั้ง
    # ที่รัน (Python สุ่ม hash ของ string ใหม่ทุก process) ทำให้ .sample(random_state=42)
    # ตอน cap หมวดได้แถวคนละชุดทุกรอบ dataset จึงทำซ้ำไม่ได้ เรียงก่อนคืนค่าให้ผลคงที่
    return sorted(all_chunks)


# ---------------------------------------------------------------------------
# 2) ติด label ด้วยระบบ keyword ที่มีอยู่แล้ว (weak supervision)
# ---------------------------------------------------------------------------
# FIX (class imbalance ของ category_dataset_v2.csv): หมวด "อาหารได้ไม่ตรง/ช้า" มีแค่
# 31 แถว (26 แถวเป็นประโยคที่เขียนเอง มีจาก Wongnai จริงแค่ 5 แถว) เทียบกับหมวดใหญ่
# 1,200 แถว (~39:1) เพราะ keyword table แม็พมาหมวดนี้แค่ "ยกเลิก/คืนเงิน" ซึ่งแคบมาก
# — ทั้งที่รีวิว Wongnai จริงมีคนบ่น "รออาหารนาน / เสิร์ฟผิด / ออเดอร์ตกหล่น" อยู่ไม่น้อย
#
# แก้ที่ต้นเหตุ (เพิ่มข้อมูลจริง) แทนการ copy แถวซ้ำ: ใช้ pattern เฉพาะเจาะจงขุดประโยค
# จริงที่พูดถึง "อาหารช้าหลังสั่ง" หรือ "ได้ไม่ตรง/ไม่ครบตามที่สั่ง" — ตั้งใจให้แคบ
# (precision มาก่อน recall) เพราะผลลัพธ์กลายเป็น label:
#   - SLOW  : ช้าหลังสั่งเท่านั้น ไม่นับการรอคิวหน้าร้าน (นั่นเป็นเรื่องร้าน/ความนิยม)
#   - WRONG : เสิร์ฟผิด/ไม่ครบ/ตกหล่น/ไม่ตรงที่สั่ง/ต้องทวง
#   - EXCLUDE: ตัดรูปปฏิเสธและความหมายอื่นที่ pattern ข้างบนจับผิด (เช่น "ไม่ผิดหวัง",
#     "ออเดอร์ไม่ตกหล่น", "ทำผิดกฎจราจร", "ถ่ายรูปมาไม่ครบ", "รอคิว", "ถือว่าไวมาก",
#     "สั่งผิดเมนู" ที่ลูกค้าสั่งผิดเอง)
# ประโยคที่ match ได้ source="wongnai_pattern" (ยังเป็นรีวิวจริง แยกไว้ให้ตรวจ/รายงานได้)
# และมีลำดับความสำคัญเหนือ keyword label เพราะ keyword ไม่มีทางติดหมวดนี้ให้อยู่แล้ว
# ---------------------------------------------------------------------------
_ORDER_SLOW = (
    r"(อาหาร|ของ|จาน|เมนู)(ที่สั่ง)?(มา|ออก|ได้|เสิร์ฟ|เสิฟ)(ค่อนข้าง)?(ช้า|นาน)"
    r"|(เสิร์ฟ|เสิฟ)(ช้า|นาน)"
    r"|อาหาร(ค่อนข้าง)?ช้า|ทำ(อาหาร)?(ค่อนข้าง)?ช้า"
    r"|รอ(อาหาร|ออเดอร์|ของ)(นาน|ประมาณ|เกือบ|ตั้ง|ร่วม|อีก|กว่า)"
    r"|สั่ง(ไป|แล้ว|อาหาร).{0,25}รอ.{0,15}(นาน|ชั่วโมง|ชม|นาที)"
    r"|กว่า(อาหาร|จะได้อาหาร|ของ).{0,10}(จะ)?(มา|ได้|ออก)"
)
_ORDER_WRONG = (
    r"(เสิร์ฟ|เสิฟ|เสริฟ|เสริพ|ทำ|ได้|จด)(อาหาร|เมนู|ของ)?ผิด(?!หวัง|กฎ)"
    r"|ผิดออเดอร์|ออเดอร์ผิด|ผิดเมนู|ไม่ตรง(กับ)?(ที่|ตาม)สั่ง|ไม่ได้ตามที่สั่ง|ที่ไม่ได้สั่ง"
    r"|(มา|ได้|เสิร์ฟ|เสิฟ)(อาหาร|ของ|เมนู)?ไม่ครบ|ลืม(ออเดอร์|เสิร์ฟ|เสิฟ|ทำ)"
    r"|ตกหล่น|ทวง(อาหาร|ออเดอร์|ถาม|เมนู)|ไม่ได้รับ(อาหาร|ของ|เมนู)|ไม่เข้าระบบ"
)
_ORDER_EXCLUDE = (
    r"รอคิว|ต่อคิว|คิวยาว|จองคิว|ที่จอด|ผิดหวัง|ผิดกฎ|ลืมสั่ง|ถ่าย|ไม่ได้ผิด"
    r"|รอไม่นาน|ไม่ต้องรอ|ไม่ช้า|ไม่นาน|ไม่มีตกหล่น|ไม่ตกหล่น|ไม่เคยผิด|เร็ว|ไวมาก|ไวดี|สั่งผิด|ผิดเอง"
)
ORDER_ISSUE_PATTERN = re.compile(f"{_ORDER_SLOW}|{_ORDER_WRONG}")
ORDER_ISSUE_EXCLUDE = re.compile(_ORDER_EXCLUDE)
ORDER_ISSUE_CATEGORY = "อาหารได้ไม่ตรง/ช้า"
ORDER_ISSUE_MAX_LEN = 250  # ท่อนยาวมากมักเล่าหลายเรื่องปนกัน label จะเพี้ยน


def is_order_issue(chunk: str) -> bool:
    chunk = str(chunk)
    if not (8 <= len(chunk) <= ORDER_ISSUE_MAX_LEN):
        return False
    return bool(ORDER_ISSUE_PATTERN.search(chunk)) and not ORDER_ISSUE_EXCLUDE.search(chunk)


# ---------------------------------------------------------------------------
# เติมหมวดที่ยังไม่ถึง MAX_PER_CATEGORY ด้วยประโยคจริง (top-up)
# ---------------------------------------------------------------------------
# หลัง keyword labeling หมวดเหล่านี้ยังต่ำกว่า 1,200 (Delivery 559, ชมการบริการ 911,
# สอบถามโปรโมชั่น 943) ทั้งที่ยังมีประโยคจริงอีก ~40,000 ประโยคที่ keyword ไม่ได้ติด
# label เลย จึงใช้ pattern เฉพาะหมวด (precision มาก่อน) ขุดเพิ่ม โดย:
#   1. หยิบจากประโยคที่ยังไม่มี label ก่อน
#   2. ถ้ายังไม่พอ ค่อยหยิบจากหมวดที่มีเกินมามาก (> 2 เท่าของ cap) ซึ่งยังไงก็ถูก
#      สุ่มตัดทิ้งอยู่แล้ว — หมวดต้นทางจึงยังเต็ม 1,200 เหมือนเดิม
#   3. หยุดเมื่อหมวดนั้นถึง cap — เรียงประโยคก่อนเสมอ ผลจึงทำซ้ำได้
# ประโยคที่ได้ติด source="wongnai_pattern" เหมือนหมวด "อาหารได้ไม่ตรง/ช้า"
#
# หมายเหตุ: หมวด "สอบถามโปรโมชั่น" ในรีวิวจริงคือการ "พูดถึง" ดีล/ส่วนลด ไม่ใช่การถาม
# ตรงๆ — สอดคล้องกับที่ keyword ติด label หมวดนี้ไว้เดิมอยู่แล้ว
# ---------------------------------------------------------------------------
TOPUP_PATTERNS = {
    "การจัดส่ง (Delivery)": (
        re.compile(
            r"(?i)เดลิเวอ|เดริเวอ|delivery|grab ?food|แกร็บ|แกรบ|grab|line ?man|ไลน์แมน"
            r"|foodpanda|food panda|แพนด้า|robinhood|โรบินฮู้ด|ไรเดอร์|rider|ค่าส่ง|ส่งถึงบ้าน"
            r"|มาส่ง|ส่งมา(ถึง|ที่|ให้)|สั่ง(ผ่าน|ทาง)(แอ|app)|ส่งฟรี|ส่งไว|ส่งเร็ว|ส่งช้า|ส่งนาน"
            r"|แพ็ค(มา|อย่าง)|แพคเกจจิ้ง|packaging"
        ),
        # ป้ายรางวัล/แฮชแท็ก LINEMAN Wongnai Users' Choice และลิงก์ ไม่ใช่เรื่องการจัดส่ง
        re.compile(r"(?i)users.?ch|choice|รางวัล|https?://|www\."),
    ),
    "ชมการบริการ": (
        re.compile(
            r"(พนักงาน|น้องๆ?|พี่ๆ?|เจ้าของร้าน|แม่ค้า|พ่อค้า|เด็กเสิร์ฟ|การบริการ|บริการ)"
            r"(ทุกคน|ที่ร้าน|ในร้าน)?\s?(ก็|ค่อนข้าง)?\s?(บริการ)?"
            r"(ดีมาก|ดี(?!ไม่|ขึ้น|กว่า)|ดีเยี่ยม|น่ารัก|ยิ้มแย้ม|เป็นกันเอง|อัธยาศัยดี|ใส่ใจ"
            r"|เอาใจใส่|สุภาพ|รวดเร็ว|ประทับใจ|ดูแลดี|ดูแลอย่างดี|แนะนำเมนู)"
        ),
        # ตัดประโยคติ/เสนอแนะ เช่น "บริการไม่ค่อยดี", "น่าจะบริการดีกว่านี้"
        re.compile(r"ไม่ค่อย|ไม่ได้|ไม่ดี|แย่|ช้า|ห่วย|ไม่ประทับใจ|หน้าบึ้ง|ไม่สนใจ|ไม่ใส่ใจ"
                   r"|กว่านี้|น่าจะ|ควรจะ|ดีเด่น"),
    ),
    "สอบถามโปรโมชั่น": (
        re.compile(
            r"(?i)โปรโมชั่น|โปรโมชัน|โปร(?=\s|ลด|วัน|เดือน|บัตร|นี้|ดี)|ส่วนลด|ลด\s?\d+\s?(%|เปอร์|บาท)"
            r"|คูปอง|voucher|วอเชอร์|บัตรสมาชิก|สมาชิก(ลด|ได้)|ซื้อ\s?\d\s?แถม\s?\d|1\s?แถม\s?1"
            r"|happy ?hour|ราคาพิเศษ|ดีล(?!ิ)|deal"
        ),
        re.compile(r"ไม่มีโปร|ไม่ลด|โปรแกรม|โปรดิว|โปรไฟล์|โปรตีน"),
    ),
}
TOPUP_MAX_LEN = 300


def topup_category(chunk: str):
    """คืนหมวดแรกที่ pattern top-up จับได้ (ตามลำดับใน TOPUP_PATTERNS) หรือ None"""
    chunk = str(chunk)
    if not (8 <= len(chunk) <= TOPUP_MAX_LEN):
        return None
    for cat, (pattern, exclude) in TOPUP_PATTERNS.items():
        if pattern.search(chunk) and not exclude.search(chunk):
            return cat
    return None


def label_chunk(chunk: str):
    text = clean_text(chunk)
    text_boundary = tokenize_boundary(text)

    and_match = keyword_category_and_match(text_boundary)
    category_detail = and_match or keyword_category_best_match(text_boundary)
    if not category_detail:
        return None

    category_ml = map_to_ml_category(category_detail, None)
    if category_ml not in ML_CATEGORY_CLASSES:
        return None
    return category_ml


# ---------------------------------------------------------------------------
# 3) เสริมหมวดคำถาม (FAQ) ด้วยตัวอย่างที่เขียนขึ้นเอง — รีวิวจริงแทบไม่มีคำถามแบบนี้เลย
# ---------------------------------------------------------------------------
SYNTHETIC_FAQ = {
    "สอบถามข้อมูลร้าน": [
        "ร้านเปิดกี่โมงคะ", "วันนี้เปิดไหมคะ", "ร้านอยู่ตรงไหนคะ ขอพิกัดหน่อย",
        "มีที่จอดรถไหมคะ", "รับบัตรเครดิตไหมคะ", "จองโต๊ะล่วงหน้าได้ไหมคะ",
        "มีกี่สาขาคะ สาขาไหนใกล้ BTS บ้าง", "ร้านปิดกี่โมงคะวันนี้",
        "ต้องจองโต๊ะก่อนไหมคะ วันเสาร์คนเยอะไหม", "เบอร์โทรร้านคืออะไรคะ",
        "มีไวไฟให้ใช้ไหมคะ ขอรหัสหน่อย", "พาสัตว์เลี้ยงเข้าร้านได้ไหมคะ",
        "มีห้องส่วนตัวสำหรับจัดงานไหมคะ", "รับจัดเลี้ยงนอกสถานที่ไหมคะ",
        "เปิดรับสมัครพนักงานไหมคะ", "วันหยุดนักขัตฤกษ์เปิดไหมคะ",
        "มีที่นั่งสำหรับเด็กไหมคะ", "ร้านอยู่ใกล้ MRT สถานีไหนคะ",
        "ต้องรอคิวนานไหมคะช่วงเย็น", "มีบริการจัดส่งไหมคะ ส่งไกลแค่ไหน",
    ],
    "สอบถามโปรโมชั่น": [
        "วันนี้มีโปรโมชั่นอะไรบ้างคะ", "มีส่วนลดสำหรับสมาชิกไหมคะ",
        "ตอนนี้มีคูปองส่วนลดไหมคะ", "โปรลด 50% ยังมีอยู่ไหมคะ",
        "ซื้อ 1 แถม 1 ตอนนี้มีไหมคะ", "มีโปรวันเกิดไหมคะ",
        "สมัครสมาชิกแล้วได้ส่วนลดเท่าไหร่คะ", "โปรโมชั่นเดือนนี้มีอะไรบ้างคะ",
        "มีโค้ดส่วนลดสำหรับสั่งเดลิเวอรี่ไหมคะ", "ลดราคากี่เปอร์เซ็นต์คะช่วงนี้",
    ],
    "สอบถามเมนูอาหาร": [
        "มีเมนูแนะนำไหมคะ อยากลองอะไรใหม่ๆ", "เมนูขายดีของร้านคืออะไรคะ",
        "มีเมนูมังสวิรัติไหมคะ", "ขอเมนูเด็ดของร้านหน่อยค่ะ",
        "มีเมนูสำหรับเด็กไหมคะ", "เมนูนี้เผ็ดมากไหมคะ ปรับความเผ็ดได้ไหม",
        "มีเมนูใหม่ประจำเดือนนี้ไหมคะ", "สั่งอะไรดีคะ ไม่เคยมาก่อน",
        "มีเมนูเซ็ตสำหรับ 4 คนไหมคะ", "มีเครื่องดื่มแอลกอฮอล์ขายไหมคะ",
    ],
}

# ---------------------------------------------------------------------------
# เสริมหมวด "อาหารได้ไม่ตรง/ช้า" ด้วยตัวเอง เพราะในตาราง KEYWORD_CATEGORY_TO_ML_CATEGORY
# มีแค่หมวดละเอียด "ยกเลิก/คืนเงิน" เท่านั้นที่แม็พมาลงคลาสนี้ (แคบมาก) และรีวิวจริง
# ก็แทบไม่มีการเล่าเรื่อง "ออเดอร์ผิด/มาช้า" แบบเจาะจงในเชิงรีวิวหลังกินเสร็จอยู่แล้ว
# (มักบ่นผ่านช่องทางร้องเรียนโดยตรงมากกว่ารีวิว) ทำให้ weak-supervision จาก Wongnai
# ได้ตัวอย่างหมวดนี้น้อยมาก (5 แถว) จึงเขียนตัวอย่างเสริมขึ้นเองให้มีจำนวนพอเทรนได้
# ---------------------------------------------------------------------------
SYNTHETIC_COMPLAINT = {
    "อาหารได้ไม่ตรง/ช้า": [
        "อาหารมาช้ามากค่ะ รอเป็นชั่วโมง", "ได้อาหารผิดจากที่สั่งไปค่ะ",
        "สั่งไก่ทอดแต่ได้หมูทอดมาแทน", "ออเดอร์มาไม่ครบเลยค่ะ ขาดไปหนึ่งจาน",
        "รอนานมากกว่าจะได้อาหาร ทั้งที่คนไม่เยอะ", "สั่งไปตั้งนานอาหารเพิ่งมาตอนใกล้ปิดร้าน",
        "ได้อาหารช้ากว่าที่แจ้งไว้เยอะมากค่ะ", "สั่งเมนูนี้แต่ทำเมนูอื่นมาให้",
        "รายการอาหารที่ได้ไม่ตรงกับที่สั่งในแอปเลยค่ะ", "รอออเดอร์นานจนอาหารเย็นหมดแล้ว",
        "โต๊ะข้างๆ สั่งทีหลังแต่ได้อาหารก่อนเราอีก", "สั่งเผ็ดน้อยแต่ทำมาเผ็ดมากทั้งที่แจ้งแล้ว",
        "ยกเลิกเมนูที่สั่งไปแล้วแต่ไม่มีใครแจ้งลูกค้าเลย", "รอเกือบชั่วโมงอาหารยังไม่มาเลยสักจาน",
        "สั่งผ่านแอปแล้วรายการไม่ตรงกับใบเสร็จเลยค่ะ", "ได้ของหวานผิดรสที่สั่งไป",
        "อาหารจานหลักมาก่อนของทานเล่นที่สั่งไว้ทั้งที่สั่งพร้อมกัน", "รอคิวออกอาหารนานผิดปกติมากวันนี้",
        "ลืมเสิร์ฟเครื่องเคียงที่สั่งไปด้วย", "ได้น้ำจิ้มผิดชนิดจากที่ขอไว้",
        "สั่งไซส์ใหญ่แต่ได้ไซส์เล็กมาแทน", "พนักงานลืมออเดอร์ของโต๊ะเราไปเลย ต้องทวงถามเอง",
        "กว่าจะได้อาหารครบทุกจานก็เกือบจะกลับบ้านพอดี", "สั่งไม่ใส่ผักแต่ได้มาแบบใส่ผักเต็มจาน",
        "ของหวานที่สั่งไปหายไปจากบิลเฉยเลย ไม่ได้เสิร์ฟ", "รายการที่สั่งเดลิเวอรี่มาไม่ครบตามใบเสร็จ",
    ],
}


def build_synthetic_complaint_rows():
    rows = []
    for cat, examples in SYNTHETIC_COMPLAINT.items():
        for ex in examples:
            rows.append({"clean_text": clean_text(ex), "Category": cat, "source": "synthetic_complaint"})
    return rows


def build_synthetic_rows():
    rows = []
    for cat, examples in SYNTHETIC_FAQ.items():
        for ex in examples:
            rows.append({"clean_text": clean_text(ex), "Category": cat, "source": "synthetic_faq"})
    return rows


def main():
    print("กำลังโหลดและตัดรีวิว Wongnai เป็นประโยคย่อย...")
    chunks = load_wongnai_chunks()
    print(f"  ประโยค/ท่อนที่ไม่ซ้ำกันทั้งหมด: {len(chunks)}")

    print("กำลังติด label ด้วยระบบ keyword (weak supervision)...")
    rows = []
    n_pattern = 0
    for chunk in chunks:
        if is_order_issue(chunk):
            rows.append({"clean_text": clean_text(chunk), "Category": ORDER_ISSUE_CATEGORY,
                         "source": "wongnai_pattern"})
            n_pattern += 1
            continue
        cat = label_chunk(chunk)
        if cat is not None:
            rows.append({"clean_text": clean_text(chunk), "Category": cat, "source": "wongnai_real"})
    print(f"  ประโยคที่ติด label ได้จริง: {len(rows)} "
          f"(keyword {len(rows) - n_pattern}, pattern '{ORDER_ISSUE_CATEGORY}' {n_pattern})")

    # --- เติมหมวดที่ยังไม่ถึง cap ด้วยประโยคจริง (ดู TOPUP_PATTERNS) ---
    MAX_PER_CATEGORY = 1200
    labeled_by = {r["clean_text"]: r["Category"] for r in rows}
    counts = pd.Series(list(labeled_by.values())).value_counts()
    surplus = {c for c, n in counts.items() if n > 2 * MAX_PER_CATEGORY}
    need = {c: MAX_PER_CATEGORY - int(counts.get(c, 0)) for c in TOPUP_PATTERNS}
    unlabeled = [ch for ch in chunks if clean_text(ch) not in labeled_by]
    from_surplus = [ch for ch in chunks if labeled_by.get(clean_text(ch)) in surplus]
    moved = {}
    for pool_name, pool in (("unlabeled", unlabeled), ("surplus", from_surplus)):
        for ch in pool:
            cat = topup_category(ch)
            if cat is None or need[cat] <= 0:
                continue
            key = clean_text(ch)
            if key in moved:
                continue
            moved[key] = cat
            need[cat] -= 1
    if moved:
        rows = [r for r in rows if r["clean_text"] not in moved]
        rows.extend({"clean_text": k, "Category": c, "source": "wongnai_pattern"}
                    for k, c in moved.items())
    print("  เติมด้วย pattern (top-up):",
          dict(pd.Series(list(moved.values())).value_counts()) if moved else {})

    rows.extend(build_synthetic_rows())
    rows.extend(build_synthetic_complaint_rows())

    df = pd.DataFrame(rows).drop_duplicates(subset=["clean_text"]).reset_index(drop=True)

    # -----------------------------------------------------------------------
    # จำกัดจำนวนสูงสุดต่อหมวด (cap) เพื่อไม่ให้หมวดที่ weak-supervision จับได้เยอะ
    # เกินจริง (เช่น "สอบถามข้อมูลร้าน" ที่กวาดคำทั่วไปหลายกลุ่มมารวมกัน) ครอบงำ
    # การเทรนจนหมวดเล็กๆ แทบไม่มีน้ำหนักเลย — สุ่มตัดแบบ random_state คงที่เพื่อทำซ้ำได้
    # ---------------------------------------------------------------------------
    parts = []
    for cat, grp in df.groupby("Category"):
        parts.append(grp.sample(min(len(grp), MAX_PER_CATEGORY), random_state=42))
    df = pd.concat(parts).reset_index(drop=True)

    print("\nสรุปจำนวนต่อหมวด (หลัง dedupe + cap ที่ {} ต่อหมวด):".format(MAX_PER_CATEGORY))
    print(df["Category"].value_counts())
    counts = df["Category"].value_counts()
    print(f"\nอัตราส่วนหมวดใหญ่สุด:เล็กสุด = {counts.max() / counts.min():.1f}:1 "
          "(ส่วนที่เหลือชดเชยด้วย class_weight ตอนเทรน ดู 02_train_models.py)")
    print("\nสรุปตาม source:")
    print(df["source"].value_counts())

    OUT.mkdir(exist_ok=True)
    out_path = OUT / "category_dataset_v2.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    data_copy = BASE / "data" / "category_dataset_v2.csv"
    df.to_csv(data_copy, index=False, encoding="utf-8-sig")
    print(f"\nบันทึกไว้ที่: {out_path}  (รวม {len(df)} แถว)")
    print(f"สำเนาที่: {data_copy}")


if __name__ == "__main__":
    main()
