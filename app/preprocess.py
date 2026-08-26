# -*- coding: utf-8 -*-
import re
import unicodedata
import pandas as pd

THAI_NORMALIZE_MAP = {
    "พนง": "พนักงาน",
    "โปร": "โปรโมชั่น",
    "โปรฯ": "โปรโมชั่น",
    "อาหย่อย": "อร่อย",
    "อร่อยย": "อร่อย",
    "อร่อยยย": "อร่อย",
    "ช้ามากก": "ช้ามาก",
    "ช้ามากกก": "ช้ามาก",
    "ห่วยย": "ห่วย",
    "ห่วยยย": "ห่วย",
}

# ---------------------------------------------------------------------------
# FIX: normalize คำถามท้ายประโยคที่คนพิมพ์หลายแบบ ให้เหลือรูปเดียวกันคือ "ไหม"
#
# ปัญหาเดิม: keyword ใน CATEGORY_KEYWORDS หลายหมวดเขียนไว้แค่รูป "...ไหม"
# (เช่น "คิวยาวไหม") แต่ลูกค้าพิมพ์จริงเป็นภาษาพูดหลายแบบ เช่น "คิวยาวมั้ย",
# "จองได้ปะ", "รอนานรึป่าว" ทำให้ keyword ไม่ match เลย แล้วตกไปใช้ label ของ ML
# model ดิบ ซึ่ง choose_action() ไม่รู้จัก สุดท้ายไหลไปจบที่ general_support
# ทั้งที่ควรตอบตรงหมวดได้
#
# แก้โดย normalize ตัวคำถามให้เป็น "ไหม" เสมอ *ก่อน* จะเอาไปเช็ค keyword ทำให้
# keyword ที่มีอยู่แล้วแบบ "...ไหม" ครอบคลุมคำพูดภาษาพูดเหล่านี้ได้ทันที
# โดยไม่ต้องเพิ่ม keyword ซ้ำทุกคำในทุกหมวด
#
# ระวังคำที่สะกดคล้ายกันแต่ไม่ใช่คำถาม เช่น "ป่าวประกาศ", "ปะทะ", "ปะปน",
# "ปะติดปะต่อ" จึงจำกัดการแทนที่ "ป่าว"/"ปะ" เดี่ยวๆ ไว้เฉพาะตอนอยู่ท้ายข้อความ
# หรือตามด้วยช่องว่าง/เครื่องหมายวรรคตอน/คำลงท้ายสุภาพเท่านั้น
# ---------------------------------------------------------------------------
_QUESTION_BOUNDARY = r"(?=$|\s|[?!.,]|ค่ะ|คะ|ครับ|นะ|เลย|บ้าง)"

QUESTION_PARTICLE_PATTERNS = [
    (re.compile(r"มั๊ย"), "ไหม"),
    (re.compile(r"มั้ย"), "ไหม"),
    (re.compile(r"(รึ|หรือ)เปล่า"), "ไหม"),
    (re.compile(r"(รึ|หรือ)ป่าว"), "ไหม"),
    # "ป่าว" / "ปะ" เดี่ยวๆ ท้ายประโยค เช่น "จริงป่าว", "ไปปะ", "ได้ปะ"
    (re.compile(r"ป่าว" + _QUESTION_BOUNDARY), "ไหม"),
    (re.compile(r"ปะ" + _QUESTION_BOUNDARY), "ไหม"),
]


def normalize_question_particles(text: str) -> str:
    for pattern, repl in QUESTION_PARTICLE_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def reduce_repeated_thai_chars(text: str) -> str:
    # มากกกก -> มาก, ดี๊ดี -> ดี๊ดี (ไม่ลบวรรณยุกต์หนักเกินไป)
    text = re.sub(r"([ก-ฮ])\1{2,}", r"\1", text)
    text = re.sub(r"([ะาิีึืุูเแโใไ])\1{2,}", r"\1", text)
    return text


def clean_text(text) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = reduce_repeated_thai_chars(text)
    # ทำก่อนตัดวรรคตอนทิ้ง เพราะ boundary ของ normalize คำถามอ้างอิงเครื่องหมาย
    # วรรคตอน/ช่องว่างที่ยังอยู่ครบตอนนี้
    text = normalize_question_particles(text)
    for k, v in THAI_NORMALIZE_MAP.items():
        text = text.replace(k, v)
    # เก็บไทย อังกฤษ ตัวเลข และเว้นวรรค
    text = re.sub(r"[^0-9A-Za-zก-๙\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def star_to_sentiment(star):
    try:
        star = float(star)
    except Exception:
        return None
    if star <= 2:
        return "negative"
    if star == 3:
        return "neutral"
    return "positive"
