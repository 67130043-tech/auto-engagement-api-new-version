# -*- coding: utf-8 -*-
import re
import unicodedata
import pandas as pd
from pythainlp.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# FIX (ข้อความจริง): "ขอชมว่าน้ำแข็งเย็นทุกก้อน" (คำชม เรื่องน้ำแข็ง) ถูกตอบกลับเป็น
# apology_escalate เพราะ keyword_sentiment_override เดิมเช็คแบบ substring ธรรมดา
# ("แข็ง" in text) ซึ่งคำว่า "แข็ง" (เนื้อ/อาหารแข็งไป = negative) ดันเป็น substring
# ที่ซ้อนอยู่ใน "น้ำแข็ง" (ice) ทำให้คำชมกลายเป็นคำร้องเรียนไปโดยไม่ตั้งใจ
#
# ภาษาไทยไม่มีช่องว่างคั่นคำ การเช็ค "keyword in text" แบบ substring จึงเสี่ยงชน
# กับคำประสมอื่นที่บังเอิญมี keyword นั้นซ้อนอยู่ข้างในเสมอ (เช่น "แข็ง" ใน "น้ำแข็ง",
# "มัน" ใน "มันฝรั่ง"/"มันเทศ") แก้โดยตัดคำด้วย PyThaiNLP (เครื่องมือที่ระบุไว้ในขอบเขต
# งานวิจัยข้อ 1.3.2 อยู่แล้ว) ก่อน แล้วเช็ค keyword แบบ "ต้องตรงกับคำที่ตัดแล้วทั้งคำ
# (word-boundary) เท่านั้น" แทนการเช็ค substring ดิบ ทำให้ "แข็ง" จะ match เฉพาะตอนที่
# ตัวตัดคำแยก "แข็ง" ออกมาเป็นคำของมันเองจริงๆ (เช่น "เนื้อแข็งไปหน่อย") ไม่ match ตอนที่
# มันเป็นส่วนหนึ่งของคำอื่นอย่าง "น้ำแข็ง" ที่ตัวตัดคำจะรวมเป็นคำเดียวกันเสมอ
# ---------------------------------------------------------------------------
_BOUNDARY_SEP = "\x1f"


def tokenize_boundary(text: str) -> str:
    """
    ตัดคำ (PyThaiNLP, engine=newmm) แล้วคืนสตริงคั่นด้วยตัวคั่นพิเศษรอบทุกคำ เช่น
    "น้ำแข็งเย็นทุกก้อน" -> "\\x1fน้ำแข็ง\\x1fเย็น\\x1fทุก\\x1fก้อน\\x1f"
    ใช้เทียบหา keyword ทั้งคำ (word-boundary) แทนการเช็ค substring ดิบ:
    ทั้งข้อความลูกค้าและ keyword ต้องผ่านฟังก์ชันนี้ก่อนเทียบกันเสมอ (ดู engine.py)
    """
    if not text:
        return _BOUNDARY_SEP
    tokens = [t.lower() for t in word_tokenize(str(text), engine="newmm") if t and t.strip()]
    return _BOUNDARY_SEP + _BOUNDARY_SEP.join(tokens) + _BOUNDARY_SEP

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
    # ---------------------------------------------------------------------
    # FIX: unicodedata.normalize("NFKC", ...) แยก "ำ" (SARA AM, U+0E33) ออกเป็น
    # 2 ตัวอักษร "ํ"+"า" (NIKHAHIT U+0E4D + SARA AA U+0E32) เสมอ (เป็นพฤติกรรม
    # มาตรฐานของ NFKC กับอักษรไทยกลุ่มนี้) ทำให้คำที่มี "ำ" ทุกคำ (เช่น "น้ำแข็ง",
    # "ทำงาน", "จำได้") ถูกเปลี่ยนรูปแบบสตริงไปแบบเงียบๆ ทั้งที่หน้าตาเหมือนเดิมทุก
    # ประการเวลาพิมพ์ออกมา ผลคือ PyThaiNLP (tokenize_boundary ด้านล่างและใน
    # keywords_data.py) หาไม่เจอว่า "น้ำแข็ง" เป็นคำในดิกชันนารี เพราะสตริงที่ส่ง
    # เข้าไปจริงๆ กลายเป็น "น้ํา"+"แข็ง" (ตัดคำผิดเป็น 2 คำ) ทำให้ "แข็ง" (คำเดี่ยว
    # แปลว่าอาหารแข็งไป = negative) โดน tokenize แยกออกมาแล้ว match ผิดเป็นคำ
    # ร้องเรียน ทั้งที่ข้อความจริงเป็นคำชม เช่น "ขอชมว่าน้ำแข็งเย็นทุกก้อน"
    # แก้โดยรวม "ํ"+"า" กลับเป็น "ำ" ทันทีหลัง NFKC (ผลลัพธ์สุดท้ายเหมือนสตริง
    # ต้นฉบับทุกตัวอักษร ไม่กระทบการ normalize อย่างอื่นที่ NFKC ทำ เช่น เลข/ตัวอักษร
    # เต็มความกว้าง -> ครึ่งความกว้าง)
    # ---------------------------------------------------------------------
    text = text.replace("ํา", "ำ")
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
