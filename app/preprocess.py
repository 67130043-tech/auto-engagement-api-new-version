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
