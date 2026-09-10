# -*- coding: utf-8 -*-
"""
ตัวช่วยตรวจสอบ sentiment ซ้ำด้วย OpenAI (LLM) — เรียกเฉพาะกรณีที่ ML ล้วนๆ (ไม่มี
keyword ไหน match เลย ต้องเชื่อผล ML เดาเอง — ดู engine.predict_message()) แล้ว
sentiment_confidence ต่ำกว่า threshold ที่ตกลงกันไว้ (60%) เท่านั้น ไม่เรียกทุกข้อความ
เพื่อคุมค่าใช้จ่าย API ให้ต่ำที่สุด (เคสที่ keyword match แล้วมั่นใจอยู่แล้วไม่ต้องเรียกซ้ำ)

**ไม่แตะ/เปลี่ยน interface ของ Make.com HTTP module เดิมเลย** — endpoint, request/
response schema ของ /make/predict เหมือนเดิมทุกอย่าง ฟังก์ชันนี้ถูกเรียก "ข้างใน"
predict_message() เท่านั้น ฝั่ง Make ไม่รู้เลยว่ามีการเรียก OpenAI เกิดขึ้นระหว่างทาง
(ตรงตามที่ตกลงกันไว้ว่าต้องคงโมดูล HTTP เดิมของ Make)

ปิดใช้งานอัตโนมัติถ้าไม่ได้ตั้งค่า OPENAI_API_KEY (environment variable บน Render) —
จะ fallback กลับไปใช้ผลจาก ML เดิมทุกประการ ไม่ทำให้ระบบพังหรือหยุดทำงานถ้ายังไม่ได้
ตั้งค่า/ตั้งค่าผิด/เรียกไม่สำเร็จ (network error, rate limit, timeout, ตอบมาไม่ใช่
รูปแบบที่คาดไว้ ฯลฯ) — ผู้เรียกต้องเช็ค return เป็น (None, None) แล้ว fallback เอง
"""
import os
import json

OPENAI_MODEL = os.environ.get("OPENAI_VERIFIER_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT_SECONDS = 8
VALID_SENTIMENTS = {"positive", "negative", "neutral"}

_client = None
_client_init_failed = False


def _get_client():
    """สร้าง OpenAI client แบบ lazy (สร้างครั้งแรกที่ใช้จริง) แล้วเก็บไว้ใช้ซ้ำ
    ถ้าไม่มี OPENAI_API_KEY หรือสร้างไม่สำเร็จ จะจำไว้ (ไม่ลองใหม่ทุก request) แล้ว
    คืน None เพื่อให้ verify_sentiment() ข้ามไปเรียก ML fallback ทันที"""
    global _client, _client_init_failed
    if _client is not None or _client_init_failed:
        return _client
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        _client_init_failed = True
        return None
    try:
        from openai import OpenAI
        _client = OpenAI(api_key=api_key, timeout=OPENAI_TIMEOUT_SECONDS)
    except Exception:
        _client_init_failed = True
        _client = None
    return _client


_SYSTEM_PROMPT = (
    "คุณเป็นผู้ช่วยวิเคราะห์ความรู้สึก (sentiment) ของข้อความลูกค้าร้านอาหาร/คาเฟ่ "
    "ภาษาไทย ให้จัดข้อความเป็นหนึ่งใน 3 ประเภทเท่านั้นคือ positive, negative, neutral "
    "(neutral คือคำถามล้วนๆ หรือข้อความที่ไม่ได้แสดงความรู้สึกชัดเจนไปทางบวก/ลบ) "
    "ตอบกลับเป็น JSON เท่านั้น ห้ามมีข้อความอื่นนอกเหนือ JSON รูปแบบ: "
    '{"sentiment": "positive หรือ negative หรือ neutral", "confidence": จำนวนเต็ม 0 ถึง 100}'
)


def verify_sentiment(text: str):
    """
    เรียก OpenAI ให้จัด sentiment ของ text อีกรอบ (ใช้เป็นความเห็นที่สอง เมื่อ ML
    ล้วนๆ มั่นใจต่ำ) คืนค่า (sentiment, confidence) ถ้าเรียกสำเร็จและ parse ได้ตาม
    รูปแบบที่คาดไว้ หรือคืนค่า (None, None) ถ้าเรียกไม่ได้/ตอบมาไม่ถูกรูปแบบ — ผู้เรียก
    (engine.predict_message()) ต้อง fallback ไปใช้ผล ML เดิมเองเสมอเมื่อได้ (None, None)
    """
    client = _get_client()
    if client is None:
        return None, None

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f'ข้อความ: "{text}"'},
            ],
            response_format={"type": "json_object"},
            max_tokens=30,
            temperature=0,
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)
        sentiment = str(data.get("sentiment", "")).strip().lower()
        if sentiment not in VALID_SENTIMENTS:
            return None, None
        try:
            confidence = float(data.get("confidence", 75.0))
        except (TypeError, ValueError):
            confidence = 75.0
        confidence = max(0.0, min(100.0, confidence))
        return sentiment, round(confidence, 2)
    except Exception:
        # ครอบคลุมทุกกรณี: network error, timeout, rate limit (429), API key ผิด,
        # ตอบมาไม่ใช่ JSON, ไม่มี key "sentiment"/"confidence" ฯลฯ — ตั้งใจไม่ระบุ
        # exception type เจาะจง เพราะเป้าหมายคือ "ห้ามทำให้ /make/predict พังเด็ดขาด"
        # ถ้าเรียก OpenAI มีปัญหาอะไรก็ตาม ให้ fallback เงียบๆ ไปใช้ ML เดิมเสมอ
        return None, None
