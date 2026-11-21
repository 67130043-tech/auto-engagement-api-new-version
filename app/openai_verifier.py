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

OPENAI_MODEL = os.environ.get("OPENAI_VERIFIER_MODEL", "gpt-5-mini")
OPENAI_TIMEOUT_SECONDS = 8
VALID_SENTIMENTS = {"positive", "negative", "neutral"}

_client = None
_client_init_failed = False
def _token_args(visible_tokens):
    """พารามิเตอร์จำกัดความยาวที่ถูกต้องตามรุ่นโมเดล
    - รุ่น reasoning (gpt-5*, o-series) ไม่รับ max_tokens และนับ token ที่ใช้ "คิด" รวมในเพดาน
      จึงใช้ max_completion_tokens พร้อมเผื่อที่ให้การคิด และขอ reasoning_effort ต่ำสุด
    - รุ่นอื่นใช้ max_tokens ตามเดิม"""
    m = OPENAI_MODEL.lower()
    if m.startswith(("gpt-5", "o1", "o3", "o4")):
        return {"max_completion_tokens": visible_tokens + 2000, "reasoning_effort": "minimal"}
    return {"max_tokens": visible_tokens}


_stats = {"ok": 0, "failed": 0, "last_error": None}


def _record(ok, err=None):
    _stats["ok" if ok else "failed"] += 1
    if err is not None:
        _stats["last_error"] = f"{type(err).__name__}: {str(err)[:160]}"


def llm_status():
    """สถานะชั้นที่ 3 (LLM) สำหรับแสดงที่หน้า / ของ API — ไม่เปิดเผยค่า key"""
    return {"api_key_set": bool(os.environ.get("OPENAI_API_KEY", "").strip()), "model": OPENAI_MODEL,
            "client_ready": _client is not None, "calls_ok": _stats["ok"], "calls_failed": _stats["failed"],
            "last_error": _stats["last_error"], "rejected": _stats.get("rejected", 0),
            "last_rejection": _stats.get("last_rejection")}


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
            **_token_args(30),
            # หมายเหตุ: โมเดลตระกูล gpt-5 (รวม gpt-5-nano) ไม่รองรับพารามิเตอร์
            # temperature ค่าอื่นนอกจากค่า default (1) — ถ้าใส่ temperature=0 แบบเดิม
            # จะโดน API ตอบ 400 error ทุกครั้ง (ถูก try/except ด้านล่างจับเงียบๆ กลาย
            # เป็นไม่ได้ผลอะไรเลยโดยไม่มี error ให้เห็น) จึงตัดพารามิเตอร์นี้ออกไปเลย
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
        _record(True)
        return sentiment, round(confidence, 2)
    except Exception as _e:
        _record(False, _e)
        # ครอบคลุมทุกกรณี: network error, timeout, rate limit (429), API key ผิด,
        # ตอบมาไม่ใช่ JSON, ไม่มี key "sentiment"/"confidence" ฯลฯ — ตั้งใจไม่ระบุ
        # exception type เจาะจง เพราะเป้าหมายคือ "ห้ามทำให้ /make/predict พังเด็ดขาด"
        # ถ้าเรียก OpenAI มีปัญหาอะไรก็ตาม ให้ fallback เงียบๆ ไปใช้ ML เดิมเสมอ
        return None, None


# ---------------------------------------------------------------------------
# verify_category() — เพิ่มเข้ามาคู่กับ verify_sentiment() ด้านบน (แนวทางที่ 1)
# ใช้ pattern เดียวกันเป๊ะ: เรียกเฉพาะตอน category_source == "model" (keyword ไม่
# match อะไรเลย ต้องเชื่อผล ML เดาเอง) แล้ว category_confidence ต่ำกว่า threshold
# เท่านั้น — ดูจุดเรียกจริงใน engine.predict_message()
#
# ให้ LLM เลือกจาก 45 หมวดละเอียดชุดเดียวกับ CATEGORY_KEYWORDS ใน keywords_data.py
# (ไม่ใช่แค่ 10 คลาสของ ML) เพราะ choose_action() ใช้ category_detail (45 หมวด) นี้
# ตัดสิน reply template แบบเจาะจง (เช่น wifi_info, send_hours) ถ้าให้ LLM ตอบแค่
# 10 คลาสกว้างๆ จะยังคงตกไปที่ general_support เหมือนเดิม ไม่ได้ช่วยอะไรเพิ่ม
# ---------------------------------------------------------------------------
VALID_CATEGORIES = [
    "การจัดส่ง (Delivery)", "ข้อเสนอแนะทั่วไป", "จองโต๊ะ (Reservation)",
    "จัดเลี้ยง/อีเวนต์", "ชมความสะอาด", "ชมบรรยากาศร้าน", "ชมพนักงาน",
    "ชมรสชาติอาหาร", "ช่องทางการชำระเงิน", "ซื้อกลับบ้าน (Takeaway)",
    "ที่ตั้งร้าน (Location)", "ปัญหาบัตรสมาชิก/แต้ม", "ปัญหาแอพ/ระบบสั่งอาหาร",
    "ยกเลิก/คืนเงิน", "รีวิว/ให้คะแนน", "ร้องเรียนการบริการ", "ร้องเรียนความสะอาด",
    "ร้องเรียนคุณภาพอาหาร", "ร้องเรียนบรรยากาศ", "ร้องเรียนบิล/ยอดเงินผิด",
    "ร้องเรียนอุณหภูมิ/แอร์", "สมัครงาน", "สมาชิก/สะสมแต้ม", "สอบถาม Corkage",
    "สอบถาม WiFi", "สอบถาม/ร้องเรียนห้องน้ำ", "สอบถามข้อจำกัดด้านอาหาร",
    "สอบถามคิวรอโต๊ะ", "สอบถามช่องทางติดต่อ", "สอบถามดนตรีสด/กิจกรรม",
    "สอบถามที่จอดรถ", "สอบถามบุฟเฟ่ต์", "สอบถามปรับระดับความเผ็ด/รส",
    "สอบถามพาสัตว์เลี้ยงเข้าร้าน", "สอบถามพื้นที่จัดส่ง", "สอบถามยอดสั่งขั้นต่ำ",
    "สอบถามราคา", "สอบถามสาขา/ทำเล", "สอบถามสิ่งอำนวยความสะดวกสำหรับเด็ก",
    "สอบถามห้องส่วนตัว/VIP", "สอบถามเครื่องดื่มแอลกอฮอล์", "สอบถามเมนู",
    "สอบถามแฟรนไชส์", "สอบถามโปรโมชั่น", "เวลาเปิด-ปิดร้าน", "ทั่วไป",
]
_VALID_CATEGORY_SET = set(VALID_CATEGORIES)

_CATEGORY_SYSTEM_PROMPT = (
    "คุณเป็นผู้ช่วยจัดหมวดหมู่ข้อความ/คอมเมนต์ของลูกค้าร้านอาหาร/คาเฟ่ภาษาไทย "
    "ให้เลือกหมวดที่ตรงที่สุดเพียงหมวดเดียวจากรายการนี้เท่านั้น (พิมพ์ให้ตรงตัวสะกด "
    "เป๊ะๆ ตามที่ให้มา ห้ามแต่งคำใหม่): " + ", ".join(VALID_CATEGORIES) + ". "
    "ถ้าไม่มีหมวดไหนตรงเลยจริงๆ ให้ตอบ \"ทั่วไป\" "
    "ตอบกลับเป็น JSON เท่านั้น ห้ามมีข้อความอื่นนอกเหนือ JSON รูปแบบ: "
    '{"category": "ชื่อหมวดตามรายการ", "confidence": จำนวนเต็ม 0 ถึง 100}'
)


def verify_category(text: str):
    """
    เรียก OpenAI ให้จัด category ของ text อีกรอบ (ใช้เป็นความเห็นที่สอง เมื่อ ML
    ล้วนๆ มั่นใจต่ำ) คืนค่า (category, confidence) ถ้าเรียกสำเร็จและ parse ได้ตาม
    รูปแบบที่คาดไว้ (และ category ต้องอยู่ใน VALID_CATEGORIES เท่านั้น ป้องกัน LLM
    ตอบชื่อหมวดที่ไม่มีจริงมาแล้ว choose_action() หา match ไม่เจอ) หรือคืนค่า
    (None, None) ถ้าเรียกไม่ได้/ตอบมาไม่ถูกรูปแบบ — ผู้เรียก (engine.predict_message())
    ต้อง fallback ไปใช้ผล ML เดิมเองเสมอเมื่อได้ (None, None)
    """
    client = _get_client()
    if client is None:
        return None, None

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _CATEGORY_SYSTEM_PROMPT},
                {"role": "user", "content": f'ข้อความ: "{text}"'},
            ],
            response_format={"type": "json_object"},
            **_token_args(30),
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)
        category = str(data.get("category", "")).strip()
        if category not in _VALID_CATEGORY_SET:
            return None, None
        try:
            confidence = float(data.get("confidence", 75.0))
        except (TypeError, ValueError):
            confidence = 75.0
        confidence = max(0.0, min(100.0, confidence))
        _record(True)
        return category, round(confidence, 2)
    except Exception as _e:
        _record(False, _e)
        # เหตุผลเดียวกับ verify_sentiment() ด้านบน: ห้ามทำให้ /make/predict พังเด็ดขาด
        return None, None


# ---------------------------------------------------------------------------
# generate_smart_reply() — FIX (ผู้ใช้ท้วงหลังแก้ keyword ทีละเคสมาหลายรอบว่า
# "อยากให้ฉลาดกว่านี้มากๆ ขี้เกียจมานั่งแก้ไขอะไรแบบนี้เยอะแล้ว"): ปัญหาจริงของ
# สถาปัตยกรรมเดิมคือ REPLY_TEMPLATES (decision_engine.py) เป็นข้อความตายตัว 100%
# ต่อ 1 action เดียวกันเสมอ ไม่ว่าลูกค้าจะพิมพ์มาต่างกันแค่ไหนก็ตอบเหมือนเดิมทุก
# ตัวอักษร — ทำให้ทุกครั้งที่เจอคอมเมนต์แปลกใหม่ที่ระบบ "ตัดสิน action ถูกแล้ว" แต่
# คำตอบยัง generic เกินไป ต้องมานั่งเพิ่ม keyword/แก้ logic เองอยู่ร่ำไป
#
# แนวทางที่เลือก (คุยกับผู้ใช้แล้ว): "กลางๆ" คือยังใช้ keyword+ML ตัดสิน sentiment/
# category/action เหมือนเดิมทุกประการ (แม่นยำ วัดผลได้ ตรงกับ thesis methodology ที่
# ทำมาตลอด ไม่ต้องเปลี่ยน pipeline หลัก) แต่เปลี่ยน "ขั้นตอนสุดท้าย" จากการหยิบ
# REPLY_TEMPLATES ตรงๆ มาเป็นให้ LLM เขียนคำตอบใหม่ที่ตอบสนองข้อความลูกค้าจริงๆ
# โดยยึด REPLY_TEMPLATES เดิมเป็น "ข้อเท็จจริง/เงื่อนไขที่ต้องคงไว้" (เช่น รหัส wifi,
# ค่า corkage, ยอดขั้นต่ำ) ห้ามให้ LLM แต่งข้อมูลใหม่เอง เพื่อกันการหลอน (hallucinate)
# ที่จะเป็นอันตรายกับธุรกิจจริง (เช่น สัญญาส่วนลด/นโยบายที่ไม่มีจริง)
#
# เปิดใช้งานเป็นค่าเริ่มต้นเสมอ (ต่างจาก verify_sentiment/verify_category ที่เรียก
# เฉพาะบางเงื่อนไข) เพราะนี่คือ "ขั้นตอนตอบกลับ" ไม่ใช่ตัวช่วยเสริม — ควบคุมได้ผ่าน
# ENABLE_SMART_REPLY=false (environment variable) ถ้าต้องการปิดเพื่อคุมค่าใช้จ่าย
# API (ทุกคอมเมนต์จะเรียก OpenAI 1 ครั้งเพิ่มจากเดิม ต่างจาก verify_* ที่เรียกเฉพาะ
# เคสไม่มั่นใจเท่านั้น) — ปิดไม่ได้ตั้งค่า/เรียกไม่สำเร็จ จะ fallback กลับไปใช้
# REPLY_TEMPLATES เดิมเงียบๆ ทันที ไม่ทำให้ /make/predict พังเด็ดขาดเหมือนกัน
# ---------------------------------------------------------------------------
def smart_reply_enabled() -> bool:
    return os.environ.get("ENABLE_SMART_REPLY", "true").strip().lower() != "false"


_REPLY_SYSTEM_PROMPT = (
    "คุณคือแอดมินเพจร้านอาหาร กำลังตอบคอมเมนต์ลูกค้าใต้โพสต์ Facebook เป็นภาษาไทย "
    "น้ำเสียงสุภาพ อบอุ่น เป็นกันเองแบบคนจริงพิมพ์ ลงท้ายด้วย 'ค่ะ'/'นะคะ'\n\n"
    "คุณจะได้รับ: ข้อความลูกค้า, ผลวิเคราะห์ของระบบ (อารมณ์/หมวด/การตอบที่ควรทำ), "
    "\"คำตอบต้นแบบ\" ตามนโยบายร้าน และ \"ข้อมูลร้าน\" ที่ยืนยันแล้ว เขียนคำตอบใหม่ตามกฎ:\n"
    "1) ตอบตรงสิ่งที่ลูกค้าพูดจริง: เอ่ยถึงเมนู/เรื่องที่ลูกค้ากล่าวถึง (เช่น ส้มตำ ความเผ็ด การรอ บิล) "
    "ถ้ามีทั้งคำชมและคำติ ให้ขอบคุณเรื่องที่ชมและขออภัย/รับเรื่องที่ติ\n"
    "2) ทำตาม \"การตอบที่ควรทำ\" และคงข้อเท็จจริง ตัวเลข เงื่อนไขใน \"คำตอบต้นแบบ\" ไว้ "
    "แต่เรียบเรียงใหม่ให้เป็นธรรมชาติได้ ไม่ต้องคัดลอกทั้งประโยค\n"
    "3) ข้อเท็จจริงเกี่ยวกับร้าน (เวลาเปิด ที่อยู่ เบอร์ ช่องทางชำระเงิน ฯลฯ) ใช้ได้เฉพาะจาก "
    "\"ข้อมูลร้าน\" หรือ \"คำตอบต้นแบบ\" เท่านั้น ถ้าลูกค้าถามเรื่องที่ไม่มีข้อมูล ให้บอกว่าจะให้ทีมงาน"
    "ตรวจสอบและแจ้งกลับ ห้ามเดา ห้ามแต่งเมนู ราคา ส่วนลด หรือสัญญาใหม่\n"
    "4) เมื่อลูกค้าบ่น ให้ขออภัยและรับเรื่อง แต่ไม่ยืนยันข้อกล่าวหาเป็นข้อเท็จจริง "
    "(เช่น 'ขออภัยที่ส้มตำรสเค็มเกินไปนะคะ' แทน 'ส้มตำเค็มจริงๆ')\n"
    "5) ถ้าลูกค้าประชด ให้ตอบตามเจตนาที่แท้จริง (คำบ่น) ไม่ตอบรับคำชมตามตัวอักษร\n"
    "6) ใช้เฉพาะข้อมูลร้านที่เกี่ยวกับสิ่งที่ลูกค้าพูด ไม่ยัดข้อมูลอื่น (เช่น ถามเดลิเวอรี่ ไม่ต้องพูดเวลารอคิว) "
    "ห้ามสัญญาสิ่งที่ไม่มีในคำตอบต้นแบบ เช่น 'กำลังปรับระบบ' 'จะติดต่อกลับ' 'มีเมนูใหม่ทุกวัน'\n"
    "7) ลิงก์/URL ต้องคัดลอกครบทุกตัวอักษร ห้ามย่อ\n"    "9) ห้ามเรียกชื่อคนที่อยู่ในคอมเมนต์ เพราะมักเป็นชื่อเพื่อนที่ถูกแท็ก ไม่ใช่คนที่คอมเมนต์ (ใช้ 'ลูกค้า' หรือไม่ต้องเรียกชื่อ)\n"
    "8) สั้น กระชับ ไม่เกิน 2 ประโยค (ยกเว้นต้องใส่ที่อยู่/ลิงก์) ไม่ทางการเกินไป ใช้อีโมจิได้ไม่เกิน 1 ตัว "
    "ตรวจคำสะกดให้ถูกต้อง\n\n"
    "ตอบกลับเป็น JSON เท่านั้น ห้ามมีข้อความอื่นนอกเหนือ JSON รูปแบบ: "
    '{"reply": "ข้อความคำตอบที่เขียนใหม่"}'
)


import re as _re

_URL_RE = _re.compile(r"https?://\S+")
_NUM_RE = _re.compile(r"\d+(?:[.:,]\d+)*")


def _reply_problem(reply, template_reply, message):
    """ตรวจคำตอบจาก LLM ก่อนใช้จริง คืนเหตุผล (str) ถ้าไม่ผ่าน หรือ None ถ้าใช้ได้
    - ลิงก์ในคำตอบต้นแบบต้องอยู่ครบตัวอักษร (LLM ชอบย่อ URL เป็น ...)
    - ตัวเลขทุกตัวในคำตอบต้องมีอยู่ในคำตอบต้นแบบ/ข้อมูลร้าน/ข้อความลูกค้า (กันแต่งราคา เวลา จำนวน)"""
    for url in _URL_RE.findall(template_reply):
        if url.rstrip(").,") not in reply:
            return "url_changed"
    allowed = template_reply + " " + _shop_facts_text() + " " + (message or "")
    for num in _NUM_RE.findall(reply):
        if num not in allowed:
            return f"new_number:{num}"
    return None


def _shop_facts_text():
    try:
        from app.shop_config import get_shop_info
        info = get_shop_info()
        labels = {"opening_hours": "เวลาเปิด", "closed_days": "วันหยุด", "address": "ที่อยู่", "map_link": "แผนที่",
                  "phone": "เบอร์โทร", "line_id": "LINE", "payment_methods": "ชำระเงิน",
                  "min_delivery_order": "ยอดสั่งเดลิเวอรี่ขั้นต่ำ", "queue_wait_estimate": "เวลารอคิวโดยประมาณ",
                  "corkage_fee": "ค่าเปิดขวด", "menu_note": "หมายเหตุเมนู", "promotion_note": "โปรโมชั่น",
                  "shop_name": "ชื่อร้าน", "cuisine": "ประเภทอาหาร", "signature_menu": "เมนูแนะนำ", "price_range": "ราคา",
                  "spice_policy": "ความเผ็ด", "branches": "สาขา", "parking": "ที่จอดรถ", "delivery_platforms": "เดลิเวอรี่",
                  "takeaway": "สั่งกลับบ้าน", "reservation": "จองโต๊ะ", "pet_policy": "สัตว์เลี้ยง", "kids_facility": "เด็ก",
                  "private_room": "ห้องส่วนตัว", "alcohol": "แอลกอฮอล์", "buffet": "บุฟเฟ่ต์", "live_music": "ดนตรีสด",
                  "membership": "สมาชิก", "catering": "จัดเลี้ยง"}
        return "; ".join(f"{labels[k]}: {info[k]}" for k in labels if info.get(k))
    except Exception:
        return ""


_MAX_SMART_REPLY_CHARS = 500  # กันเคส LLM ตอบยาวผิดปกติ/หลุด format ไปเป็นข้อความยาว


def generate_smart_reply(message: str, template_reply: str, context: dict = None):
    """
    ให้ LLM เขียนคำตอบใหม่โดยยึด template_reply (จาก REPLY_TEMPLATES) เป็นข้อเท็จจริง
    ที่ต้องคงไว้ แต่ปรับให้ตอบสนองข้อความ message จริงๆ ของลูกค้าแทนคำตอบตายตัว
    คืนค่าเป็นข้อความคำตอบ (str) ถ้าเรียกสำเร็จ หรือ None ถ้าเรียกไม่ได้/ตอบผิดรูปแบบ/
    ว่างเปล่า/ยาวผิดปกติ — ผู้เรียก (engine.predict_message()) ต้อง fallback ไปใช้
    template_reply เดิมทันทีเมื่อได้ None (ไม่ทำให้ /make/predict พังเด็ดขาด)
    """
    if not smart_reply_enabled():
        return None

    client = _get_client()
    if client is None:
        return None

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _REPLY_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f'ข้อความลูกค้า: "{message}"\n'
                    + (f'ผลวิเคราะห์ของระบบ (ใช้กำหนดน้ำเสียง เช่น ขอบคุณ/ขอโทษ/ให้ข้อมูล): '
                       f'อารมณ์={context.get("sentiment")}, หมวด={context.get("category")}, การตอบ={context.get("action")}\n'
                       if context else '')
                    + f'คำตอบต้นแบบ (ต้องคงข้อมูล/ตัวเลข/เงื่อนไขไว้ครบ): "{template_reply}"\n'
                    + f'ข้อมูลร้าน: {_shop_facts_text()}'
                )},
            ],
            response_format={"type": "json_object"},
            **_token_args(200),
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)
        reply = str(data.get("reply", "")).strip()
        if not reply or len(reply) > _MAX_SMART_REPLY_CHARS:
            return None
        problem = _reply_problem(reply, template_reply, message)
        if problem:
            _stats["rejected"] = _stats.get("rejected", 0) + 1
            _stats["last_rejection"] = problem
            _record(True)
            return None  # ผู้เรียกใช้คำตอบต้นแบบแทน
        _record(True)
        return reply
    except Exception as _e:
        _record(False, _e)
        # เหตุผลเดียวกับ verify_sentiment()/verify_category() ด้านบน: ห้ามทำให้
        # /make/predict พังเด็ดขาด ไม่ว่า OpenAI จะมีปัญหาอะไรก็ตาม
        return None


_RELEVANCE_SYSTEM_PROMPT = (
    "คุณช่วยแอดมินเพจร้านอาหารคัดคอมเมนต์ใต้โพสต์ Facebook ว่าควรให้ร้านตอบหรือไม่ "
    "จัดเป็น 1 ใน 3 แบบ:\n"
    "- restaurant: พูดถึงร้าน/อาหาร/เครื่องดื่ม/บริการ/ราคา/โปรโมชั่น/การสั่ง/การจัดส่ง หรือถามข้อมูลร้าน "
    "(ทั้งชม ติ ถาม รวมถึงประชด)\n"
    "- invite_friend: แท็กหรือชวนเพื่อนมากิน/มาร้าน โดยไม่มีคำถามหรือความเห็นต่อร้าน (เช่น 'ไปกินกันเถอะ', '@ชื่อ พาไปหน่อย')\n"
    "- unrelated: คุยเรื่องส่วนตัวกับคนอื่น บอกรัก แซว ล้อเล่น หยอกกัน ตอบรับ/ปฏิเสธคำชวนของเพื่อน "
    "(เช่น 'ไม่ไปกับคนชื่อ...หรอกนะ 😜', 'ได้เลย เดี๋ยวไปด้วย') สแปม โฆษณาอื่น หรือไม่เกี่ยวกับร้านเลย "
    "(เช่น '@ชื่อ รักเมียนะ', 'ฝันดีนะ')\n"
    "ถ้าเป็นการตอบกลับใต้คอมเมนต์ของลูกค้าคนอื่น (ไม่ใช่ของเพจ) ให้ถือว่าเป็นการคุยกันเองระหว่างผู้ใช้ "
    "เว้นแต่ข้อความพูดถึงร้าน อาหาร บริการ หรือถามร้านอย่างชัดเจน\n"
    "คำว่า 'ไม่' หรืออีโมจิแซว ไม่ได้แปลว่าเป็นคำติร้าน ให้ดูว่ากำลังพูดกับใคร\n"
    "ชื่อคนที่ขึ้นต้นคอมเมนต์มักเป็นการแท็กเพื่อน ให้ดูเนื้อหาที่เหลือเป็นหลัก "
    'ตอบกลับเป็น JSON เท่านั้น รูปแบบ: {"type": "restaurant|invite_friend|unrelated"}'
)


def check_relevance(text: str, parent_message: str = "", parent_from_page=None):
    """คืน 'restaurant' / 'invite_friend' / 'unrelated' หรือ None ถ้าเรียก LLM ไม่ได้
    (ผู้เรียกต้องถือว่า None = ตอบตามปกติ เพื่อไม่ให้พลาดคอมเมนต์ของลูกค้าจริง)"""
    client = _get_client()
    if client is None or not text or not str(text).strip():
        return None
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": _RELEVANCE_SYSTEM_PROMPT},
                      {"role": "user", "content": (
                          (f'คอมเมนต์นี้เป็นการตอบกลับใต้คอมเมนต์{"ของเพจร้าน" if str(parent_from_page).lower() in ("true", "1") else "อื่น"}: '
                           f'"{parent_message}"\n' if parent_message else "")
                          + f'คอมเมนต์: "{text}"')}],
            response_format={"type": "json_object"},
            **_token_args(20),
        )
        t = str(json.loads(resp.choices[0].message.content).get("type", "")).strip()
        _record(True)
        return t if t in ("restaurant", "invite_friend", "unrelated") else None
    except Exception as _e:
        _record(False, _e)
        return None

