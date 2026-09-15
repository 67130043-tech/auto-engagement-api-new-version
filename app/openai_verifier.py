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

OPENAI_MODEL = os.environ.get("OPENAI_VERIFIER_MODEL", "gpt-5-nano")
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
        return sentiment, round(confidence, 2)
    except Exception:
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
            max_tokens=30,
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
        return category, round(confidence, 2)
    except Exception:
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
    "คุณคือพนักงานร้านอาหาร/คาเฟ่ กำลังตอบคอมเมนต์ลูกค้าใต้โพสต์ Facebook เป็น"
    "ภาษาไทย น้ำเสียงสุภาพ เป็นกันเอง ลงท้ายด้วย 'ค่ะ' เสมอ\n\n"
    "ระบบวิเคราะห์ข้อความลูกค้ามาแล้วและมี \"คำตอบต้นแบบ\" ที่ถูกต้องตามนโยบายร้าน "
    "(มีข้อมูล/ตัวเลข/เงื่อนไขที่ถูกต้องอยู่แล้ว) ให้คุณเขียนคำตอบใหม่โดยทำตามกฎนี้"
    "อย่างเคร่งครัด:\n"
    "1) ต้องพูดถึงประเด็นที่ลูกค้าพิมพ์มาจริงๆ ให้รู้สึกว่าตอบตรงคำถาม/ความรู้สึกของ"
    "ลูกค้าคนนั้นโดยเฉพาะ ไม่ใช่คำตอบทั่วไปลอยๆ\n"
    "2) ต้องคงข้อมูล ตัวเลข รหัส เงื่อนไข หรือข้อเท็จจริงทุกอย่างจาก \"คำตอบต้นแบบ\" "
    "ไว้ให้ครบถ้วนถูกต้อง ห้ามเปลี่ยนแปลง ห้ามตัดออก\n"
    "3) ห้ามเพิ่มสัญญา ส่วนลด นโยบาย หรือข้อมูลใหม่ใดๆ ที่ไม่ได้อยู่ใน \"คำตอบต้นแบบ\" "
    "หรือข้อความลูกค้าเด็ดขาด (ห้ามแต่งข้อมูลเอง)\n"
    "4) ความยาว 1-3 ประโยคสั้นๆ กระชับ เป็นธรรมชาติเหมือนพนักงานจริงพิมพ์ตอบ ไม่ใช่"
    "ภาษาทางการหรือยาวเกินจำเป็น\n"
    "5) ตอบเป็นภาษาไทยเท่านั้น\n\n"
    "ตอบกลับเป็น JSON เท่านั้น ห้ามมีข้อความอื่นนอกเหนือ JSON รูปแบบ: "
    '{"reply": "ข้อความคำตอบที่เขียนใหม่"}'
)

_MAX_SMART_REPLY_CHARS = 500  # กันเคส LLM ตอบยาวผิดปกติ/หลุด format ไปเป็นข้อความยาว


def generate_smart_reply(message: str, template_reply: str):
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
                    f'คำตอบต้นแบบ (ต้องคงข้อมูล/ตัวเลข/เงื่อนไขไว้ครบ): "{template_reply}"'
                )},
            ],
            response_format={"type": "json_object"},
            max_tokens=200,
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)
        reply = str(data.get("reply", "")).strip()
        if not reply or len(reply) > _MAX_SMART_REPLY_CHARS:
            return None
        return reply
    except Exception:
        # เหตุผลเดียวกับ verify_sentiment()/verify_category() ด้านบน: ห้ามทำให้
        # /make/predict พังเด็ดขาด ไม่ว่า OpenAI จะมีปัญหาอะไรก็ตาม
        return None
