# -*- coding: utf-8 -*-

def normalize_label(x):
    return str(x).strip().lower() if x is not None else ""

def choose_segment(total_messages=0, negative_count=0, complaint_count=0, inactive_days=0):
    total_messages = int(total_messages or 0)
    negative_count = int(negative_count or 0)
    complaint_count = int(complaint_count or 0)
    inactive_days = int(inactive_days or 0)

    if inactive_days >= 30:
        return "Lost Customer"
    if negative_count >= 3 or complaint_count >= 3:
        return "At-Risk"
    if total_messages >= 20 and negative_count <= 2:
        return "VIP"
    if total_messages >= 5:
        return "Active"
    return "Regular"

def choose_action(sentiment, category, segment):
    s = normalize_label(sentiment)
    c = str(category or "")
    seg = str(segment or "")

    if s == "negative" and ("จัดส่ง" in c or "Delivery" in c or "ช้า" in c or "อาหารได้ไม่ตรง" in c):
        if seg in ["VIP", "At-Risk"]:
            return "apology_coupon_escalate"
        return "apology_check_order"
    if s == "negative":
        return "apology_escalate"
    if "โปรโมชั่น" in c or "Promotion" in c or "โปร" in c:
        return "send_promotion"
    if "เมนู" in c or "Menu" in c:
        return "send_menu"
    if "จอง" in c or "Reservation" in c:
        return "reservation_support"
    if s == "positive":
        return "thank_you"
    return "general_support"

REPLY_TEMPLATES = {
    "apology_coupon_escalate": "ขออภัยอย่างสูงค่ะ ทางร้านจะรีบตรวจสอบรายการนี้ให้ทันที และขอมอบคูปองส่วนลดสำหรับการสั่งครั้งถัดไปค่ะ",
    "apology_check_order": "ขออภัยในความไม่สะดวกค่ะ ทางร้านจะรีบตรวจสอบรายการอาหาร/การจัดส่งให้ทันทีค่ะ",
    "apology_escalate": "ขออภัยอย่างสูงค่ะ ทางร้านรับทราบปัญหาแล้ว และจะส่งเรื่องให้ผู้จัดการตรวจสอบเพื่อปรับปรุงบริการค่ะ",
    "send_promotion": "ตอนนี้ทางร้านมีโปรโมชั่นพิเศษค่ะ สนใจรับรายละเอียดโปรโมชันหรือเมนูแนะนำเพิ่มเติมไหมคะ",
    "send_menu": "ยินดีค่ะ ทางร้านมีเมนูแนะนำหลายรายการ เดี๋ยวส่งรายละเอียดเมนูยอดนิยมให้ลูกค้าค่ะ",
    "reservation_support": "ยินดีค่ะ ลูกค้าต้องการจองโต๊ะวันและเวลาใด แจ้งจำนวนที่นั่งได้เลยค่ะ",
    "thank_you": "ขอบคุณมากค่ะ ทางร้านดีใจที่ลูกค้าประทับใจ หวังว่าจะได้ดูแลลูกค้าอีกครั้งนะคะ",
    "general_support": "ขอบคุณที่ติดต่อมาค่ะ ทางร้านยินดีให้บริการ ต้องการสอบถามข้อมูลเพิ่มเติมเรื่องใดแจ้งได้เลยค่ะ",
}

def make_reply(action):
    return REPLY_TEMPLATES.get(action, REPLY_TEMPLATES["general_support"])
