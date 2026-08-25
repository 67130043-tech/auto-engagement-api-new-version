# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime
import numpy as np 
import pandas as pd
import joblib

from app.preprocess import clean_text
from app.decision_engine import choose_segment, choose_action, make_reply

# เพิ่มใหม่สำหรับคำนวณความแม่นยำ
def predict_with_confidence(model, text: str):
    
    pred = str(model.predict([text])[0])
    scores = model.decision_function([text])[0]
    scores = np.atleast_1d(scores)
    if scores.shape[0] == 1:
        confidence = float(1 / (1 + np.exp(-abs(scores[0])))) * 100
    else:
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()
        classes = list(model.named_steps["clf"].classes_) if hasattr(model, "named_steps") else list(model.classes_)
        idx = classes.index(pred)
        confidence = float(probs[idx]) * 100
    return pred, round(confidence, 2)


# ---------------------------------------------------------------------------
# FIX #1: เพิ่มการจัดการคำปฏิเสธ (negation) ก่อนหน้านี้ "ไม่อร่อย", "ไม่ชอบ",
# "ไม่หอม", "ไม่นุ่ม" ฯลฯ จะถูกจับด้วย positive_words แบบ substring match
# (เพราะ "อร่อย" อยู่ใน "ไม่อร่อย") ทำให้ข้อความบ่น/ร้องเรียน ถูกตัดสินว่าเป็น
# "positive" และระบบไปตอบแบบขอบคุณ (thank_you) กลับลูกค้าที่กำลังไม่พอใจ
# ---------------------------------------------------------------------------
NEGATION_PREFIXES = ["ไม่ค่อย", "ไม่ได้", "ไม่สู้", "ไม่"]  # เรียงจากยาว->สั้น เพื่อจับวลีที่เจาะจงกว่าก่อน

def keyword_sentiment_override(message: str, model_sentiment: str) -> str:
    text = str(message)
    negative_words = ["ช้า", "นาน", "ผิด", "หก", "เสีย", "แย่", "ห่วย", "ไม่ประทับใจ", "รอนาน", "เคลม", "เย็นชืด", 
                      "จืด", "เค็มเกิน", "หวานเกิน", "เผ็ดเกิน", "มันเกิน", "คาว", "เหม็น", "บูด", "เสียแล้ว", "ไม่สด", "เก่า", "แข็ง", 
                      "เหนียว", "ดิบ", "ไหม้", "จืดชืด", "รสชาติแปลกๆ", "ช้ามาก", "รอนาน", "พนักงานหยาบคาย", 
                      "ไม่สนใจลูกค้า", "บริการแย่", "ทัศนคติแย่", "หน้าบึ้ง", "ไม่ยิ้ม", "พูดจาไม่ดี", "เอาแต่ใจ" , "สกปรก", "เลอะเทอะ", "มีแมลง", 
                      "มีแมลงสาบ", "กลิ่นเหม็น", "โต๊ะสกปรก", "ห้องน้ำสกปรก", "แออัด", "เสียงดัง", "แพงเกินไป", "ไม่คุ้ม", "ราคาไม่สมเหตุสมผล", "ปริมาณน้อย", 
                      "โกงราคา", "ผิดหวัง", "เสียใจ", "เสียดายเงิน", "ไม่แนะนำ", "ไม่กลับมาอีก", "แย่มาก", "ห่วยแตก", "สยองขวัญ", "หลอกลวง",
                      "ไม่อร่อย", "ไม่ชอบ"]  # เพิ่มคำที่พบบ่อยแบบตรงๆ ไว้ด้วย กันไว้อีกชั้น
    positive_words = ["อร่อย", "ดีมาก", "ประทับใจ", "ชอบ", "บริการดี", "ขอบคุณ", "อร่อยมาก", "กลมกล่อม", "สดใหม่", "หอม", "รสชาติเข้มข้น", "เข้าเนื้อ", 
                      "กรอบ", "นุ่ม", "พอดี", "ลงตัว", "รสจัดจ้าน", "อร่อยเด็ด", "บริการเร็ว", "ยิ้มแย้ม", "เป็นกันเอง", "ใส่ใจ", "สุภาพ", "ดูแลดี", "พนักงานน่ารัก", "ตอบสนองไว" ,
                      "สะอาด", "บรรยากาศดี", "ตกแต่งสวย", "โปร่งโล่ง", "เงียบสงบ", "น่านั่ง", "สวยงาม", "คุ้มค่า", "ราคาย่อมเยา", "ปริมาณเยอะ", "คุ้มราคา", 
                      "ประทับใจมาก", "จะกลับมาอีก", "แนะนำเลย", "ชอบมาก", "สุดยอด", "เด็ดมาก", "ต้องลอง", "โดนใจ"]
    # คำที่บ่งบอกว่าเป็นคำถามข้อมูลทั่วไป ไม่ใช่การแสดงความรู้สึก
    question_indicators = ["ไหม", "กี่โมง", "เท่าไหร่", "ยังไง", "อย่างไร", "ที่ไหน", "ตรงไหน", "หรือเปล่า", "รึเปล่า", "มั้ย"]

    # 1) เช็คคำปฏิเสธ + คำบวก ก่อนเป็นอันดับแรก เช่น "ไม่อร่อย", "ไม่ชอบ", "ไม่หอม"
    #    => ถือว่าเป็น negative ทันที ไม่ว่าจะมีคำบวก/ลบอื่นปนอยู่หรือไม่
    for prefix in NEGATION_PREFIXES:
        for w in positive_words:
            if f"{prefix}{w}" in text:
                return "negative"

    if any(w in text for w in negative_words):
        return "negative"
    if any(w in text for w in positive_words):
        return "positive"
    
    # ถ้าไม่มี keyword บวก/ลบ แต่มีลักษณะเป็นคำถาม → ให้เป็น neutral แทนที่จะเชื่อ ML เฉยๆ
    if any(w in text for w in question_indicators):
        return "neutral"
    
    return model_sentiment

# ---------------------------------------------------------------------------
# FIX #2: ตัดคำกว้างๆ อย่าง "ช้า" / "นาน" ออกจากหมวด "การจัดส่ง (Delivery)"
# เพราะทำให้ข้อความร้องเรียนบริการหน้าร้าน เช่น "รอนานมาก พนักงานไม่สนใจ"
# ถูกจัดเข้าหมวด Delivery ผิดๆ (ชนกับ "ร้องเรียนการบริการ" ที่เช็คทีหลัง)
# แทนที่ด้วยวลีเจาะจงที่บ่งบอกบริบท delivery จริงๆ เช่น "ส่งช้า", "รอของนาน"
# ---------------------------------------------------------------------------
def keyword_category_override(message: str, model_category: str) -> str:
    text = str(message)

    # 1. โปรโมชั่น / ส่วนลด
    if any(w in text for w in ["โปร", "โปรโมชั่น", "ส่วนลด", "คูปอง", "ลดราคา", "แจกโค้ด"]):
        return "สอบถามโปรโมชั่น"

    # 2. การจัดส่ง (Delivery) — ใช้วลีเจาะจงแทนคำกว้างเดี่ยวๆ อย่าง "ช้า"/"นาน"
    if any(w in text for w in ["ส่ง", "ไรเดอร์", "เดลิเวอรี่", "Delivery", "ผิดเมนู", "หก", "กล่อง",
                                "ส่งช้า", "มาช้า", "รอของนาน", "อาหารมาไม่ครบ", "ส่งผิด"]):
        return "การจัดส่ง (Delivery)"

    # 3. สอบถามเมนู
    if any(w in text for w in ["เมนู", "แนะนำ", "ขายดี", "มีอะไรบ้าง", "เมนูแนะนำ", "เมนูใหม่"]):
        return "สอบถามเมนู"

    # 4. จองโต๊ะ (Reservation)
    if any(w in text for w in ["จอง", "โต๊ะ", "สำรองที่นั่ง", "ที่นั่ง", "จองคิว"]):
        return "จองโต๊ะ (Reservation)"

    # 5. เวลาเปิด-ปิดร้าน
    if any(w in text for w in ["เปิดกี่โมง", "ปิดกี่โมง", "เปิดวันไหน", "เวลาทำการ", "วันหยุด", "หยุดวันไหน"]):
        return "เวลาเปิด-ปิดร้าน"

    # 6. ที่ตั้ง / แผนที่
    if any(w in text for w in ["ที่อยู่", "ร้านอยู่ไหน", "แผนที่", "ทางไป", "location", "GPS", "พิกัด"]):
        return "ที่ตั้งร้าน (Location)"

    # 7. ที่จอดรถ
    if any(w in text for w in ["ที่จอดรถ", "จอดรถ", "ลานจอด", "parking"]):
        return "สอบถามที่จอดรถ"

    # 8. ช่องทางการชำระเงิน
    if any(w in text for w in ["โอนเงิน", "จ่ายเงิน", "บัตรเครดิต", "พร้อมเพย์", "PromptPay", "เงินสด", "QR code", "สแกนจ่าย"]):
        return "ช่องทางการชำระเงิน"

    # 9. ร้องเรียนคุณภาพอาหาร
    if any(w in text for w in ["ไม่อร่อย", "จืด", "เค็มเกิน", "บูด", "ไม่สด", "เหม็น", "แข็ง", "ไหม้", "รสชาติแปลก"]):
        return "ร้องเรียนคุณภาพอาหาร"

    # 10. ร้องเรียนบริการ
    if any(w in text for w in ["บริการแย่", "พนักงานหยาบคาย", "ไม่สนใจลูกค้า", "หน้าบึ้ง", "รอนานมาก", "ไม่พอใจ"]):
        return "ร้องเรียนการบริการ"

    # 11. สอบถามราคา
    if any(w in text for w in ["ราคาเท่าไหร่", "กี่บาท", "ราคา", "แพงไหม", "cost", "price"]):
        return "สอบถามราคา"

    # 12. แฟรนไชส์ / ร่วมธุรกิจ
    if any(w in text for w in ["แฟรนไชส์", "ลงทุน", "franchise", "เปิดสาขา", "ร่วมทุน"]):
        return "สอบถามแฟรนไชส์"

    # 13. สมัครงาน
    if any(w in text for w in ["สมัครงาน", "รับสมัคร", "ตำแหน่งงาน", "หางาน", "รับพนักงาน"]):
        return "สมัครงาน"

    # 14. จัดเลี้ยง / อีเวนต์
    if any(w in text for w in ["จัดเลี้ยง", "จัดงาน", "อีเวนต์", "งานเลี้ยง", "catering", "จัดปาร์ตี้"]):
        return "จัดเลี้ยง/อีเวนต์"

    # 15. อาหารฮาลาล / มังสวิรัติ / แพ้อาหาร
    if any(w in text for w in ["ฮาลาล", "มังสวิรัติ", "เจ", "แพ้อาหาร", "แพ้กุ้ง", "แพ้ถั่ว", "vegan", "halal"]):
        return "สอบถามข้อจำกัดด้านอาหาร"

    # 16. ช่องทางติดต่อ
    if any(w in text for w in ["เบอร์โทร", "ติดต่อ", "line id", "เพจ", "Facebook", "IG", "โทรหา"]):
        return "สอบถามช่องทางติดต่อ"

    # 17. รีวิว / ให้คะแนน
    if any(w in text for w in ["รีวิว", "ให้คะแนน", "ดาว", "review", "คอมเมนต์"]):
        return "รีวิว/ให้คะแนน"

    # 18. ยกเลิก / คืนเงิน
    if any(w in text for w in ["ยกเลิกออเดอร์", "คืนเงิน", "refund", "ขอเงินคืน", "ยกเลิกคำสั่งซื้อ"]):
        return "ยกเลิก/คืนเงิน"

    # 19. สมาชิก / สะสมแต้ม
    if any(w in text for w in ["สมาชิก", "สะสมแต้ม", "แลกแต้ม", "member", "point", "บัตรสมาชิก"]):
        return "สมาชิก/สะสมแต้ม"

    # 20. ห่อกลับ / ซื้อกลับบ้าน
    if any(w in text for w in ["ห่อกลับ", "ซื้อกลับ", "takeaway", "ใส่กล่อง", "แพ็คกลับ"]):
        return "ซื้อกลับบ้าน (Takeaway)"
        
    # 21. สอบถามสาขา/ทำเล
    if any(w in text for w in ["สาขา", "ทำเล", "ที่ตั้ง", "อยู่ตรงไหน", "ใกล้", "BTS", "MRT", "แผนที่"]):
        return "สอบถามสาขา/ทำเล"

    # 23. คิวรอโต๊ะ / Waitlist
    if any(w in text for w in ["รอคิว", "คิวยาวไหม", "ต้องรอกี่นาที", "คิวนาน", "รอโต๊ะ"]):
        return "สอบถามคิวรอโต๊ะ"

    # 24. เครื่องปรับอากาศ/อุณหภูมิ
    if any(w in text for w in ["แอร์", "ร้อนมาก", "เย็นเกินไป", "อากาศในร้าน"]):
        return "ร้องเรียนอุณหภูมิ/แอร์"

    # 25. ห้องน้ำ
    if any(w in text for w in ["ห้องน้ำ", "toilet", "restroom"]):
        return "สอบถาม/ร้องเรียนห้องน้ำ"

    # 26. WiFi
    if any(w in text for w in ["wifi", "ไวไฟ", "อินเทอร์เน็ต", "รหัสไวไฟ"]):
        return "สอบถาม WiFi"

    # 27. พาสัตว์เลี้ยงเข้าร้าน
    if any(w in text for w in ["สัตว์เลี้ยง", "หมา", "แมว", "pet friendly", "พาน้องหมาเข้า"]):
        return "สอบถามพาสัตว์เลี้ยงเข้าร้าน"

    # 28. เด็ก/เก้าอี้เด็ก
    if any(w in text for w in ["เก้าอี้เด็ก", "พาเด็กมา", "เมนูเด็ก", "kids chair", "baby chair"]):
        return "สอบถามสิ่งอำนวยความสะดวกสำหรับเด็ก"

    # 29. ห้องส่วนตัว/ห้อง VIP
    if any(w in text for w in ["ห้องส่วนตัว", "ห้องวีไอพี", "ห้อง VIP", "ห้องประชุม", "private room"]):
        return "สอบถามห้องส่วนตัว/VIP"

    # 30. นำเครื่องดื่ม/เค้กมาเอง (Corkage)
    if any(w in text for w in ["นำเครื่องดื่มมาเอง", "corkage", "เค้กมาเอง", "เก็บค่าคอร์กเกจ"]):
        return "สอบถาม Corkage"

    # 31. ปรับระดับความเผ็ด/รสชาติตามสั่ง
    if any(w in text for w in ["ไม่เผ็ด", "เผ็ดน้อย", "เผ็ดมาก", "ปรับรส", "สั่งพิเศษ"]):
        return "สอบถามปรับระดับความเผ็ด/รส"

    # 32. บุฟเฟ่ต์/เติมฟรี
    if any(w in text for w in ["บุฟเฟ่ต์", "เติมฟรี", "buffet", "all you can eat"]):
        return "สอบถามบุฟเฟ่ต์"

    # 33. บัตรสะสมแต้มหาย/มีปัญหา
    if any(w in text for w in ["บัตรสมาชิกหาย", "แต้มหาย", "แต้มไม่เข้า", "point ไม่ขึ้น"]):
        return "ปัญหาบัตรสมาชิก/แต้ม"

    # 34. บิลผิด/ยอดเงินไม่ตรง
    if any(w in text for w in ["บิลผิด", "ยอดไม่ตรง", "คิดเงินผิด", "เก็บเงินเกิน"]):
        return "ร้องเรียนบิล/ยอดเงินผิด"

    # 35. แอพ/ระบบสั่งอาหารมีปัญหา
    if any(w in text for w in ["แอพค้าง", "สั่งผ่านแอพไม่ได้", "ระบบล่ม", "กดสั่งไม่ได้", "เว็บค้าง"]):
        return "ปัญหาแอพ/ระบบสั่งอาหาร"

    # 36. ข้อเสนอแนะทั่วไป
    if any(w in text for w in ["อยากแนะนำ", "ข้อเสนอแนะ", "feedback", "อยากให้ปรับปรุง"]):
        return "ข้อเสนอแนะทั่วไป"

    # 37. ชมบรรยากาศร้าน
    if any(w in text for w in ["บรรยากาศดี", "ตกแต่งสวย", "ร้านสวย", "ถ่ายรูปสวย", "อินเทรนด์"]):
        return "ชมบรรยากาศร้าน"

    # 38. ชมพนักงาน
    if any(w in text for w in ["พนักงานน่ารัก", "บริการประทับใจ", "พนักงานดูแลดี", "ยิ้มแย้มดี"]):
        return "ชมพนักงาน"

    # 39. เครื่องดื่มแอลกอฮอล์/เบียร์
    if any(w in text for w in ["เบียร์", "ไวน์", "แอลกอฮอล์", "ค็อกเทล", "เหล้า"]):
        return "สอบถามเครื่องดื่มแอลกอฮอล์"

    # 40. ดนตรีสด/กิจกรรมในร้าน
    if any(w in text for w in ["ดนตรีสด", "กิจกรรมพิเศษ", "live music", "การแสดง"]):
        return "สอบถามดนตรีสด/กิจกรรม"

    # 41. พื้นที่ให้บริการจัดส่ง
    if any(w in text for w in ["ส่งถึงไหม", "พื้นที่จัดส่ง", "แถวนี้ส่งไหม", "delivery area"]):
        return "สอบถามพื้นที่จัดส่ง"

    # 42. ยอดสั่งขั้นต่ำสำหรับ Delivery
    if any(w in text for w in ["สั่งขั้นต่ำ", "ยอดขั้นต่ำ", "minimum order"]):
        return "สอบถามยอดสั่งขั้นต่ำ"
    
    return model_category

BASE = Path(__file__).resolve().parents[1]
MODELS = BASE / "models"
OUT = BASE / "outputs"
LOG_PATH = OUT / "prediction_log.csv"

_sentiment_model = None
_category_model = None
_behavior = None

def load_resources():
    global _sentiment_model, _category_model, _behavior
    if _sentiment_model is None:
        _sentiment_model = joblib.load(MODELS / "sentiment_model.joblib")
    if _category_model is None:
        _category_model = joblib.load(MODELS / "category_model.joblib")
    if _behavior is None:
        path = OUT / "customer_behavior_features.csv"
        if path.exists():
            _behavior = pd.read_csv(path, encoding="utf-8-sig")
        else:
            _behavior = pd.DataFrame(columns=["customer_id", "total_messages", "positive_count", "negative_count", "complaint_count", "inactive_days", "favorite_hour", "segment"])
    return _sentiment_model, _category_model, _behavior

def get_user_behavior(user_id: str):
    _, _, behavior = load_resources()
    user_id = str(user_id)
    row = behavior[behavior["customer_id"].astype(str) == user_id]
    if len(row) == 0:
        return {
            "total_messages": 0,
            "positive_count": 0,
            "negative_count": 0,
            "complaint_count": 0,
            "inactive_days": 0,
            "favorite_hour": None,
            "segment": "Regular",
        }
    r = row.iloc[0].to_dict()
    # ถ้า segment ไม่มี ให้คำนวณใหม่
    if not r.get("segment") or pd.isna(r.get("segment")):
        r["segment"] = choose_segment(r.get("total_messages", 0), r.get("negative_count", 0), r.get("complaint_count", 0), r.get("inactive_days", 0))
    return r

def predict_message(user_id: str, message: str, channel: str = "manual", display_name: str = "", source: str = "api"):
    sentiment_model, category_model, _ = load_resources()
    text = clean_text(message)

    sentiment_ml, sentiment_confidence = predict_with_confidence(sentiment_model, text)
    category_ml, category_confidence = predict_with_confidence(category_model, text)

    sentiment = keyword_sentiment_override(message, sentiment_ml)
    category = keyword_category_override(message, category_ml)

    behavior = get_user_behavior(user_id)
    segment = str(behavior.get("segment", "Regular"))
    action = choose_action(sentiment, category, segment)
    reply = make_reply(action)

    reply_confidence = round((sentiment_confidence + category_confidence) / 2, 2)

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "user_id": str(user_id),
        "display_name": display_name or "",
        "channel": channel,
        "source": source,
        "message": message,
        "clean_text": text,
        "sentiment": sentiment,
        "sentiment_confidence": sentiment_confidence,
        "category": category,
        "category_confidence": category_confidence,
        "segment": segment,
        "action": action,
        "reply_message": reply,
        "reply_confidence": reply_confidence,
        "behavior": {
             "total_messages": int(behavior.get("total_messages", 0) or 0),
            "positive_count": int(behavior.get("positive_count", 0) or 0),
            "negative_count": int(behavior.get("negative_count", 0) or 0),
            "complaint_count": int(behavior.get("complaint_count", 0) or 0),
            "inactive_days": int(behavior.get("inactive_days", 0) or 0),
            "favorite_hour": None if pd.isna(behavior.get("favorite_hour", None)) else int(behavior.get("favorite_hour")),
        }
    }
    save_log(result)
    return result

def save_log(result: dict):
    OUT.mkdir(exist_ok=True)
    row = {k: v for k, v in result.items() if k != "behavior"}
    row.update({f"behavior_{k}": v for k, v in result.get("behavior", {}).items()})
    df = pd.DataFrame([row])
    if LOG_PATH.exists():
        df.to_csv(LOG_PATH, mode="a", index=False, header=False, encoding="utf-8-sig")
    else:
        df.to_csv(LOG_PATH, index=False, encoding="utf-8-sig")
