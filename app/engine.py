# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime
import pandas as pd
import joblib

from app.preprocess import clean_text
from app.decision_engine import choose_segment, choose_action, make_reply

def keyword_sentiment_override(message: str, model_sentiment: str) -> str:
    text = str(message)
    negative_words = ["ช้า", "นาน", "ผิด", "หก", "เสีย", "แย่", "ห่วย", "ไม่ประทับใจ", "รอนาน", "เคลม", "เย็นชืด", 
                      "จืด", "เค็มเกิน", "หวานเกิน", "เผ็ดเกิน", "มันเกิน", "คาว", "เหม็น", "บูด", "เสียแล้ว", "ไม่สด", "เก่า", "แข็ง", 
                      "เหนียว", "ดิบ", "ไหม้", "จืดชืด", "รสชาติแปลกๆ", "ช้ามาก", "รอนาน", "พนักงานหยาบคาย", 
                      "ไม่สนใจลูกค้า", "บริการแย่", "ทัศนคติแย่", "หน้าบึ้ง", "ไม่ยิ้ม", "พูดจาไม่ดี", "เอาแต่ใจ" , "สกปรก", "เลอะเทอะ", "มีแมลง", 
                      "มีแมลงสาบ", "กลิ่นเหม็น", "โต๊ะสกปรก", "ห้องน้ำสกปรก", "แออัด", "เสียงดัง", "แพงเกินไป", "ไม่คุ้ม", "ราคาไม่สมเหตุสมผล", "ปริมาณน้อย", 
                      "โกงราคา", "ผิดหวัง", "เสียใจ", "เสียดายเงิน", "ไม่แนะนำ", "ไม่กลับมาอีก", "แย่มาก", "ห่วยแตก", "สยองขวัญ", "หลอกลวง"]
    positive_words = ["อร่อย", "ดีมาก", "ประทับใจ", "ชอบ", "บริการดี", "ขอบคุณ", "อร่อยมาก", "กลมกล่อม", "สดใหม่", "หอม", "รสชาติเข้มข้น", "เข้าเนื้อ", 
                      "กรอบ", "นุ่ม", "พอดี", "ลงตัว", "รสจัดจ้าน", "อร่อยเด็ด", "บริการเร็ว", "ยิ้มแย้ม", "เป็นกันเอง", "ใส่ใจ", "สุภาพ", "ดูแลดี", "พนักงานน่ารัก", "ตอบสนองไว" ,
                      "สะอาด", "บรรยากาศดี", "ตกแต่งสวย", "โปร่งโล่ง", "เงียบสงบ", "น่านั่ง", "สวยงาม", "คุ้มค่า", "ราคาย่อมเยา", "ปริมาณเยอะ", "คุ้มราคา", 
                      "ประทับใจมาก", "จะกลับมาอีก", "แนะนำเลย", "ชอบมาก", "สุดยอด", "เด็ดมาก", "ต้องลอง", "โดนใจ"]
    if any(w in text for w in negative_words):
        return "negative"
    if any(w in text for w in positive_words):
        return "positive"
    return model_sentiment

def keyword_category_override(message: str, model_category: str) -> str:
    text = str(message)

    # 1. โปรโมชั่น / ส่วนลด
    if any(w in text for w in ["โปร", "โปรโมชั่น", "ส่วนลด", "คูปอง", "ลดราคา", "แจกโค้ด"]):
        return "สอบถามโปรโมชั่น"

    # 2. การจัดส่ง (Delivery)
    if any(w in text for w in ["ส่ง", "ไรเดอร์", "เดลิเวอรี่", "Delivery", "ผิดเมนู", "หก", "กล่อง", "ช้า", "นาน"]):
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
    sentiment = str(sentiment_model.predict([text])[0])
    category = str(category_model.predict([text])[0])
    sentiment = keyword_sentiment_override(message, sentiment)
    category = keyword_category_override(message, category)
    behavior = get_user_behavior(user_id)
    segment = str(behavior.get("segment", "Regular"))
    action = choose_action(sentiment, category, segment)
    reply = make_reply(action)

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "user_id": str(user_id),
        "display_name": display_name or "",
        "channel": channel,
        "source": source,
        "message": message,
        "clean_text": text,
        "sentiment": sentiment,
        "category": category,
        "segment": segment,
        "action": action,
        "reply_message": reply,
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
