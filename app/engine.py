# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime
import pandas as pd
import joblib

from app.preprocess import clean_text
from app.decision_engine import choose_segment, choose_action, make_reply

def keyword_sentiment_override(message: str, model_sentiment: str) -> str:
    text = str(message)
    negative_words = ["ช้า", "นาน", "ผิด", "หก", "เสีย", "แย่", "ห่วย", "ไม่ประทับใจ", "รอนาน", "เคลม", "เย็นชืด"]
    positive_words = ["อร่อย", "ดีมาก", "ประทับใจ", "ชอบ", "บริการดี", "ขอบคุณ"]
    if any(w in text for w in negative_words):
        return "negative"
    if any(w in text for w in positive_words):
        return "positive"
    return model_sentiment

def keyword_category_override(message: str, model_category: str) -> str:
    text = str(message)
    if any(w in text for w in ["โปร", "โปรโมชั่น", "ส่วนลด", "คูปอง"]):
        return "สอบถามโปรโมชั่น"
    if any(w in text for w in ["ส่ง", "ไรเดอร์", "เดลิเวอรี่", "Delivery", "ผิดเมนู", "หก", "กล่อง", "ช้า", "นาน"]):
        return "การจัดส่ง (Delivery)"
    if any(w in text for w in ["เมนู", "แนะนำ", "ขายดี"]):
        return "สอบถามเมนู"
    if any(w in text for w in ["จอง", "โต๊ะ", "สำรองที่นั่ง", "ที่นั่ง"]):
        return "จองโต๊ะ (Reservation)"
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
