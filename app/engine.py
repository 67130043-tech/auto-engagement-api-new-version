# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

from app.preprocess import clean_text
from app.decision_engine import choose_segment, choose_action, make_reply
from app.keywords_data import (
    NEGATIVE_WORDS, POSITIVE_WORDS, NEGATION_PREFIXES, QUESTION_INDICATORS, CATEGORY_KEYWORDS
)

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


def keyword_sentiment_override(message: str, model_sentiment: str) -> str:
    """
    ใช้ NEGATIVE_WORDS / POSITIVE_WORDS (~986 คำ รวมกัน) จาก keywords_data.py
    1) เช็ค negation ("ไม่"+คำบวก เช่น "ไม่อร่อย","ไม่ชอบ","ไม่ค่อยสะอาด") ก่อนเป็นอันดับแรก -> negative
    2) เช็ค negative_words ตรงๆ
    3) เช็ค positive_words ตรงๆ (ครอบคลุมคำชม/คำน่ากินที่ ML มักเดาผิด เช่น "หิวเลย","น่ากิน")
    4) ถ้าไม่มี keyword แต่มีลักษณะเป็นคำถาม -> neutral
    5) ไม่งั้นเชื่อผล ML เดิม
    """
    text = str(message)

    for prefix in NEGATION_PREFIXES:
        for w in POSITIVE_WORDS:
            if f"{prefix}{w}" in text:
                return "negative"

    if any(w in text for w in NEGATIVE_WORDS):
        return "negative"
    if any(w in text for w in POSITIVE_WORDS):
        return "positive"
    if any(w in text for w in QUESTION_INDICATORS):
        return "neutral"

    return model_sentiment


def keyword_category_override(message: str, model_category: str) -> str:
    """
    ไล่เช็คทีละหมวดตามลำดับใน CATEGORY_KEYWORDS (dict คงลำดับการแทรกใน Python 3.7+)
    หมวดที่เจาะจงกว่า (เช่น ร้องเรียนความสะอาด/คุณภาพอาหาร/บริการ) ถูกจัดไว้ก่อน
    หมวดทั่วไปกว่า (เช่น สอบถามสาขา/ทำเล) ไว้ท้ายๆ เพื่อลดการชนกัน
    """
    text = str(message)
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(w in text for w in keywords):
            return category
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
    # ส่ง message ดิบเข้าไปด้วย เพื่อให้ choose_action เช็ค escalate เคสหนักๆ กับข้อความจริง
    action = choose_action(sentiment, category, segment, message)
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
