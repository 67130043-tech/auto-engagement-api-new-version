# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

from app.preprocess import clean_text, tokenize_boundary
from app.decision_engine import choose_segment, choose_action, make_reply
from app.keywords_data import (
    _NEGATIVE_BOUNDARY, _POSITIVE_BOUNDARY, _QUESTION_BOUNDARY_MAP,
    has_negation_positive, match_boundary_words,
    keyword_category_and_match, keyword_category_best_match,
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


# ---------------------------------------------------------------------------
# FIX (ข้อความจริง "ขอชมว่าน้ำแข็งเย็นทุกก้อน" ถูกตอบเป็น apology_escalate):
# เดิมทั้งสองฟังก์ชันนี้เช็คแบบ "keyword in text" (substring ดิบ) ทำให้คำว่า "แข็ง"
# (อาหารแข็งไป = negative) ไป match ซ้อนอยู่ใน "น้ำแข็ง" (ice, คำละคำกันเลย) เพราะ
# ภาษาไทยไม่มีช่องว่างคั่นคำ substring จึงชนกับคำประสมอื่นได้ง่ายมาก
#
# ตอนนี้เปลี่ยนมาตัดคำด้วย PyThaiNLP ก่อน (preprocess.tokenize_boundary — ใช้
# เครื่องมือเดียวกับที่ระบุไว้ในขอบเขตงานวิจัยข้อ 1.3.2) แล้วเช็ค keyword แบบ
# "ต้องตรงกับคำที่ตัดแล้วทั้งคำ" (word-boundary) แทน substring ดิบ ("แข็ง" จะ
# match เฉพาะตอนตัวตัดคำแยกออกมาเป็นคำของมันเองจริงๆ เช่น "เนื้อแข็งไปหน่อย" ไม่ใช่
# ตอนที่มันเป็นส่วนหนึ่งของคำอื่นอย่าง "น้ำแข็ง") ระบบ tokenize ข้อความลูกค้าแค่
# ครั้งเดียวตรงนี้ ส่วน keyword ทุกตัวถูก tokenize ล่วงหน้าไว้แล้วใน keywords_data.py
# (ไม่ต้อง tokenize ซ้ำทุก request) — ยังคง case-insensitive เหมือนเดิม เพราะ
# tokenize_boundary() แปลงเป็นตัวพิมพ์เล็กให้แล้วในตัว
#
# (เดิมยังมีปัญหาซ้ำสองว่าเรียกด้วย "message" ดิบแทน "text" ที่ clean_text() แล้ว
# — ตอนนี้ตัวเรียกจริงใน predict_message() ยังส่ง text ที่ clean แล้วเหมือนเดิม)
# ---------------------------------------------------------------------------
def keyword_sentiment_override(message: str, model_sentiment: str) -> str:
    """
    ใช้ NEGATIVE_WORDS / POSITIVE_WORDS จาก keywords_data.py (เทียบแบบ word-boundary)
    1) เช็ค negation ("ไม่"+คำบวก เช่น "ไม่อร่อย","ไม่ชอบ","ไม่ค่อยสะอาด") ก่อนเป็นอันดับแรก -> negative
    2) เช็ค negative_words ตรงๆ
    3) เช็ค positive_words ตรงๆ (ครอบคลุมคำชม/คำน่ากินที่ ML มักเดาผิด เช่น "หิวเลย","น่ากิน")
    4) ถ้าไม่มี keyword แต่มีลักษณะเป็นคำถาม -> neutral
    5) ไม่งั้นเชื่อผล ML เดิม
    """
    text_boundary = tokenize_boundary(str(message))

    if has_negation_positive(text_boundary):
        return "negative"
    if match_boundary_words(text_boundary, _NEGATIVE_BOUNDARY):
        return "negative"
    if match_boundary_words(text_boundary, _POSITIVE_BOUNDARY):
        return "positive"
    if match_boundary_words(text_boundary, _QUESTION_BOUNDARY_MAP):
        return "neutral"

    return model_sentiment


def keyword_category_override(message: str, model_category: str) -> str:
    """
    หา category ที่ตรงที่สุดแบบ word-boundary จาก CATEGORY_KEYWORDS โดยเลือกหมวดที่มี
    keyword ยาว/เจาะจงที่สุดที่ match (longest-match-wins) แทนการใช้หมวดแรกตามลำดับ
    การประกาศในไฟล์ เพื่อไม่ให้คำทั่วไป (เช่น "เหม็น" ที่อยู่ในหมวดอาหารด้วย) บัง
    คำที่เจาะจงกว่าของอีกหมวด (เช่น "ห้องน้ำสกปรก")
    """
    text_boundary = tokenize_boundary(str(message))

    # เช็คกติกาแบบ AND ก่อน (สำหรับกรณีคำสำคัญถูกพิมพ์แยกกัน ไม่ติดกันเป็นวลีเดียว
    # เช่น "เอาเค้กวันเกิดไปเองได้ไหม" ดู CATEGORY_AND_KEYWORDS ใน keywords_data.py)
    and_match = keyword_category_and_match(text_boundary)
    if and_match:
        return and_match

    best_match = keyword_category_best_match(text_boundary)
    return best_match or model_category


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

    sentiment = keyword_sentiment_override(text, sentiment_ml)
    category = keyword_category_override(text, category_ml)

    # ---------------------------------------------------------------------
    # FIX: ถ้า keyword rule เปลี่ยน label ไปจากที่ ML เดามา แปลว่าคำตอบสุดท้าย
    # มาจากการ match คำแบบตรงๆ (deterministic) ไม่ใช่ ML เดา จึงไม่ควรใช้ค่า
    # sentiment_confidence / category_confidence เดิม (ซึ่งเป็นความมั่นใจของ
    # label ที่ถูกทิ้งไปแล้ว) มาคำนวณ reply_confidence ต่อ
    # ให้ตั้งเป็นค่าคงที่สูง (keyword match = เชื่อถือได้) แทน
    # ---------------------------------------------------------------------
    KEYWORD_MATCH_CONFIDENCE = 95.0

    sentiment_source = "model"
    category_source = "model"

    if sentiment != sentiment_ml:
        sentiment_confidence = KEYWORD_MATCH_CONFIDENCE
        sentiment_source = "keyword"

    if category != category_ml:
        category_confidence = KEYWORD_MATCH_CONFIDENCE
        category_source = "keyword"

    behavior = get_user_behavior(user_id)
    segment = str(behavior.get("segment", "Regular"))
    action = choose_action(sentiment, category, segment, text)
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
        "sentiment_source": sentiment_source,
        "category_source": category_source,
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
