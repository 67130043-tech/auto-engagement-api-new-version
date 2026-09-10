# -*- coding: utf-8 -*-
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

from app.preprocess import clean_text, tokenize_boundary, tokenize_words
from app.decision_engine import choose_segment, choose_action, make_reply
from app.keywords_data import (
    _NEGATIVE_BOUNDARY, _POSITIVE_BOUNDARY, _QUESTION_BOUNDARY_MAP,
    _NEGATIVE_TOKENS, _POSITIVE_TOKENS,
    has_negation_positive, has_negation_of_negative,
    match_boundary_words, count_nonoverlapping_matches,
    keyword_category_and_match, keyword_category_best_match,
)
from app.openai_verifier import verify_sentiment

# เกณฑ์เรียก OpenAI เป็นความเห็นที่สอง: เฉพาะตอนไม่มี keyword ไหน match เลย (ต้อง
# เชื่อผล ML เดาเอง) แล้ว sentiment_confidence (ที่ calibrate แล้ว) ต่ำกว่านี้เท่านั้น
# (ตกลงกับผู้ใช้ไว้ที่ 60% — ดู comment ใน predict_message() จุดที่เรียกใช้จริง)
SENTIMENT_VERIFY_THRESHOLD = 60.0

# ---------------------------------------------------------------------------
# FIX (พัฒนาความแม่นยำของ sentiment_confidence/category_confidence): เดิมฟังก์ชัน
# นี้คำนวณ confidence เองด้วยสูตร sigmoid (2 คลาส) หรือ softmax (หลายคลาส) ทับ
# model.decision_function() เพราะ LinearSVC ไม่มี predict_proba() ในตัว — สูตรนี้
# ให้ค่าที่ "หน้าตาเหมือน" ความน่าจะเป็น (อยู่ในช่วง 0-100%) แต่ไม่เคยถูกสอบเทียบ
# (calibrate) กับข้อมูลจริงเลยว่าตอนโมเดลบอกมั่นใจ 90% แม่นจริง ~90% ของเวลาไหม
# ค่าที่ได้จึงเชื่อถือไม่ได้ในเชิงสถิติ (ดูปัญหานี้ชัดๆ ตอน full-system sentiment
# accuracy ตกฮวบทั้งที่ confidence ที่รายงานออกมายังดูสูงอยู่)
#
# ตอนนี้ 02_train_models.py ห่อโมเดลด้วย CalibratedClassifierCV (Platt scaling)
# แล้ว ทำให้มี predict_proba() ที่สอบเทียบแล้วจริงๆ ใช้งานได้ ฟังก์ชันนี้เลยเปลี่ยน
# มาเรียก predict_proba() โดยตรงเป็นค่าเริ่มต้น — เก็บสูตร sigmoid/softmax เดิมไว้
# เป็น fallback เฉยๆ เผื่อกรณีโหลด model.joblib รุ่นเก่าที่ยังไม่ได้ calibrate
# (ไม่มี predict_proba) เพื่อไม่ให้ระบบพังถ้า deploy ไม่ครบ/ไม่พร้อมกัน
# ---------------------------------------------------------------------------
def predict_with_confidence(model, text: str):

    pred = str(model.predict([text])[0])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([text])[0]
        classes = list(model.classes_)
        idx = classes.index(pred)
        confidence = float(probs[idx]) * 100
        return pred, round(confidence, 2)

    # Fallback แบบเดิม (uncalibrated) — ใช้เฉพาะกับโมเดลรุ่นเก่าที่ไม่มี predict_proba
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

    1) เช็ค negation ("ไม่"+คำบวก เช่น "ไม่อร่อย","ไม่ชอบ","ไม่ค่อยสะอาด") ก่อนเป็นอันดับแรก
       -> negative เสมอ (สัญญาณชัดเจนที่สุด ไม่ต้องนับคะแนน)
    2) นับจำนวนคำลบ (neg_hits) และคำบวก (pos_hits) ที่ match ทั้งหมดในข้อความ (ไม่ใช่แค่
       เช็คว่ามีคำใดคำหนึ่งไหมแบบเดิม) แล้วหักคะแนน neg_hits ลง 1 ถ้าเจอ "ไม่"+คำลบ
       (เช่น "ไม่แย่") เพราะความหมายถูกปฏิเสธไปแล้ว
    3) ถ้าไม่มีทั้งคำลบและคำบวกเลย -> เช็คว่าเป็นคำถามไหม (neutral) ไม่งั้นเชื่อผล ML เดิม
    4) ถ้ามีแค่ฝั่งเดียว (neg_hits>0 หรือ pos_hits>0 อย่างใดอย่างหนึ่ง) -> เชื่อฝั่งนั้น
       (พฤติกรรมเหมือนเดิมทุกประการสำหรับคอมเมนต์สั้นประเด็นเดียว)
    5) ถ้ามีทั้งสองฝั่ง (ข้อความ/รีวิวยาวที่มีทั้งคำชมและคำติปนกัน) -> ใช้เสียงส่วนใหญ่
       ฝั่งที่เยอะกว่าอย่างน้อย 2 เท่าชนะ ถ้าก้ำกึ่งกันให้ตอบ "neutral" แทน (ความเห็นแบบ
       ผสม ไม่ใช่ชมล้วนหรือติล้วน) — เดิมกติกาคือ "เจอคำลบคำเดียวที่ไหนก็ตาม -> negative
       ทันที" ซึ่งทำให้รีวิวยาวที่ชมเป็นหลักแต่ติเล็กน้อยกลายเป็น negative ผิดๆ เกือบทุกครั้ง
       (ดู comment เหนือ _NEGATION_COMBO_NEG_BOUNDARY ใน keywords_data.py สำหรับหลักฐาน)
    """
    text_boundary = tokenize_boundary(str(message))

    if has_negation_positive(text_boundary):
        return "negative"

    # ใช้ token list จริง (ไม่ใช่แค่สตริงรวม) เพื่อนับ hit แบบไม่ซ้อนทับกัน — ป้องกัน
    # คำรากกับคำผสมของมันเอง (เช่น "หวาน" กับ "หวานไปนิด") ถูกนับเป็น 2 คะแนนจาก
    # จุดเดียวกัน ซึ่งจะทำให้ voting ระหว่างคำบวก/คำลบเพี้ยน (ดู comment เหนือ
    # count_nonoverlapping_matches ใน keywords_data.py)
    customer_tokens = tokenize_words(str(message))
    neg_hits = count_nonoverlapping_matches(customer_tokens, _NEGATIVE_TOKENS)
    pos_hits = count_nonoverlapping_matches(customer_tokens, _POSITIVE_TOKENS)

    if has_negation_of_negative(text_boundary) and neg_hits > 0:
        neg_hits -= 1

    if neg_hits == 0 and pos_hits == 0:
        if match_boundary_words(text_boundary, _QUESTION_BOUNDARY_MAP):
            return "neutral"
        return model_sentiment

    if neg_hits > 0 and pos_hits > 0:
        if neg_hits >= pos_hits * 2:
            return "negative"
        if pos_hits >= neg_hits * 2:
            return "positive"
        return "neutral"

    return "negative" if neg_hits > 0 else "positive"


def keyword_category_override(message: str, model_category: str) -> str:
    """
    หา category ที่ตรงที่สุดแบบ word-boundary จาก CATEGORY_KEYWORDS โดยเลือกหมวดที่มี
    keyword ยาว/เจาะจงที่สุดที่ match (longest-match-wins) แทนการใช้หมวดแรกตามลำดับ
    การประกาศในไฟล์ เพื่อไม่ให้คำทั่วไป (เช่น "เหม็น" ที่อยู่ในหมวดอาหารด้วย) บัง
    คำที่เจาะจงกว่าของอีกหมวด (เช่น "ห้องน้ำสกปรก")

    คืนค่าเป็น category "แบบละเอียด" (~35 หมวด ตาม CATEGORY_KEYWORDS) ใช้สำหรับเลือก
    reply template ใน decision_engine.choose_action() เท่านั้น — ไม่ใช่ label เดียวกับ
    ที่ category_model (ML) ถูกเทรนมา (ดู map_to_ml_category() ด้านล่างสำหรับค่าที่ใช้
    รายงาน/วัดความแม่นยำเทียบกับ ground truth)
    """
    text_boundary = tokenize_boundary(str(message))

    # เช็คกติกาแบบ AND ก่อน (สำหรับกรณีคำสำคัญถูกพิมพ์แยกกัน ไม่ติดกันเป็นวลีเดียว
    # เช่น "เอาเค้กวันเกิดไปเองได้ไหม" ดู CATEGORY_AND_KEYWORDS ใน keywords_data.py)
    and_match = keyword_category_and_match(text_boundary)
    if and_match:
        return and_match

    best_match = keyword_category_best_match(text_boundary)
    return best_match or model_category


# ---------------------------------------------------------------------------
# FIX (พบจากการ evaluate baseline อย่างเป็นระบบครั้งแรก 10 ก.ย.):
# category_model (ML) ถูกเทรนมาให้รู้จักแค่ 10 คลาสจาก Dataset_Restaurant_CRM
# (ML_CATEGORY_CLASSES ด้านล่าง) แต่ keyword_category_override() ข้างบนคืนค่าจาก
# CATEGORY_KEYWORDS ซึ่งมีถึง ~35 หมวด (ละเอียดกว่า เพื่อเลือก reply template ที่ตรง
# เป๊ะ เช่น "สอบถามที่จอดรถ", "สอบถาม WiFi") พอเอา category ละเอียดนี้ไปเทียบกับ
# ground truth ของ ML (ซึ่งไม่มีหมวดพวกนี้อยู่เลย) ผลคือ "ผิดเสมอ" ทุกครั้งที่ keyword
# ยิงหมวดที่ ML ไม่รู้จัก ทำให้ full-system accuracy ที่วัดได้จริงตกจาก ~66-100%
# (ML ล้วนๆ) เหลือแค่ ~32-34% ทั้งที่ในทางความหมาย keyword อาจตอบถูกกว่า ML ด้วยซ้ำ
#
# แก้โดยแยก 2 ระดับให้ชัดเจน:
#   - category_detail = ผลจาก keyword_category_override() (ละเอียด ~35 หมวด)
#     ใช้ส่งให้ choose_action() เลือก reply template เท่านั้น ไม่เปลี่ยนพฤติกรรมเดิม
#   - category (ที่ log/รายงาน/วัด accuracy) = map_to_ml_category(category_detail, ...)
#     บังคับให้อยู่ในกรอบ 10 คลาสเดียวกับที่ ML เทรนมาเสมอ โดย map หมวดละเอียดที่ไม่มี
#     คู่ตรงมาสู่คลาสที่ใกล้เคียงที่สุด (ดูคอมเมนต์รายบรรทัดในตาราง) ถ้าหมวดไหนไม่มีคลาส
#     ใกล้เคียงที่สมเหตุสมผลเลย จะ fallback กลับไปเชื่อค่าที่ ML เดาเอง (ไม่เดามั่ว)
#
# หมายเหตุสำคัญสำหรับวิทยานิพนธ์: การ map นี้เป็นการประมาณ (approximation) เพื่อให้
# วัดผลเทียบ ground truth ได้อย่างยุติธรรมในตอนนี้ ทางที่ถูกต้องกว่าในระยะยาวคือ
# re-label training dataset ด้วย taxonomy ละเอียด ~35 หมวดนี้ตรงๆ แล้วเทรนโมเดลใหม่
# ---------------------------------------------------------------------------
ML_CATEGORY_CLASSES = {
    "การจัดส่ง (Delivery)",
    "ชมสถานที่/บรรยากาศ",
    "ชมรสชาติอาหาร",
    "ชมการบริการ",
    "บริการไม่ดี",
    "ติชมอาหารและบริการ",
    "อาหารได้ไม่ตรง/ช้า",
    "สอบถามข้อมูลร้าน",
    "สอบถามโปรโมชั่น",
    "สอบถามเมนูอาหาร",
}

KEYWORD_CATEGORY_TO_ML_CATEGORY = {
    # ตรงตัว/ใกล้เคียงมาก
    "สอบถามโปรโมชั่น": "สอบถามโปรโมชั่น",
    "การจัดส่ง (Delivery)": "การจัดส่ง (Delivery)",
    "สอบถามเมนู": "สอบถามเมนูอาหาร",
    "ชมบรรยากาศร้าน": "ชมสถานที่/บรรยากาศ",
    "ชมความสะอาด": "ชมสถานที่/บรรยากาศ",
    "ชมพนักงาน": "ชมการบริการ",
    "ร้องเรียนการบริการ": "บริการไม่ดี",
    # คำถามข้อมูลทั่วไปเกี่ยวกับร้าน (ไม่มีคลาสเฉพาะใน ML) -> รวมเป็น "สอบถามข้อมูลร้าน"
    "จองโต๊ะ (Reservation)": "สอบถามข้อมูลร้าน",
    "เวลาเปิด-ปิดร้าน": "สอบถามข้อมูลร้าน",
    "ที่ตั้งร้าน (Location)": "สอบถามข้อมูลร้าน",
    "สอบถามที่จอดรถ": "สอบถามข้อมูลร้าน",
    "ช่องทางการชำระเงิน": "สอบถามข้อมูลร้าน",
    "สอบถามราคา": "สอบถามข้อมูลร้าน",
    "สอบถามแฟรนไชส์": "สอบถามข้อมูลร้าน",
    "สมัครงาน": "สอบถามข้อมูลร้าน",
    "จัดเลี้ยง/อีเวนต์": "สอบถามข้อมูลร้าน",
    "สอบถามช่องทางติดต่อ": "สอบถามข้อมูลร้าน",
    "สมาชิก/สะสมแต้ม": "สอบถามข้อมูลร้าน",
    "ซื้อกลับบ้าน (Takeaway)": "สอบถามข้อมูลร้าน",
    "สอบถามสาขา/ทำเล": "สอบถามข้อมูลร้าน",
    "สอบถามคิวรอโต๊ะ": "สอบถามข้อมูลร้าน",
    "สอบถาม WiFi": "สอบถามข้อมูลร้าน",
    "สอบถามพาสัตว์เลี้ยงเข้าร้าน": "สอบถามข้อมูลร้าน",
    "สอบถามสิ่งอำนวยความสะดวกสำหรับเด็ก": "สอบถามข้อมูลร้าน",
    "สอบถามห้องส่วนตัว/VIP": "สอบถามข้อมูลร้าน",
    "สอบถาม Corkage": "สอบถามข้อมูลร้าน",
    "สอบถามดนตรีสด/กิจกรรม": "สอบถามข้อมูลร้าน",
    # เกี่ยวกับเมนู/อาหาร -> "สอบถามเมนูอาหาร"
    "สอบถามข้อจำกัดด้านอาหาร": "สอบถามเมนูอาหาร",
    "สอบถามปรับระดับความเผ็ด/รส": "สอบถามเมนูอาหาร",
    "สอบถามบุฟเฟ่ต์": "สอบถามเมนูอาหาร",
    "สอบถามเครื่องดื่มแอลกอฮอล์": "สอบถามเมนูอาหาร",
    # ร้องเรียน/ปัญหาที่ไม่มีคลาสเฉพาะ -> "บริการไม่ดี"
    "ร้องเรียนความสะอาด": "บริการไม่ดี",
    "ร้องเรียนอุณหภูมิ/แอร์": "บริการไม่ดี",
    "สอบถาม/ร้องเรียนห้องน้ำ": "บริการไม่ดี",
    "ร้องเรียนบรรยากาศ": "บริการไม่ดี",
    "ปัญหาบัตรสมาชิก/แต้ม": "บริการไม่ดี",
    "ร้องเรียนบิล/ยอดเงินผิด": "บริการไม่ดี",
    "ปัญหาแอพ/ระบบสั่งอาหาร": "บริการไม่ดี",
    # ผลกระทบด้านอาหาร/บริการโดยรวม -> "ติชมอาหารและบริการ"
    "ร้องเรียนคุณภาพอาหาร": "ติชมอาหารและบริการ",
    "รีวิว/ให้คะแนน": "ติชมอาหารและบริการ",
    "ข้อเสนอแนะทั่วไป": "ติชมอาหารและบริการ",
    # เกี่ยวกับออเดอร์ผิด/ล่าช้า -> "อาหารได้ไม่ตรง/ช้า"
    "ยกเลิก/คืนเงิน": "อาหารได้ไม่ตรง/ช้า",
    # เกี่ยวกับพื้นที่/เงื่อนไขการจัดส่ง -> "การจัดส่ง (Delivery)"
    "สอบถามพื้นที่จัดส่ง": "การจัดส่ง (Delivery)",
    "สอบถามยอดสั่งขั้นต่ำ": "การจัดส่ง (Delivery)",
}


def map_to_ml_category(category_detail: str, model_category: str) -> str:
    """
    บังคับ category ที่จะ log/รายงาน/วัด accuracy ให้อยู่ในกรอบ 10 คลาสเดียวกับที่
    category_model (ML) เทรนมาเสมอ (ดูคอมเมนต์ด้านบน) — ใช้ category_detail ที่ได้จาก
    keyword_category_override() เป็นหลัก ถ้า map ไม่ได้ (ไม่ควรเกิดขึ้นถ้าตารางครบ)
    ให้ fallback กลับไปเชื่อค่าที่ ML เดาเอง (model_category) แทนการเดามั่ว
    """
    if category_detail in ML_CATEGORY_CLASSES:
        return category_detail
    return KEYWORD_CATEGORY_TO_ML_CATEGORY.get(category_detail, model_category)


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
    category_detail = keyword_category_override(text, category_ml)
    category = map_to_ml_category(category_detail, category_ml)

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

    # ---------------------------------------------------------------------
    # FIX (ขั้นที่ 3 ตามแผนพัฒนา confidence — ตกลงกับผู้ใช้ไว้ว่าเช็คเฉพาะ sentiment
    # เพราะ category แม่นยำ/calibrate ดีอยู่แล้วที่ ~100%): เคสที่ไม่มี keyword ไหน
    # match เลย (sentiment_source ยังเป็น "model" คือเชื่อ ML ล้วนๆ) แล้ว ML ยังไม่ค่อย
    # มั่นใจ (sentiment_confidence < 60% ตามที่ตกลงกันไว้) ให้ถาม OpenAI เป็นความเห็น
    # ที่สอง แทนที่จะปล่อยให้ใช้คำตอบของ ML ที่ตัวมันเองก็ไม่มั่นใจไปตรงๆ — เรียกเฉพาะ
    # เคสนี้เท่านั้น (ไม่ใช่ทุกข้อความ) เพื่อคุมค่าใช้จ่าย API
    #
    # **ไม่กระทบ Make.com HTTP module เดิมเลย** — เกิดขึ้นข้างในนี้ทั้งหมด ก่อนจะคืนค่า
    # result dict ที่มี field ชุดเดิมทุกอย่าง (เพิ่มแค่ sentiment_source บอกที่มา)
    # ถ้าเรียก OpenAI ไม่สำเร็จ (ยังไม่ได้ตั้ง OPENAI_API_KEY, network error, ฯลฯ)
    # verify_sentiment() คืน (None, None) แล้วโค้ดจะข้ามไปใช้ผล ML เดิมทันที ไม่พัง
    # ---------------------------------------------------------------------
    if sentiment_source == "model" and sentiment_confidence < SENTIMENT_VERIFY_THRESHOLD:
        verified_sentiment, verified_confidence = verify_sentiment(text)
        if verified_sentiment is not None:
            sentiment = verified_sentiment
            sentiment_confidence = verified_confidence
            sentiment_source = "openai"

    behavior = get_user_behavior(user_id)
    segment = str(behavior.get("segment", "Regular"))
    # ส่ง category_detail (ละเอียด ~35 หมวด) เข้า choose_action() เพื่อเลือก reply
    # template ที่ตรงเป๊ะเหมือนเดิม (เช่น แยก "สอบถามที่จอดรถ" ออกจาก "สอบถาม WiFi" ได้)
    # ส่วน category (ตัวแปรบรรทัดบน) ที่ map เข้ากรอบ 10 คลาสแล้ว มีไว้ log/รายงานเท่านั้น
    action = choose_action(sentiment, category_detail, segment, text)
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
        "category_detail": category_detail,
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
