# -*- coding: utf-8 -*-
import os
import io
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from app.engine import predict_message
from app.dashboard import compute_summary, render_dashboard_html, render_no_data_html
import requests

app = FastAPI(title="User Behavior-based Auto Engagement AI Processing API")

class PredictRequest(BaseModel):
    user_id: str
    message: str
    channel: str = "facebook"
    display_name: str = ""
    source: str = "api"

@app.get("/")
def root():
    return {"status": "ok", "service": "Auto Engagement AI Processing API", "endpoint": "/predict"}

#@app.post("/predict")
#def predict(req: PredictRequest):
#    return predict_message(
#        user_id=req.user_id,
#        message=req.message,
#        channel=req.channel,
#        display_name=req.display_name,
#        source=req.source,
#    )

@app.post("/make/test")
def make_test(request: PredictRequest):
    return {
        "user_id": request.user_id,
        "message": request.message,
        "sentiment": "Negative",
        "category": "Delivery",
        "segment": "Regular",
        "action": "Apology",
        "reply_message": "ขออภัยในความไม่สะดวกค่ะ ทางร้านจะรีบตรวจสอบให้ทันทีค่ะ"
    }

# endpoint นี้ตั้งชื่อให้ Make จำง่าย
#@app.post("/make/predict")
#def make_predict(req: PredictRequest):
#    return predict(req)

@app.post("/make/predict")
def make_predict(req: PredictRequest):
    return predict_message(
        user_id=req.user_id,
        message=req.message,
        channel=req.channel,
        display_name=req.display_name,
        source=req.source
    )

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(threshold: float = 70.0):
    """
    เปิดหน้านี้ในเบราว์เซอร์ได้เลย เช่น https://your-app.onrender.com/dashboard
    อ่านข้อมูลจาก Google Sheet เดียวกับที่ Make เขียนอยู่ (ต้องตั้งค่า env var GOOGLE_SHEET_CSV_URL
    บน Render ก่อน ชี้ไปที่ลิงก์ export CSV ของชีตนั้น) แล้วคำนวณ + แสดงผลสดทุกครั้งที่เปิดหน้านี้
    ปรับเกณฑ์ความมั่นใจสูงได้ผ่าน query string เช่น /dashboard?threshold=80
    """
    sheet_url = os.environ.get("GOOGLE_SHEET_CSV_URL", "")
    if not sheet_url:
        return render_no_data_html(
            "ยังไม่ได้ตั้งค่า environment variable GOOGLE_SHEET_CSV_URL บน Render "
            "ให้ชี้ไปที่ลิงก์ export CSV ของ Google Sheet ที่ Make เขียนอยู่"
        )
    try:
        resp = requests.get(sheet_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), on_bad_lines="skip", engine="python")
    except Exception as e:
        return render_no_data_html(f"อ่านข้อมูลจาก Google Sheet ไม่สำเร็จ: {e}")
        
    summary = compute_summary(df, threshold=threshold)
    if summary is None:
        return render_no_data_html(
            "ยังไม่มีคอลัมน์ confidence ครบ หรือยังไม่มีคอมเมนต์จริงเข้ามาเลย "
            "ตรวจสอบว่าเพิ่มคอลัมน์ sentiment_confidence / category_confidence / reply_confidence "
            "ในชีตแล้ว และมีอย่างน้อย 1 แถวข้อมูล"
        )
    return render_dashboard_html(summary)


@app.get("/debug-sheet")
def debug_sheet():
    sheet_url = os.environ.get("GOOGLE_SHEET_CSV_URL", "")
    if not sheet_url:
        return {"error": "GOOGLE_SHEET_CSV_URL not set"}
    try:
        resp = requests.get(sheet_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        info = {
            "status_code": resp.status_code,
            "content_type": resp.headers.get("content-type"),
            "first_200_chars": resp.text[:200],
        }
        df = pd.read_csv(io.StringIO(resp.text), on_bad_lines="skip", engine="python")
        info["shape"] = df.shape
        info["columns"] = list(df.columns)
        if "reply_confidence" in df.columns:
            info["reply_confidence_sample"] = df["reply_confidence"].head(10).tolist()
        return info
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# DEBUG (ชั่วคราว): เอาไว้เช็คว่าไฟล์ keywords_data.py / engine.py ที่รันอยู่จริง
# บนเซิร์ฟเวอร์นี้ ตรงกับเวอร์ชันที่แก้ไปหรือยัง ลบทิ้งได้เมื่อเช็คเสร็จแล้ว
# ---------------------------------------------------------------------------
@app.get("/debug-keywords")
def debug_keywords():
    from app.keywords_data import NEGATIVE_WORDS, CATEGORY_KEYWORDS
    from app.engine import keyword_sentiment_override, keyword_category_override
    from app.preprocess import clean_text

    test_msg = "ผัดไทยเผ็ดไปหน่อยนะครับ"
    text = clean_text(test_msg)

    return {
        "negative_words_count": len(NEGATIVE_WORDS),
        "category_keywords_count": len(CATEGORY_KEYWORDS),
        "has_target_keyword": "เผ็ดไปหน่อย" in NEGATIVE_WORDS,
        "clean_text_output": text,
        "sentiment_override_result": keyword_sentiment_override(text, "neutral"),
        "category_override_result": keyword_category_override(text, "ML_FALLBACK_LABEL"),
    }
