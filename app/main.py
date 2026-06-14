# -*- coding: utf-8 -*-
from fastapi import FastAPI
from pydantic import BaseModel
from app.engine import predict_message

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
