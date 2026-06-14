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

@app.post("/predict")
def predict(req: PredictRequest):
    return predict_message(
        user_id=req.user_id,
        message=req.message,
        channel=req.channel,
        display_name=req.display_name,
        source=req.source,
    )

# endpoint นี้ตั้งชื่อให้ Make จำง่าย
@app.post("/make/predict")
def make_predict(req: PredictRequest):
    return predict(req)
