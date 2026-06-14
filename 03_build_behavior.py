# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
from app.decision_engine import choose_segment

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"

print("STEP 6: Build User Behavior Features")
crm = pd.read_csv(OUT / "crm_cleaned.csv", encoding="utf-8-sig")
crm["Date/Time"] = pd.to_datetime(crm["Date/Time"], errors="coerce")
max_date = crm["Date/Time"].max()

# กำหนด negative/positive จาก AI Sentiment ภาษาไทย
crm["is_negative"] = crm["AI Sentiment"].astype(str).str.contains("ลบ|negative", case=False, regex=True).astype(int)
crm["is_positive"] = crm["AI Sentiment"].astype(str).str.contains("บวก|positive", case=False, regex=True).astype(int)
crm["is_complaint"] = (
    crm["Category"].astype(str).str.contains("ช้า|ผิด|จัดส่ง|ร้องเรียน|เคลม|Delivery", case=False, regex=True)
    | crm["Automation Status"].astype(str).str.contains("ผู้จัดการ|เคลม|ตรวจสอบ", case=False, regex=True)
).astype(int)
crm["hour"] = crm["Date/Time"].dt.hour

# favorite hour = hour mode ต่อ customer
fav_hour = crm.groupby("Customer Name")["hour"].agg(lambda x: int(x.mode().iloc[0]) if len(x.mode()) else None).reset_index(name="favorite_hour")

behavior = crm.groupby("Customer Name").agg(
    total_messages=("Comment Text", "count"),
    positive_count=("is_positive", "sum"),
    negative_count=("is_negative", "sum"),
    complaint_count=("is_complaint", "sum"),
    first_visit=("Date/Time", "min"),
    last_visit=("Date/Time", "max"),
).reset_index().merge(fav_hour, on="Customer Name", how="left")

behavior["inactive_days"] = (max_date - behavior["last_visit"]).dt.days.fillna(0).astype(int)
behavior["segment"] = behavior.apply(lambda r: choose_segment(r["total_messages"], r["negative_count"], r["complaint_count"], r["inactive_days"]), axis=1)
behavior = behavior.rename(columns={"Customer Name": "customer_id"})
behavior.to_csv(OUT / "customer_behavior_features.csv", index=False, encoding="utf-8-sig")
print("Saved:", OUT / "customer_behavior_features.csv")
print(behavior.head(10).to_string())
