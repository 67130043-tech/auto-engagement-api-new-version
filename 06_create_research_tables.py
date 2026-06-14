# -*- coding: utf-8 -*-
from pathlib import Path
import json
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"

print("STEP 8: Create research summary tables")
with open(OUT / "model_metrics.json", "r", encoding="utf-8") as f:
    metrics = json.load(f)

model_summary = pd.DataFrame([
    {"model": "Sentiment Classification", "dataset": "wongnai.csv", "accuracy": metrics["sentiment_accuracy"]},
    {"model": "Category Classification", "dataset": "Dataset_Restaurant_CRM_4000_Rows.xlsx", "accuracy": metrics["category_accuracy"]},
])
model_summary.to_csv(OUT / "research_model_summary.csv", index=False, encoding="utf-8-sig")

# template สำหรับกรอกผลหลังทดลองจริงผ่าน Facebook/Make
system_metrics = pd.DataFrame([
    {"metric": "Response Rate", "before": "", "after": "", "formula": "customers_replied / messages_sent * 100"},
    {"metric": "Engagement Rate", "before": "", "after": "", "formula": "interactions / total_customers * 100"},
    {"metric": "Retention Rate", "before": "", "after": "", "formula": "returning_customers / total_customers * 100"},
    {"metric": "Average Reply Time", "before": "", "after": "", "formula": "average(response_timestamp - customer_message_timestamp)"},
])
system_metrics.to_csv(OUT / "research_system_metrics_template.csv", index=False, encoding="utf-8-sig")
print("Saved research tables in outputs/")
print(model_summary.to_string(index=False))
