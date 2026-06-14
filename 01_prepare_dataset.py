# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
from app.preprocess import clean_text, star_to_sentiment

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

crm_path = DATA / "Dataset_Restaurant_CRM_4000_Rows.xlsx"
wongnai_path = DATA / "wongnai.csv"

print("STEP 1: Read datasets")
crm = pd.read_excel(crm_path)
wongnai = pd.read_csv(wongnai_path, encoding="cp874")

print("CRM shape:", crm.shape)
print("Wongnai shape:", wongnai.shape)
print("CRM columns:", crm.columns.tolist())
print("Wongnai columns:", wongnai.columns.tolist())

print("\nSTEP 2: Clean text")
crm["clean_text"] = crm["Comment Text"].apply(clean_text)
wongnai["clean_text"] = wongnai["review_body"].apply(clean_text)
wongnai["sentiment"] = wongnai["stars"].apply(star_to_sentiment)

crm = crm[crm["clean_text"].str.len() > 0].copy()
wongnai = wongnai[(wongnai["clean_text"].str.len() > 0) & wongnai["sentiment"].notna()].copy()

crm.to_csv(OUT / "crm_cleaned.csv", index=False, encoding="utf-8-sig")
wongnai.to_csv(OUT / "wongnai_cleaned.csv", index=False, encoding="utf-8-sig")

print("Saved:", OUT / "crm_cleaned.csv")
print("Saved:", OUT / "wongnai_cleaned.csv")
print("\nCRM sample")
print(crm[["Customer Name", "Comment Text", "clean_text", "Category", "AI Sentiment"]].head(3).to_string())
print("\nWongnai sample")
print(wongnai[["review_body", "stars", "sentiment", "clean_text"]].head(3).to_string())
