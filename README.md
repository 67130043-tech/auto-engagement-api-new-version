# User Behavior-based Auto Engagement API สำหรับ Google Colab + Make + Facebook

โปรเจกต์นี้ทำตาม Flow ตั้งแต่ต้นจนจบ:

Dataset → Preprocessing → Train Sentiment Model → Train Category Model → User Behavior Engine → Decision Engine → FastAPI → Make → Facebook Messenger → Logging → Research Tables

## ไฟล์ข้อมูล

- `data/sentiment_dataset_v2.csv` ใช้ฝึก Sentiment Model — สร้างด้วย `10_build_sentiment_dataset.py`
  จาก Wisesight Sentiment (ข้อความโซเชียลที่คนติดป้าย, CC0) + รีวิว Wongnai (ป้ายจากดาว)
- `data/wongnai.csv` รีวิว Wongnai 19,458 รีวิว (`review_body`, `stars`)
- `data/category_dataset_v2.csv` ใช้ฝึก Category Model — สร้างด้วย `09_build_category_dataset_from_wongnai.py`
  จากรีวิวจริงล้วน (ไม่มีประโยคสังเคราะห์) ติดป้ายด้วยกฎ keyword/pattern ของระบบ (คอลัมน์ `source` บอกที่มา)
- `data/external/wongnai-corpus/` Wongnai corpus ฉบับเต็ม (LGPL-3) ใช้เติมหมวดที่ `wongnai.csv` มีไม่พอ
- `data/Dataset_Restaurant_CRM_4000_Rows.xlsx` ใช้สร้าง User Behavior (ไม่ใช้ฝึก Category Model แล้ว
  เพราะมีข้อความไม่ซ้ำกันแค่ 85 ประโยค)

## โมเดลหลัก: WangchanBERTa

ระบบใช้ [WangchanBERTa](https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased) ที่ fine-tune แล้ว
เป็นตัวตัดสิน sentiment และ category (keyword เหลือแค่กฎความปลอดภัย และเลือกคำตอบละเอียดในหมวดที่โมเดลทาย)
ถ้ายังไม่มีโมเดล API จะใช้ TF-IDF+SVM (`models/*.joblib`) แบบเดิมโดยอัตโนมัติ — เปิด `/` เพื่อดูว่ากำลังใช้โมเดลไหน

1. เปิด `11_finetune_wangchanberta.ipynb` บน Google Colab (Runtime → T4 GPU) แล้วรันทุก cell
2. ตั้งค่าบน Render: `WANGCHAN_SENTIMENT_REPO`, `WANGCHAN_CATEGORY_REPO` (repo บน HuggingFace Hub ที่ notebook อัปโหลดให้)
   และ `HF_TOKEN` ถ้า repo เป็น private — หรือวางโฟลเดอร์ `models/wangchanberta_sentiment/`, `models/wangchanberta_category/` เอง
3. เทียบผลทุกระบบบน test set เดียวกัน: `python 12_compare_models.py` (เพิ่ม `--llm` เพื่อรวม LLM zero-shot)

| ตัวแปร | ความหมาย |
|---|---|
| `WANGCHAN_<TASK>_REPO` / `WANGCHAN_<TASK>_DIR` | ที่อยู่โมเดล (`<TASK>` = `SENTIMENT` หรือ `CATEGORY`) |
| `DECISION_MODE` | `model` = โมเดลตัดสิน, `hybrid` = keyword ทับผลโมเดลแบบเดิม (ค่าเริ่มต้น: `model` เมื่อใช้ WangchanBERTa) |
| `DISABLE_TRANSFORMER=true` | บังคับใช้ TF-IDF+SVM |

**ชุดทดสอบมาตรฐาน (Gold Standard):** ป้าย category ใน `category_dataset_v2.csv` มาจากกฎ keyword
ตัวเลขบนชุดนั้นจึงวัดความสอดคล้องกับกฎ ไม่ใช่ความถูกต้องจริง ให้คัดลอก `data/gold_standard_template.csv`
เป็น `data/gold_standard.csv` แล้วใส่คอมเมนต์จริงจากเพจที่คนติดป้ายเอง `12_compare_models.py` จะวัดผลบนชุดนี้ด้วย

## ลำดับรันใน Google Colab

### 1) Upload ZIP เข้า Colab แล้วแตกไฟล์
```python
!unzip colab_auto_engagement_make_project_ready.zip
%cd colab_auto_engagement_make_project
```

### 2) ติดตั้ง Library
```python
!pip install -r requirements.txt
```

### 3) เตรียม Dataset
```python
!python 01_prepare_dataset.py
!python 09_build_category_dataset_from_wongnai.py   # data/category_dataset_v2.csv
!python 10_build_sentiment_dataset.py              # data/sentiment_dataset_v2.csv
```
ผลลัพธ์:
- `outputs/crm_cleaned.csv`
- `outputs/wongnai_cleaned.csv`

### 4) Train Model
```python
!python 02_train_models.py
```
ผลลัพธ์:
- `models/sentiment_model.joblib`
- `models/category_model.joblib`
- `outputs/model_metrics.json`

### 5) สร้าง User Behavior Features
```python
!python 03_build_behavior.py
```
ผลลัพธ์:
- `outputs/customer_behavior_features.csv`

### 6) ทดสอบ Prediction ไม่ต้องเปิด API
```python
!python 04_test_prediction.py
```
จะเห็น sentiment, category, segment, action และ reply_message

### 7) เปิด FastAPI ใน Colab
```python
!uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 8) ทดสอบ API จาก Cell ใหม่
```python
import requests
payload = {
    "user_id": "facebook_user_001",
    "message": "อาหารช้ามาก ไม่ประทับใจเลย",
    "channel": "facebook",
    "display_name": "Customer Test",
    "source": "make"
}
res = requests.post("http://127.0.0.1:8000/make/predict", json=payload)
res.json()
```

## ใช้กับ Make

ใน Make ให้สร้าง Scenario:

Facebook Messenger: Watch Messages → HTTP: Make a request → Facebook Messenger: Send Message → Google Sheets: Add Row

### HTTP Module

Method: `POST`

URL:
```
https://YOUR_PUBLIC_COLAB_OR_DEPLOY_URL/make/predict
```

Headers:
```
Content-Type: application/json
```

Body:
```json
{
  "user_id": "{{Sender ID}}",
  "message": "{{Message Text}}",
  "channel": "facebook",
  "display_name": "{{Sender Name}}",
  "source": "make"
}
```

Make ต้องนำค่า `reply_message` จาก Response ไปใส่ใน Facebook Messenger: Send Message

## สร้าง Public URL ใน Colab

วิธีง่ายคือใช้ ngrok:

```python
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_TOKEN")
public_url = ngrok.connect(8000)
print(public_url)
```

แล้วใช้ URL ที่ได้ เช่น:
```
https://xxxx.ngrok-free.app/make/predict
```

## ผลลัพธ์สำหรับบทความ

รัน:
```python
!python 06_create_research_tables.py
```

ไฟล์ที่ใช้เขียนบทความ:
- `outputs/model_metrics.json`
- `outputs/research_model_summary.csv`
- `outputs/research_system_metrics_template.csv`
- `outputs/customer_behavior_features.csv`
- `outputs/prediction_log.csv`

