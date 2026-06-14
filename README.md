# User Behavior-based Auto Engagement API สำหรับ Google Colab + Make + Facebook

โปรเจกต์นี้ทำตาม Flow ตั้งแต่ต้นจนจบ:

Dataset → Preprocessing → Train Sentiment Model → Train Category Model → User Behavior Engine → Decision Engine → FastAPI → Make → Facebook Messenger → Logging → Research Tables

## ไฟล์ข้อมูล

- `data/wongnai.csv` ใช้ฝึก Sentiment Model จาก `review_body` และ `stars`
- `data/Dataset_Restaurant_CRM_4000_Rows.xlsx` ใช้ฝึก Category Model และสร้าง User Behavior

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

