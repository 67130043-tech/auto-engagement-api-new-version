# -*- coding: utf-8 -*-
import requests

API_URL = "http://127.0.0.1:8000/make/predict"
payload = {
    "user_id": "facebook_user_001",
    "message": "อาหารช้ามาก ไม่ประทับใจเลย",
    "channel": "facebook",
    "display_name": "Customer Test",
    "source": "make"
}

res = requests.post(API_URL, json=payload, timeout=30)
print(res.status_code)
print(res.json())
