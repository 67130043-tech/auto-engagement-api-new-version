# -*- coding: utf-8 -*-
from app.engine import predict_message

print("STEP 7: Test prediction without API server")
examples = [
    ("New_Gen_58", "อาหารมาช้ามาก รอเกือบชั่วโมง", "facebook"),
    ("Peanut_B_33", "อาหารอร่อยมากค่ะ บริการดี", "facebook"),
    ("New_User_001", "วันนี้มีโปรอะไรบ้าง", "facebook"),
    ("New_User_002", "อยากจองโต๊ะวันนี้ 4 คน", "facebook"),
    ("New_User_003", "ส่งอาหารผิดเมนูค่ะ", "facebook"),
]
for user_id, msg, channel in examples:
    result = predict_message(user_id=user_id, message=msg, channel=channel, source="local_test")
    print("\nINPUT:", msg)
    print("sentiment:", result["sentiment"])
    print("category:", result["category"])
    print("segment:", result["segment"])
    print("action:", result["action"])
    print("reply_message:", result["reply_message"])
