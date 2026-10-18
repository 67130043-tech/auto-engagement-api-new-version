# Wongnai corpus — review dataset (ฉบับเต็ม)

ใช้เป็นแหล่งประโยค **รีวิวร้านอาหารจริง** เพิ่มเติม สำหรับเติมหมวดที่ `data/wongnai.csv`
(19,458 รีวิว) มีข้อมูลไม่พอ ใน `09_build_category_dataset_from_wongnai.py`

- ที่มา: https://github.com/wongnai/wongnai-corpus (`review/review_dataset.zip`)
- commit ต้นทาง: `113ad2962374fa3274eeed7e4aa74540472e3bd5` (26 ส.ค. 2019)
- เนื้อหา: `w_review_train.csv` 40,000 รีวิว และ `test_file.csv` 6,203 รีวิว
  (คั่นด้วย `;`) — ใช้เฉพาะข้อความรีวิว ไม่ได้ใช้คะแนนดาว
- สัญญาอนุญาต: GNU LGPL v3 (ไฟล์ `LICENSE` ในโฟลเดอร์นี้) เก็บไฟล์ต้นฉบับไว้โดยไม่แก้ไข
- อ้างอิง: Wongnai Corpus, Wongnai Media Co., Ltd., 2018–2019

สคริปต์จะตัดประโยคที่ซ้ำกับ `data/wongnai.csv` ออกก่อนใช้เสมอ
