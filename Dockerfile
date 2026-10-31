# ใช้ได้ทั้ง HuggingFace Spaces (Docker) และ Render (Docker runtime)
# โมเดล WangchanBERTa ไม่ได้อยู่ใน image — ดาวน์โหลดจาก HuggingFace Hub ตอนเริ่มทำงาน
# ตามค่า WANGCHAN_SENTIMENT_REPO / WANGCHAN_CATEGORY_REPO (ดู README)
# ต้องการ RAM ประมาณ 1.5 GB ขึ้นไปเมื่อใช้ WangchanBERTa ทั้งสองโมเดล
FROM python:3.12-slim

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
RUN mkdir -p outputs && chown -R user outputs

USER user
ENV HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
