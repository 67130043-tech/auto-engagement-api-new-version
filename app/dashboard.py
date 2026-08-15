# -*- coding: utf-8 -*-
"""
ตรรกะสรุปผล + สร้าง dashboard HTML แบบใช้ร่วมกันได้ทั้ง 2 แบบ:
1) รันเป็นสคริปต์ (07_confidence_report.py + 08_generate_dashboard.py) - เหมาะกับ Colab
2) เรียกผ่าน endpoint บนเว็บ (/dashboard) - เหมาะกับ Render หรือ deploy แบบอื่นที่ไม่มี shell ให้รัน python เอง
"""
import json
import pandas as pd


def compute_summary(df: pd.DataFrame, threshold: float = 70.0):
    if "source" in df.columns:
        real = df[df["source"].astype(str).str.lower() == "make"].copy()
        if len(real) == 0:
            real = df.copy()
    else:
        real = df.copy()

    required_cols = {"reply_confidence", "sentiment_confidence", "category_confidence"}
    missing = required_cols - set(real.columns)
    if len(real) == 0 or missing:
        return None

    real["reply_confidence"] = pd.to_numeric(real["reply_confidence"], errors="coerce")
    real = real.dropna(subset=["reply_confidence"])
    if len(real) == 0:
        return None

    total = len(real)
    avg_sentiment_conf = round(real["sentiment_confidence"].astype(float).mean(), 2)
    avg_category_conf = round(real["category_confidence"].astype(float).mean(), 2)
    avg_reply_conf = round(real["reply_confidence"].astype(float).mean(), 2)

    correct = int((real["reply_confidence"] >= threshold).sum())
    incorrect = total - correct
    accuracy_pct = round(correct / total * 100, 2)

    by_sentiment = (
        real.groupby("sentiment")["reply_confidence"]
        .apply(lambda s: round((s >= threshold).mean() * 100, 2))
        .reset_index()
    )
    by_sentiment.columns = ["label", "accuracy_percent"]

    by_category = (
        real.groupby("category")["reply_confidence"]
        .apply(lambda s: round((s >= threshold).mean() * 100, 2))
        .reset_index()
    )
    by_category.columns = ["label", "accuracy_percent"]

    return {
        "total": total,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy_pct": accuracy_pct,
        "threshold": threshold,
        "avg_sentiment_conf": avg_sentiment_conf,
        "avg_category_conf": avg_category_conf,
        "avg_reply_conf": avg_reply_conf,
        "by_sentiment": by_sentiment,
        "by_category": by_category,
    }


def render_dashboard_html(summary: dict) -> str:
    sentiment_labels = summary["by_sentiment"]["label"].tolist()
    sentiment_values = summary["by_sentiment"]["accuracy_percent"].tolist()
    category_labels = summary["by_category"]["label"].tolist()
    category_values = summary["by_category"]["accuracy_percent"].tolist()

    return f"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<title>Auto Engagement System - Live Dashboard</title>
<meta http-equiv="refresh" content="120">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #f4f6f8; margin: 0; padding: 24px; color: #1f2937; }}
  h1 {{ font-size: 22px; margin-bottom: 4px; }}
  .subtitle {{ color: #6b7280; margin-bottom: 24px; font-size: 14px; }}
  .cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 28px; }}
  .card {{ background: white; border-radius: 12px; padding: 20px 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); flex: 1; min-width: 160px; }}
  .card .label {{ font-size: 13px; color: #6b7280; margin-bottom: 6px; }}
  .card .value {{ font-size: 30px; font-weight: 700; }}
  .accuracy .value {{ color: #16a34a; }}
  .correct .value {{ color: #2563eb; }}
  .incorrect .value {{ color: #dc2626; }}
  .charts {{ display: flex; gap: 20px; flex-wrap: wrap; }}
  .chart-box {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); flex: 1; min-width: 340px; }}
  .chart-box h2 {{ font-size: 15px; margin-top: 0; }}
  canvas {{ max-height: 320px; }}
  .note {{ font-size: 12px; color: #9ca3af; margin-top: 24px; }}
</style>
</head>
<body>
  <h1>Auto Engagement System — Live Dashboard</h1>
  <div class="subtitle">คำนวณจากคอมเมนต์จริงที่เข้ามาผ่าน Make + Facebook หน้านี้รีเฟรชอัตโนมัติทุก 2 นาที</div>

  <div class="cards">
    <div class="card"><div class="label">จำนวนคอมเมนต์จริงที่ตอบแล้ว</div><div class="value">{summary["total"]}</div></div>
    <div class="card correct"><div class="label">ตอบด้วยความมั่นใจสูง (&ge;{summary["threshold"]}%)</div><div class="value">{summary["correct"]}</div></div>
    <div class="card incorrect"><div class="label">ความมั่นใจต่ำ</div><div class="value">{summary["incorrect"]}</div></div>
    <div class="card accuracy"><div class="label">% ตอบด้วยความมั่นใจสูง</div><div class="value">{summary["accuracy_pct"]}%</div></div>
  </div>

  <div class="cards">
    <div class="card"><div class="label">ค่าเฉลี่ยความมั่นใจรวม</div><div class="value">{summary["avg_reply_conf"]}%</div></div>
  </div>

  <div class="charts">
    <div class="chart-box"><h2>% ความมั่นใจสูง แยกตาม Sentiment</h2><canvas id="sentimentChart"></canvas></div>
    <div class="chart-box"><h2>% ความมั่นใจสูง แยกตาม Category</h2><canvas id="categoryChart"></canvas></div>
  </div>

  <div class="note">
    "ความมั่นใจสูง" คำนวณจากคะแนน confidence ของโมเดล ณ ขณะตอบคอมเมนต์จริงแต่ละข้อความ
    ไม่ใช่การเทียบกับคำตอบที่มนุษย์ยืนยันไว้ล่วงหน้า
  </div>

<script>
new Chart(document.getElementById('sentimentChart'), {{
  type: 'bar',
  data: {{ labels: {json.dumps(sentiment_labels, ensure_ascii=False)}, datasets: [{{ label: 'High-Confidence Rate (%)', data: {json.dumps(sentiment_values)}, backgroundColor: '#2563eb' }}] }},
  options: {{ scales: {{ y: {{ beginAtZero: true, max: 100 }} }}, plugins: {{ legend: {{ display: false }} }} }}
}});
new Chart(document.getElementById('categoryChart'), {{
  type: 'bar',
  data: {{ labels: {json.dumps(category_labels, ensure_ascii=False)}, datasets: [{{ label: 'High-Confidence Rate (%)', data: {json.dumps(category_values)}, backgroundColor: '#16a34a' }}] }},
  options: {{ indexAxis: 'y', scales: {{ x: {{ beginAtZero: true, max: 100 }} }}, plugins: {{ legend: {{ display: false }} }} }}
}});
</script>
</body>
</html>"""


def render_no_data_html(reason: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="th"><head><meta charset="UTF-8"><title>Dashboard</title></head>
<body style="font-family: sans-serif; padding: 40px; color: #444;">
  <h2>ยังไม่มีข้อมูลให้แสดงผล</h2>
  <p>{reason}</p>
</body></html>"""
