# -*- coding: utf-8 -*-
"""
ตรรกะสรุปผล + สร้าง dashboard HTML แบบใช้ร่วมกันได้ทั้ง 2 แบบ:
1) รันเป็นสคริปต์ (07_confidence_report.py + 08_generate_dashboard.py) - เหมาะกับ Colab
2) เรียกผ่าน endpoint บนเว็บ (/dashboard) - เหมาะกับ Render หรือ deploy แบบอื่นที่ไม่มี shell ให้รัน python เอง
"""
import json
from datetime import datetime
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

    # ---------------------------------------------------------------------
    # FIX (พบจากหน้าตา dashboard จริงที่ดู "ตลกๆ": แทบทุกหมวดขึ้น 100% เป๊ะยกเว้น
    # หมวดเดียว): สาเหตุจริงมี 2 ชั้น
    #   1) ตัวชี้วัดเดิมคือ "% ของคอมเมนต์ในหมวดนั้นที่ confidence >= threshold" —
    #      เป็นค่า binary ต่อคอมเมนต์ (ผ่าน/ไม่ผ่าน) พอมาเฉลี่ยกับกลุ่มที่มีแค่
    #      1 คอมเมนต์ ผลจึงกระโดดสุดโต่งได้แค่ 0% หรือ 100% เท่านั้น ไม่มีค่ากลาง
    #      เลย (ผู้ใช้ท้วงตรงจุดนี้พอดี: "มันไม่ควร 100% มันควรเป็นค่าเฉลี่ยของ
    #      แต่ละ Category") แก้โดยเปลี่ยนไปโชว์ "ค่าเฉลี่ย confidence จริง" ของ
    #      แต่ละกลุ่มแทน (ต่อเนื่อง ไม่ใช่ binary) เช่น ถ้าหมวดนั้นมีคอมเมนต์เดียว
    #      มั่นใจ 88% กราฟจะขึ้น 88% ตรงๆ ไม่ใช่ปัดเป็น 100%
    #   2) ต่อให้เฉลี่ยแล้ว กลุ่มที่มีตัวอย่างแค่ 1-2 ข้อความก็ยังไม่นิ่งพอจะเชื่อ
    #      ถือได้ 100% จึงยังเก็บ count ไว้คู่กัน ให้ dashboard แสดงจำนวนตัวอย่าง
    #      กำกับทุกแท่ง และลดน้ำหนักภาพ (สีจาง) ให้หมวดที่ตัวอย่างยังน้อยเกินไป
    #      (ดูการใช้งานคู่กันใน render_dashboard_html())
    # ---------------------------------------------------------------------
    def _group_stats(s):
        s = s.astype(float)
        return pd.Series({
            "avg_confidence": round(s.mean(), 2),
            "count": int(s.shape[0]),
        })

    by_sentiment = real.groupby("sentiment")["reply_confidence"].apply(_group_stats).unstack()
    by_sentiment = by_sentiment.reset_index()
    by_sentiment.columns = ["label", "avg_confidence", "count"]
    by_sentiment["count"] = by_sentiment["count"].astype(int)

    by_category = real.groupby("category")["reply_confidence"].apply(_group_stats).unstack()
    by_category = by_category.reset_index()
    by_category.columns = ["label", "avg_confidence", "count"]
    by_category["count"] = by_category["count"].astype(int)

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
    # ---------------------------------------------------------------------
    # เกณฑ์ "ตัวอย่างน้อยเกินไปจะเชื่อถือได้" — หมวด/กลุ่มที่มีคอมเมนต์จริงน้อยกว่านี้
    # จะถูกแสดงเป็นแท่งสีจาง (muted) แทนสีเข้มปกติ เพื่อไม่ให้ตัวเลข 100%/0% ที่มาจาก
    # ตัวอย่างแค่ 1-2 ข้อความดูน่าเชื่อถือเกินจริง (ดู comment ใน compute_summary())
    # ป้ายกำกับแต่ละแท่งจะต่อท้ายด้วย "(n=จำนวนตัวอย่าง)" เสมอ ให้เห็นชัดเจนไม่ต้อง
    # เดา ไม่ได้ซ่อนข้อมูลหมวดไหนออกไปเลย แค่ลดน้ำหนักภาพของหมวดที่ยังสรุปไม่ได้จริง
    # ---------------------------------------------------------------------
    MIN_RELIABLE_N = 3
    MUTED_BLUE = "#c7cdf7"
    MUTED_GREEN = "#bfe8cf"

    # หมายเหตุ: ใช้ .to_dict("records") แทน .itertuples()/.iterrows() ตรงๆ เพราะ
    # itertuples() คืน namedtuple ซึ่งชื่อคอลัมน์ "count" ชนกับเมธอด .count() ที่
    # tuple มีอยู่แล้วในตัว (เข้าถึงด้วย row["count"] แบบ string key ไม่ได้เลย จะ
    # error ทันที ส่วน row.count ก็เสี่ยงกำกวม) แปลงเป็น dict ก่อนจะชัดเจนและปลอดภัยกว่า
    def _labels_with_n(df):
        return [f"{row['label']} (n={row['count']})" for row in df.to_dict("records")]

    def _bar_colors(df, solid_color, muted_color):
        return [solid_color if row["count"] >= MIN_RELIABLE_N else muted_color for row in df.to_dict("records")]

    sentiment_labels = _labels_with_n(summary["by_sentiment"])
    sentiment_values = summary["by_sentiment"]["avg_confidence"].tolist()
    sentiment_colors = _bar_colors(summary["by_sentiment"], "#4f5fe8", MUTED_BLUE)

    # เรียง category จากน้อยไปมากก่อนส่งเข้ากราฟแนวนอน — Chart.js วาดแถวแรกของ labels
    # ไว้บนสุดเสมอ ดังนั้นเรียงน้อยไปมากแบบนี้จะทำให้หมวดที่ความมั่นใจเฉลี่ยต่ำสุด
    # (จุดที่ควรตรวจสอบก่อน) ลอยขึ้นไปอยู่บนสุดของกราฟ เห็นได้ทันทีโดยไม่ต้องเลื่อนดู
    by_category_sorted = summary["by_category"].sort_values("avg_confidence", ascending=True)
    category_labels = _labels_with_n(by_category_sorted)
    category_values = by_category_sorted["avg_confidence"].tolist()
    category_colors = _bar_colors(by_category_sorted, "#16a34a", MUTED_GREEN)

    low_n_categories = int((summary["by_category"]["count"] < MIN_RELIABLE_N).sum())
    low_n_note = (
        f" หมวดที่มีคอมเมนต์น้อยกว่า {MIN_RELIABLE_N} รายการ ({low_n_categories} หมวด) แสดงเป็นแท่งสีจาง เพราะตัวเลข % ยังไม่น่าเชื่อถือพอ"
        if low_n_categories > 0 else ""
    )

    generated_at = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    return f"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Auto Engagement System — Live Dashboard</title>
<meta http-equiv="refresh" content="120">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Sans+Thai:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  :root {{
    --bg: #f3f5f9;
    --surface: #ffffff;
    --border: #e6e9f0;
    --text: #16192b;
    --text-muted: #6b7186;
    --text-faint: #9aa0b4;
    --accent: #4f5fe8;
    --accent-soft: #eef0fd;
    --green: #16a34a;
    --green-soft: #eafaf0;
    --red: #dc2626;
    --red-soft: #fdecec;
    --amber: #d97706;
    --shadow: 0 1px 2px rgba(16,24,64,0.04), 0 8px 24px -12px rgba(16,24,64,0.10);
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: 'Inter', 'IBM Plex Sans Thai', 'Segoe UI', Tahoma, sans-serif;
    background: var(--bg);
    margin: 0;
    padding: 0;
    color: var(--text);
    -webkit-font-smoothing: antialiased;
  }}
  .wrap {{ max-width: 1180px; margin: 0 auto; padding: 32px 28px 48px; }}

  .topbar {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
    margin-bottom: 28px;
  }}
  .topbar h1 {{
    font-size: 24px;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0 0 4px;
  }}
  .topbar .subtitle {{ color: var(--text-muted); font-size: 14px; margin: 0; }}
  .live-badge {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: var(--surface);
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
    border-radius: 999px;
    padding: 8px 16px;
    font-size: 12.5px;
    font-weight: 600;
    color: var(--text-muted);
    white-space: nowrap;
  }}
  .live-dot {{
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--green);
    box-shadow: 0 0 0 0 rgba(22,163,74, 0.55);
    animation: pulse 2s infinite;
  }}
  @keyframes pulse {{
    0%   {{ box-shadow: 0 0 0 0 rgba(22,163,74, 0.55); }}
    70%  {{ box-shadow: 0 0 0 7px rgba(22,163,74, 0); }}
    100% {{ box-shadow: 0 0 0 0 rgba(22,163,74, 0); }}
  }}

  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 16px;
  }}
  @media (max-width: 900px) {{ .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }} }}
  @media (max-width: 520px) {{ .kpi-grid {{ grid-template-columns: 1fr; }} }}

  .kpi-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px 20px;
    box-shadow: var(--shadow);
    display: flex;
    flex-direction: column;
    gap: 10px;
    position: relative;
    overflow: hidden;
  }}
  .kpi-card::before {{
    content: "";
    position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
    background: var(--bar-color, var(--accent));
  }}
  .kpi-top {{ display: flex; align-items: center; justify-content: space-between; }}
  .kpi-icon {{
    width: 34px; height: 34px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    background: var(--icon-bg, var(--accent-soft));
    color: var(--icon-color, var(--accent));
    flex-shrink: 0;
  }}
  .kpi-icon svg {{ width: 18px; height: 18px; }}
  .kpi-label {{ font-size: 12.5px; color: var(--text-muted); font-weight: 600; line-height: 1.4; }}
  .kpi-value {{ font-size: 28px; font-weight: 800; letter-spacing: -0.01em; line-height: 1; color: var(--value-color, var(--text)); }}

  .highlight-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    box-shadow: var(--shadow);
    padding: 20px 24px;
    margin: 16px 0 24px;
    display: flex;
    align-items: center;
    gap: 24px;
    flex-wrap: wrap;
  }}
  .highlight-text {{ min-width: 220px; }}
  .highlight-label {{ font-size: 13px; color: var(--text-muted); font-weight: 600; margin-bottom: 4px; }}
  .highlight-value {{ font-size: 34px; font-weight: 800; color: var(--accent); letter-spacing: -0.02em; }}
  .progress-track {{
    flex: 1;
    min-width: 220px;
    height: 10px;
    background: #eceefb;
    border-radius: 999px;
    overflow: hidden;
  }}
  .progress-fill {{
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #6c7bf5, var(--accent));
  }}

  .charts-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }}
  @media (max-width: 900px) {{ .charts-grid {{ grid-template-columns: 1fr; }} }}

  .chart-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    box-shadow: var(--shadow);
    padding: 20px 22px 8px;
  }}
  .chart-card h2 {{
    font-size: 14.5px;
    font-weight: 700;
    margin: 0 0 2px;
    color: var(--text);
  }}
  .chart-card .chart-sub {{ font-size: 12px; color: var(--text-faint); margin: 0 0 14px; }}
  .chart-card canvas {{ max-height: 300px; }}
  #categoryChart {{ min-height: 340px; }}

  .footer-note {{
    font-size: 12.5px;
    color: var(--text-faint);
    line-height: 1.6;
    margin-top: 28px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
  }}
</style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div>
        <h1>Auto Engagement System</h1>
        <p class="subtitle">Live Dashboard · คำนวณจากคอมเมนต์จริงที่เข้ามาผ่าน Make + Facebook</p>
      </div>
      <div class="live-badge"><span class="live-dot"></span>รีเฟรชอัตโนมัติทุก 2 นาที</div>
    </div>

    <div class="kpi-grid">
      <div class="kpi-card" style="--bar-color:#4f5fe8;">
        <div class="kpi-top">
          <div class="kpi-icon" style="--icon-bg:#eef0fd; --icon-color:#4f5fe8;">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
          </div>
        </div>
        <div class="kpi-label">จำนวนคอมเมนต์จริงที่ตอบแล้ว</div>
        <div class="kpi-value">{summary["total"]:,}</div>
      </div>

      <div class="kpi-card" style="--bar-color:#16a34a;">
        <div class="kpi-top">
          <div class="kpi-icon" style="--icon-bg:#eafaf0; --icon-color:#16a34a;">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>
          </div>
        </div>
        <div class="kpi-label">ตอบด้วยความมั่นใจสูง (&ge;{summary["threshold"]:g}%)</div>
        <div class="kpi-value" style="--value-color:#16a34a;">{summary["correct"]:,}</div>
      </div>

      <div class="kpi-card" style="--bar-color:#dc2626;">
        <div class="kpi-top">
          <div class="kpi-icon" style="--icon-bg:#fdecec; --icon-color:#dc2626;">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
          </div>
        </div>
        <div class="kpi-label">ความมั่นใจต่ำ</div>
        <div class="kpi-value" style="--value-color:#dc2626;">{summary["incorrect"]:,}</div>
      </div>

      <div class="kpi-card" style="--bar-color:#d97706;">
        <div class="kpi-top">
          <div class="kpi-icon" style="--icon-bg:#fef3e2; --icon-color:#d97706;">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>
          </div>
        </div>
        <div class="kpi-label">% ตอบด้วยความมั่นใจสูง</div>
        <div class="kpi-value" style="--value-color:#d97706;">{summary["accuracy_pct"]:g}%</div>
      </div>
    </div>

    <div class="highlight-card">
      <div class="highlight-text">
        <div class="highlight-label">ค่าเฉลี่ยความมั่นใจรวม (Reply Confidence)</div>
        <div class="highlight-value">{summary["avg_reply_conf"]:g}%</div>
      </div>
      <div class="progress-track"><div class="progress-fill" style="width:{min(summary["avg_reply_conf"], 100):g}%;"></div></div>
    </div>

    <div class="charts-grid">
      <div class="chart-card">
        <h2>ความมั่นใจเฉลี่ย แยกตาม Sentiment</h2>
        <p class="chart-sub">ค่าเฉลี่ยคะแนนความมั่นใจของโมเดล (ไม่ใช่สัดส่วนที่ผ่านเกณฑ์ {summary["threshold"]:g}%) ในแต่ละกลุ่มความรู้สึก</p>
        <canvas id="sentimentChart"></canvas>
      </div>
      <div class="chart-card">
        <h2>ความมั่นใจเฉลี่ย แยกตาม Category</h2>
        <p class="chart-sub">เรียงจากค่าต่ำสุดไปสูงสุด เพื่อให้เห็นจุดที่ควรตรวจสอบก่อน{low_n_note}</p>
        <canvas id="categoryChart"></canvas>
      </div>
    </div>

    <div class="footer-note">
      <span>"ความมั่นใจสูง" คำนวณจากคะแนน confidence ของโมเดล ณ ขณะตอบคอมเมนต์จริงแต่ละข้อความ ไม่ใช่การเทียบกับคำตอบที่มนุษย์ยืนยันไว้ล่วงหน้า</span>
      <span>อัปเดตล่าสุด: {generated_at}</span>
    </div>
  </div>

<script>
Chart.defaults.font.family = "'Inter', 'IBM Plex Sans Thai', 'Segoe UI', sans-serif";
Chart.defaults.color = '#6b7186';

const tooltipStyle = {{
  backgroundColor: '#16192b',
  titleFont: {{ weight: '600', size: 12.5 }},
  bodyFont: {{ size: 12.5 }},
  padding: 10,
  cornerRadius: 8,
  displayColors: false,
}};

new Chart(document.getElementById('sentimentChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(sentiment_labels, ensure_ascii=False)},
    datasets: [{{
      label: 'ความมั่นใจเฉลี่ย (%)',
      data: {json.dumps(sentiment_values)},
      backgroundColor: {json.dumps(sentiment_colors)},
      borderRadius: 8,
      maxBarThickness: 56,
    }}]
  }},
  options: {{
    responsive: true,
    scales: {{
      y: {{ beginAtZero: true, max: 100, grid: {{ color: '#eef0f5', drawBorder: false }}, ticks: {{ callback: v => v + '%' }} }},
      x: {{ grid: {{ display: false }} }},
    }},
    plugins: {{ legend: {{ display: false }}, tooltip: tooltipStyle }},
  }}
}});

new Chart(document.getElementById('categoryChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(category_labels, ensure_ascii=False)},
    datasets: [{{
      label: 'ความมั่นใจเฉลี่ย (%)',
      data: {json.dumps(category_values)},
      backgroundColor: {json.dumps(category_colors)},
      borderRadius: 6,
      maxBarThickness: 22,
    }}]
  }},
  options: {{
    indexAxis: 'y',
    responsive: true,
    scales: {{
      x: {{ beginAtZero: true, max: 100, grid: {{ color: '#eef0f5', drawBorder: false }}, ticks: {{ callback: v => v + '%' }} }},
      y: {{ grid: {{ display: false }}, ticks: {{ font: {{ size: 11.5 }} }} }},
    }},
    plugins: {{ legend: {{ display: false }}, tooltip: tooltipStyle }},
  }}
}});
</script>
</body>
</html>"""


def render_no_data_html(reason: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Auto Engagement System — Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=IBM+Plex+Sans+Thai:wght@400;500&display=swap" rel="stylesheet">
<style>
  body {{
    font-family: 'Inter', 'IBM Plex Sans Thai', 'Segoe UI', sans-serif;
    background: #f3f5f9;
    margin: 0;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #16192b;
  }}
  .card {{
    background: #fff;
    border: 1px solid #e6e9f0;
    border-radius: 16px;
    box-shadow: 0 8px 24px -12px rgba(16,24,64,0.12);
    padding: 40px 44px;
    max-width: 420px;
    text-align: center;
  }}
  .icon {{
    width: 48px; height: 48px; margin: 0 auto 16px;
    border-radius: 12px; background: #eef0fd; color: #4f5fe8;
    display: flex; align-items: center; justify-content: center;
  }}
  h2 {{ font-size: 18px; margin: 0 0 8px; }}
  p {{ font-size: 14px; color: #6b7186; line-height: 1.6; margin: 0; }}
</style>
</head>
<body>
  <div class="card">
    <div class="icon">
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
    </div>
    <h2>ยังไม่มีข้อมูลให้แสดงผล</h2>
    <p>{reason}</p>
  </div>
</body>
</html>"""
