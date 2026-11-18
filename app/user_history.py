# -*- coding: utf-8 -*-
"""
user_history.py
===============
คำนวณพฤติกรรมลูกค้าจาก "ประวัติคอมเมนต์ของ user_id เดียวกัน" ที่ระบบเคยตอบไปแล้ว

เดิม engine หา segment จากไฟล์ CRM ที่ใช้ชื่อลูกค้าเป็นรหัส แต่ Facebook ส่ง user_id มา จึงจับคู่
ไม่ได้และทุกคนกลายเป็น Regular โมดูลนี้สร้างคุณลักษณะเดียวกัน (total/positive/negative/complaint/
inactive_days/favorite_hour) จากประวัติจริงของ user_id นั้น:
  - ตอนเริ่มระบบ: โหลดประวัติจากชีต AI_Log (ถ้าเปิด Google Sheets) หรือ outputs/prediction_log.csv
  - ทุกคอมเมนต์ใหม่: เพิ่มเข้าประวัติในหน่วยความจำทันที
ค่าที่คืนเป็นพฤติกรรม "ก่อน" คอมเมนต์ปัจจุบัน (ไม่นับคอมเมนต์ที่กำลังตอบ)
"""
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_lock = threading.Lock()
_hist = defaultdict(list)  # user_id -> [(datetime, sentiment, category)]
_loaded = False
_source = None

COMPLAINT_PREFIXES = ("ติ", "ปัญหา", "ร้องเรียน", "บริการไม่ดี", "อาหารได้ไม่ตรง")


def _parse_ts(x):
    try:
        ts = pd.to_datetime(x, utc=True)
        return None if pd.isna(ts) else ts.to_pydatetime()
    except Exception:
        return None


def _add(user_id, ts, sentiment, category):
    if not user_id or ts is None:
        return
    _hist[str(user_id)].append((ts, str(sentiment or ""), str(category or "")))


def _rows_from_sheet():
    try:
        from app.sheets_logger import _get_worksheet
        ws = _get_worksheet()
        return ws.get_all_records() if ws is not None else None
    except Exception as e:
        print(f"[user_history] อ่านประวัติจาก Google Sheets ไม่สำเร็จ: {e}")
        return None


def load(log_path: Path):
    """โหลดประวัติครั้งแรก (เรียกซ้ำได้ ไม่โหลดซ้ำ)"""
    global _loaded, _source
    with _lock:
        if _loaded:
            return
        _loaded = True
        rows = _rows_from_sheet()
        if rows is not None:
            _source = "google_sheets"
        elif log_path.exists():
            try:
                rows = pd.read_csv(log_path, encoding="utf-8-sig", usecols=lambda c: c in
                                   {"timestamp", "user_id", "sentiment", "category"}).to_dict("records")
                _source = "local_log"
            except Exception as e:
                print(f"[user_history] อ่าน {log_path} ไม่สำเร็จ: {e}")
                rows = []
        else:
            rows, _source = [], "empty"
        for r in rows or []:
            _add(r.get("user_id"), _parse_ts(r.get("timestamp")), r.get("sentiment"), r.get("category"))
        print(f"[user_history] โหลดประวัติ {sum(len(v) for v in _hist.values())} คอมเมนต์ "
              f"จาก {len(_hist)} ผู้ใช้ ({_source})")


def record(user_id, sentiment, category, ts=None):
    with _lock:
        _add(user_id, ts or datetime.now(timezone.utc), sentiment, category)


def features(user_id, now=None):
    """พฤติกรรมของ user_id จากประวัติ หรือ None ถ้าไม่เคยคอมเมนต์มาก่อน"""
    with _lock:
        h = list(_hist.get(str(user_id), []))
    if not h:
        return None
    now = now or datetime.now(timezone.utc)
    last = max(t for t, _, _ in h)
    hours = Counter(t.astimezone().hour for t, _, _ in h)
    return {
        "total_messages": len(h),
        "positive_count": sum(s == "positive" for _, s, _ in h),
        "negative_count": sum(s == "negative" for _, s, _ in h),
        "complaint_count": sum(c.startswith(COMPLAINT_PREFIXES) for _, _, c in h),
        "inactive_days": max(0, (now - last).days),
        "favorite_hour": hours.most_common(1)[0][0],
        "behavior_source": "history",
    }


def info():
    with _lock:
        return {"source": _source, "users": len(_hist), "comments": sum(len(v) for v in _hist.values())}
