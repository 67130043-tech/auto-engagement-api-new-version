# -*- coding: utf-8 -*-
"""
โมดูลสำหรับบันทึกผลการตอบกลับแต่ละคอมเมนต์ลง Google Sheets โดยตรงจาก Python
(แทน/เสริมจากการให้ Make เขียนผ่านโมดูล "Google Sheets: Add Row" เอง)

วิธีตั้งค่า (ทำครั้งเดียว):
1) ไปที่ https://console.cloud.google.com -> สร้างโปรเจกต์ -> เปิดใช้งาน
   "Google Sheets API" และ "Google Drive API"
2) สร้าง Service Account -> สร้างคีย์ประเภท JSON -> ดาวน์โหลดไฟล์เก็บไว้เป็น
   credentials/service_account.json (โฟลเดอร์ credentials/ อยู่ในโปรเจกต์นี้)
3) เปิด Google Sheet ที่ต้องการใช้ -> Share -> แชร์ให้กับอีเมลของ Service Account
   (อีเมลอยู่ในไฟล์ json ช่อง "client_email") ให้สิทธิ์ระดับ Editor
4) ตั้งค่า environment variable 2 ตัว (หรือแก้ค่า default ด้านล่างตรง ๆ ก็ได้):
   - GOOGLE_SHEET_ID          = ID ของ Google Sheet (ดูจาก URL ระหว่าง /d/ กับ /edit)
   - GOOGLE_CREDENTIALS_PATH  = พาธไปยังไฟล์ credentials/service_account.json
   - ENABLE_GOOGLE_SHEETS     = "true" เพื่อเปิดใช้งาน (ค่าเริ่มต้นคือปิดไว้ ถ้ายังไม่ตั้งค่า)

ถ้ายังไม่ได้ตั้งค่า/เชื่อมต่อไม่สำเร็จ ระบบจะไม่ error และไม่หยุดการตอบกลับลูกค้า
(แค่บันทึกลงไฟล์ local ต่อไปตามปกติ) เพื่อไม่ให้ Google Sheets ล่มแล้วกระทบระบบหลัก
"""
import os
from pathlib import Path

_gc = None
_worksheet = None
_init_attempted = False

SHEET_HEADER = [
    "timestamp", "user_id", "display_name", "channel", "source",
    "message", "clean_text",
    "sentiment", "sentiment_confidence",
    "category", "category_confidence",
    "segment", "action", "reply_message", "reply_confidence",
    "behavior_total_messages", "behavior_positive_count", "behavior_negative_count",
    "behavior_complaint_count", "behavior_inactive_days", "behavior_favorite_hour",
]


def _is_enabled() -> bool:
    return os.environ.get("ENABLE_GOOGLE_SHEETS", "false").strip().lower() == "true"


def _get_worksheet():
    """เชื่อมต่อกับ Google Sheet ครั้งแรกแล้วเก็บ cache ไว้ใช้ซ้ำ (เชื่อมครั้งเดียวต่อการรัน)"""
    global _gc, _worksheet, _init_attempted
    if _worksheet is not None:
        return _worksheet
    if _init_attempted:
        return None
    _init_attempted = True

    if not _is_enabled():
        return None

    sheet_id = os.environ.get("GOOGLE_SHEET_ID", "")
    creds_path = os.environ.get("GOOGLE_CREDENTIALS_PATH", "credentials/service_account.json")

    if not sheet_id or not Path(creds_path).exists():
        print("[sheets_logger] ยังไม่ได้ตั้งค่า GOOGLE_SHEET_ID หรือหาไฟล์ credentials ไม่เจอ "
              "ข้ามการเชื่อมต่อ Google Sheets (ระบบยังทำงานปกติ บันทึกลงไฟล์ local ต่อไป)")
        return None

    try:
        import gspread
        from google.oauth2.service_account import Credentials

        scopes = ["https://www.googleapis.com/auth/spreadsheets",
                  "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_file(creds_path, scopes=scopes)
        _gc = gspread.authorize(creds)
        sh = _gc.open_by_key(sheet_id)

        try:
            ws = sh.worksheet("prediction_log")
        except gspread.exceptions.WorksheetNotFound:
            ws = sh.add_worksheet(title="prediction_log", rows=1000, cols=len(SHEET_HEADER))
            ws.append_row(SHEET_HEADER)

        if ws.row_count == 0 or not ws.row_values(1):
            ws.append_row(SHEET_HEADER)

        _worksheet = ws
        print("[sheets_logger] เชื่อมต่อ Google Sheets สำเร็จ จะบันทึกคอมเมนต์จริงทุกครั้งลงชีตนี้ด้วย")
        return _worksheet
    except Exception as e:
        print(f"[sheets_logger] เชื่อมต่อ Google Sheets ไม่สำเร็จ: {e} "
              "(ระบบยังทำงานปกติ บันทึกลงไฟล์ local ต่อไป)")
        return None


def log_to_sheet(result: dict):
    """บันทึก 1 แถวลง Google Sheet ตาม SHEET_HEADER ไม่ทำให้ API ล่มถ้า Sheets มีปัญหา"""
    ws = _get_worksheet()
    if ws is None:
        return False
    try:
        behavior = result.get("behavior", {}) or {}
        row = [
            result.get("timestamp", ""),
            result.get("user_id", ""),
            result.get("display_name", ""),
            result.get("channel", ""),
            result.get("source", ""),
            result.get("message", ""),
            result.get("clean_text", ""),
            result.get("sentiment", ""),
            result.get("sentiment_confidence", ""),
            result.get("category", ""),
            result.get("category_confidence", ""),
            result.get("segment", ""),
            result.get("action", ""),
            result.get("reply_message", ""),
            result.get("reply_confidence", ""),
            behavior.get("total_messages", ""),
            behavior.get("positive_count", ""),
            behavior.get("negative_count", ""),
            behavior.get("complaint_count", ""),
            behavior.get("inactive_days", ""),
            behavior.get("favorite_hour", ""),
        ]
        ws.append_row(row, value_input_option="USER_ENTERED")
        return True
    except Exception as e:
        print(f"[sheets_logger] บันทึกแถวลง Google Sheets ไม่สำเร็จ: {e}")
        return False
