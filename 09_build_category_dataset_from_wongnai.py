# -*- coding: utf-8 -*-
"""
09_build_category_dataset_from_wongnai.py
===========================================
สร้าง training dataset สำหรับ category_model (data/category_dataset_v2.csv) จาก
รีวิวร้านอาหารจริงล้วน ไม่มีประโยคสังเคราะห์หรือประโยคที่เขียนขึ้นเอง

ทำไมต้องสร้างใหม่ (พบ 11 ก.ย.): Dataset_Restaurant_CRM_4000_Rows.xlsx เดิมมี 4,000 แถว
แต่ข้อความไม่ซ้ำกันจริงแค่ 85 ประโยค train/test split จึงมีประโยคเดียวกันทั้งสองฝั่ง
accuracy 100% ที่วัดได้เป็นตัวเลขหลอก

วิธีสร้าง (weak supervision — ติดป้ายด้วยกฎแทนคน):
  1. ตัดรีวิว Wongnai (data/wongnai.csv, 19,458 รีวิว) เป็นประโยค/ท่อนสั้น ให้ใกล้
     เคียงคอมเมนต์ Facebook มากขึ้น
  2. ติดป้ายแต่ละท่อนด้วยกฎ keyword ของระบบ (keyword_category_*) แล้วแปลงเป็น 10 คลาส
     เก็บเฉพาะท่อนที่กฎจับได้จริง                                   -> source=wongnai_real
     ยกเว้นท่อนที่ pattern "อาหารได้ไม่ตรง/ช้า" จับได้ ให้เข้าหมวดนั้น
     (keyword table แม็พมาหมวดนี้ได้แค่ "ยกเลิก/คืนเงิน")          -> source=wongnai_pattern
  3. หมวดที่ยังไม่ถึง 1,200 เติมด้วยท่อนจริงที่ pattern เฉพาะหมวดจับได้
     (TOPUP_PATTERNS / is_order_issue) ไล่หยิบตามลำดับ:
       ก. ท่อนจาก data/wongnai.csv ที่กฎ keyword ไม่ได้ติดป้าย        -> wongnai_pattern
       ข. ท่อนจาก Wongnai corpus ฉบับเต็ม (46,203 รีวิว) ที่ไม่ซ้ำกับ
          data/wongnai.csv — ดู data/external/wongnai-corpus/README.md -> wongnai_corpus_pattern
       ค. ท่อนจาก data/wongnai.csv ในหมวดที่มีเกิน 2 เท่าของ cap (ถูกสุ่มตัดทิ้ง
          อยู่แล้ว หมวดต้นทางจึงยังเต็ม 1,200)                       -> wongnai_pattern
  4. ลบข้อความซ้ำ แล้วสุ่มจำกัดไม่เกิน 1,200 แถวต่อหมวด (random_state=42)

หมวดที่รีวิวจริงมีไม่ถึง 1,200 (รีวิวส่วนใหญ่เป็นการนั่งกินที่ร้าน จึงมีเรื่องเดลิเวอรี่และ
เรื่องได้อาหารผิด/ช้าน้อย) จะมีแถวน้อยกว่า 1,200 ตามจริง ส่วนต่างที่เหลือชดเชยด้วย
class_weight="balanced" ตอนเทรน (ดู 02_train_models.py) — เลือกไม่เติมด้วยประโยคสังเคราะห์
เพราะประโยคจากแม่แบบเดียวกันคล้ายกันมาก ทำให้ผลวัดสูงเกินจริง

pattern ทุกตัวตั้งใจให้แคบ (precision มาก่อน recall) เพราะผลลัพธ์กลายเป็น label

ผลลัพธ์: data/category_dataset_v2.csv (คอลัมน์ clean_text, Category, source, split, text)
        text = ท่อนข้อความดิบก่อน clean_text (มีอีโมจิ/วรรคตอน) สำหรับ WangchanBERTa
        และสำเนาที่ outputs/category_dataset_v2.csv
ผลทำซ้ำได้: ทุกขั้นเรียงข้อมูลก่อนสุ่ม รันกี่ครั้งก็ได้ไฟล์เดียวกัน
"""
import io
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from app.preprocess import clean_text, tokenize_boundary
from app.keywords_data import (
    keyword_category_and_match,
    keyword_category_best_match,
)
from app.engine import map_to_ml_category, ML_CATEGORY_CLASSES

OUT = BASE / "outputs"
DATA = BASE / "data"
EXTRA_CORPUS_ZIP = DATA / "external" / "wongnai-corpus" / "review_dataset.zip"
EXTRA_CORPUS_URL = "https://github.com/wongnai/wongnai-corpus/raw/master/review/review_dataset.zip"
MAX_PER_CATEGORY = 1200


# ---------------------------------------------------------------------------
# 1) โหลดรีวิวแล้วตัดเป็นท่อนสั้น
# ---------------------------------------------------------------------------
def split_into_chunks(text: str):
    """ตัดรีวิวยาวๆ เป็นประโยค/ท่อนสั้นๆ ด้วยตัวแบ่งที่มีจริงในข้อความ (ขึ้นบรรทัดใหม่,
    จุด/!/? ตามด้วยช่องว่าง, ไม้ยมกตามด้วยช่องว่าง, ช่องว่างตั้งแต่ 2 ตัว) เก็บเฉพาะ
    ท่อนที่ยาวตั้งแต่ 6 อักขระ"""
    text = str(text)
    parts = re.split(r"[\n\r]+|(?<=[.!?])\s+|ๆ\s+|\s{2,}", text)
    return [p.strip() for p in parts if p and len(p.strip()) >= 6]


def _chunks_of(reviews) -> set:
    out = set()
    for review in reviews:
        out.update(split_into_chunks(review))
    return out


def load_wongnai_chunks():
    df = pd.read_csv(DATA / "wongnai.csv", encoding="cp874", on_bad_lines="skip")
    # เรียงก่อนคืนค่าเสมอ: ลำดับของ set เปลี่ยนทุก process (string hash randomization)
    # ถ้าไม่เรียง .sample(random_state=42) จะได้แถวคนละชุดทุกครั้ง dataset ทำซ้ำไม่ได้
    return sorted(_chunks_of(df["review_body"].astype(str)))


def load_extra_corpus_chunks(exclude: set):
    """ท่อนจาก Wongnai corpus ฉบับเต็ม (train 40,000 + test 6,203 รีวิว) ที่ไม่ซ้ำกับ
    exclude (ท่อนจาก data/wongnai.csv) ถ้าไม่มีไฟล์ในเครื่อง จะดาวน์โหลดจาก GitHub"""
    if not EXTRA_CORPUS_ZIP.exists():
        import requests
        print(f"  ดาวน์โหลด Wongnai corpus จาก {EXTRA_CORPUS_URL} ...")
        EXTRA_CORPUS_ZIP.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(EXTRA_CORPUS_URL, timeout=120)
        resp.raise_for_status()
        EXTRA_CORPUS_ZIP.write_bytes(resp.content)
    with zipfile.ZipFile(EXTRA_CORPUS_ZIP) as z:
        train = pd.read_csv(io.BytesIO(z.read("w_review_train.csv")), sep=";", header=None,
                            names=["review", "rating"], engine="python")
        test = pd.read_csv(io.BytesIO(z.read("test_file.csv")), sep=";", engine="python")
    reviews = pd.concat([train["review"], test["review"]]).astype(str)
    print(f"  Wongnai corpus ฉบับเต็ม: {len(train)} + {len(test)} รีวิว")
    return sorted(_chunks_of(reviews) - set(exclude))


# ---------------------------------------------------------------------------
# 2) ติดป้ายด้วยกฎ keyword ของระบบ
# ---------------------------------------------------------------------------
def label_chunk(chunk: str):
    text_boundary = tokenize_boundary(clean_text(chunk))
    category_detail = keyword_category_and_match(text_boundary) or keyword_category_best_match(text_boundary)
    if not category_detail:
        return None
    category_ml = map_to_ml_category(category_detail, None)
    return category_ml if category_ml in ML_CATEGORY_CLASSES else None


# ---------------------------------------------------------------------------
# 3) pattern เฉพาะหมวด
# ---------------------------------------------------------------------------
# "อาหารได้ไม่ตรง/ช้า": keyword table แม็พมาหมวดนี้ได้แค่ "ยกเลิก/คืนเงิน" (แคบมาก)
# ทั้งที่รีวิวจริงมีคนบ่น "รออาหารนาน / เสิร์ฟผิด / ออเดอร์ตกหล่น" อยู่ไม่น้อย
#   - SLOW   : ช้าหลังสั่งเท่านั้น ไม่นับการรอคิวหน้าร้าน
#   - WRONG  : เสิร์ฟผิด/ไม่ครบ/ตกหล่น/ไม่ตรงที่สั่ง/ต้องทวง
#   - EXCLUDE: รูปปฏิเสธและความหมายอื่นที่ pattern จับผิด เช่น "ไม่ผิดหวัง", "ออเดอร์
#     ไม่ตกหล่น", "ทำผิดกฎจราจร", "ถ่ายรูปมาไม่ครบ", "รอคิว", "ถือว่าไวมาก",
#     "สั่งผิดเมนู"/"ผิดเอง" ที่ลูกค้าผิดเอง
# ---------------------------------------------------------------------------
_ORDER_SLOW = (
    r"(อาหาร|ของ|จาน|เมนู)(ที่สั่ง)?(มา|ออก|ได้|เสิร์ฟ|เสิฟ)(ค่อนข้าง)?(ช้า|นาน)"
    r"|(เสิร์ฟ|เสิฟ)(ช้า|นาน)"
    r"|อาหาร(ค่อนข้าง)?ช้า|ทำ(อาหาร)?(ค่อนข้าง)?ช้า"
    r"|รอ(อาหาร|ออเดอร์|ของ)(นาน|ประมาณ|เกือบ|ตั้ง|ร่วม|อีก|กว่า)"
    r"|สั่ง(ไป|แล้ว|อาหาร).{0,25}รอ.{0,15}(นาน|ชั่วโมง|ชม|นาที)"
    r"|กว่า(อาหาร|จะได้อาหาร|ของ).{0,10}(จะ)?(มา|ได้|ออก)"
)
_ORDER_WRONG = (
    r"(เสิร์ฟ|เสิฟ|เสริฟ|เสริพ|ทำ|ได้|จด|ส่ง)(อาหาร|เมนู|ของ)?ผิด(?!หวัง|กฎ)"
    r"|ผิดออเดอร์|ออเดอร์ผิด|ผิดเมนู|ไม่ตรง(กับ)?(ที่|ตาม)สั่ง|ไม่ได้ตามที่สั่ง"
    r"|(ได้|เอา|เสิร์ฟ|เสิฟ|ให้)(ของ|อาหาร|เมนู|จาน)?ที่ไม่ได้สั่ง"
    r"|(มา|ได้|เสิร์ฟ|เสิฟ)(อาหาร|ของ|เมนู)?ไม่ครบ|ลืม(ออเดอร์|เสิร์ฟ|เสิฟ|ทำ)"
    r"|ตกหล่น|ทวง(อาหาร|ออเดอร์|ถาม|เมนู)|ไม่ได้รับ(อาหาร|ของ|เมนู)|ไม่เข้าระบบ"
    r"|สั่ง.{1,40}แต่(ได้|เอา|ส่ง|เสิร์ฟ|เสิฟ|ทำ).{1,30}มา(ให้)?แทน"  # สั่ง X แต่ได้ Y มาแทน
)
_ORDER_EXCLUDE = (
    r"รอคิว|ต่อคิว|คิวยาว|จองคิว|ที่จอด|ผิดหวัง|ผิดกฎ|ลืมสั่ง|ถ่าย|ไม่ได้ผิด"
    r"|รอไม่นาน|ไม่ต้องรอ|ไม่ช้า|ไม่นาน|ไม่มีตกหล่น|ไม่ตกหล่น|ไม่เคยผิด|เร็ว|ไวมาก|ไวดี"
    r"|สั่งผิด|ผิดเอง"
)
ORDER_ISSUE_PATTERN = re.compile(f"{_ORDER_SLOW}|{_ORDER_WRONG}")
ORDER_ISSUE_EXCLUDE = re.compile(_ORDER_EXCLUDE)
ORDER_ISSUE_CATEGORY = "อาหารได้ไม่ตรง/ช้า"
ORDER_ISSUE_MAX_LEN = 250  # ท่อนยาวมากมักเล่าหลายเรื่องปนกัน label จะเพี้ยน


def is_order_issue(chunk: str) -> bool:
    chunk = str(chunk)
    if not (8 <= len(chunk) <= ORDER_ISSUE_MAX_LEN):
        return False
    return bool(ORDER_ISSUE_PATTERN.search(chunk)) and not ORDER_ISSUE_EXCLUDE.search(chunk)


# หมวดอื่นที่ keyword ติดป้ายได้ไม่ถึง cap (หมายเหตุ: "สอบถามโปรโมชั่น" ในรีวิวจริงคือการ
# "พูดถึง" ดีล/ส่วนลด ไม่ใช่การถามตรงๆ — สอดคล้องกับที่ keyword ติดป้ายหมวดนี้อยู่แล้ว)
TOPUP_PATTERNS = {
    "การจัดส่ง (Delivery)": (
        re.compile(
            r"(?i)เดลิเวอ|เดริเวอ|delivery|grab ?food|แกร็บ|แกรบ|grab(?!\s+(a|an|some|your|the|and|&)\b)"
            r"|line ?man|ไลน์แมน|food ?panda|ฟู้ดแพนด้า|robinhood|โรบินฮู้ด|ไรเดอร์|ค่าส่ง|คนส่ง"
            r"|ส่งถึงบ้าน|(มา|ไป)ส่ง(ถึง|ที่บ้าน|ที่ออฟฟิศ|ที่ทำงาน|ที่คอนโด)|ส่งอาหาร|ส่งมาถึง"
            r"|สั่ง.{0,30}(มาส่ง|ส่งมา)|สั่ง(ผ่าน|ทาง)(แอ|app)|ส่งฟรี|ส่งไว|ส่งเร็ว|ส่งช้า|ส่งนาน"
            r"|(?<!คูล)แพ็ค(มา|อย่าง)ดี|แพคเกจจิ้ง|packaging"
        ),
        # ป้ายรางวัล/แฮชแท็ก LINEMAN Wongnai Users' Choice, ลิงก์, ท่อนภาษาอังกฤษล้วน, เรียกรถ,
        # มาสค์ไรเดอร์ (ตัวการ์ตูน), ซัพพลายเออร์ส่งของให้ร้าน
        re.compile(r"(?i)users.?ch|choice|รางวัล|https?://|www\.|^[\x00-\x7f]+$|แท็กซี่|taxi|เรียกรถ"
                   r"|มาสค์|kamen|ยังไม่มาส่ง|ส่งให้ชิม|ส่งมาให้ชิม"),
    ),
    "ชมการบริการ": (
        re.compile(
            r"(พนักงาน|น้องๆ?|พี่ๆ?|เจ้าของร้าน|แม่ค้า|พ่อค้า|เด็กเสิร์ฟ|การบริการ|บริการ)"
            r"(ทุกคน|ที่ร้าน|ในร้าน)?\s?(ก็|ค่อนข้าง)?\s?(บริการ)?"
            r"(ดีมาก|ดี(?!ไม่|ขึ้น|กว่า)|ดีเยี่ยม|น่ารัก|ยิ้มแย้ม|เป็นกันเอง|อัธยาศัยดี|ใส่ใจ"
            r"|เอาใจใส่|สุภาพ|รวดเร็ว|ประทับใจ|ดูแลดี|ดูแลอย่างดี|แนะนำเมนู)"
        ),
        # ตัดประโยคติ/เสนอแนะ เช่น "บริการไม่ค่อยดี", "น่าจะบริการดีกว่านี้"
        re.compile(r"ไม่ค่อย|ไม่ได้|ไม่ดี|แย่|ช้า|ห่วย|ไม่ประทับใจ|หน้าบึ้ง|ไม่สนใจ|ไม่ใส่ใจ"
                   r"|กว่านี้|น่าจะ|ควรจะ|ดีเด่น"),
    ),
    "สอบถามโปรโมชั่น": (
        re.compile(
            r"(?i)โปรโมชั่น|โปรโมชัน|โปร(?=\s|ลด|วัน|เดือน|บัตร|นี้|ดี(?!๊))|ส่วนลด|ลด\s?\d+\s?(%|เปอร์|บาท)"
            r"|คูปอง|voucher|วอเชอร์|บัตรสมาชิก|สมาชิก(ลด|ได้)|ซื้อ\s?\d\s?แถม\s?\d|1\s?แถม\s?1"
            r"|happy ?hour|ราคาพิเศษ|ดีล(?!ิ)|deal"
        ),
        re.compile(r"ไม่มีโปร|ไม่ลด|โปรแกรม|โปรดิว|โปรไฟล์|โปรตีน|โปรด|สิงคโปร์|^[\x00-\x7f]+$"),
    ),
}
TOPUP_MAX_LEN = 300


def pattern_category(chunk: str):
    """หมวดที่ pattern จับได้ ("อาหารได้ไม่ตรง/ช้า" ก่อน แล้ว TOPUP_PATTERNS) หรือ None"""
    chunk = str(chunk)
    if is_order_issue(chunk):
        return ORDER_ISSUE_CATEGORY
    if not (8 <= len(chunk) <= TOPUP_MAX_LEN):
        return None
    for cat, (pattern, exclude) in TOPUP_PATTERNS.items():
        if pattern.search(chunk) and not exclude.search(chunk):
            return cat
    return None


# ---------------------------------------------------------------------------
# 4) สร้าง dataset
# ---------------------------------------------------------------------------
def main():
    print("กำลังโหลดและตัดรีวิว Wongnai (data/wongnai.csv) เป็นประโยคย่อย...")
    chunks = load_wongnai_chunks()
    print(f"  ประโยค/ท่อนที่ไม่ซ้ำกันทั้งหมด: {len(chunks)}")

    print("กำลังติดป้ายด้วยกฎ keyword (weak supervision)...")
    labeled = {}  # clean_text -> (Category, source, ข้อความดิบ)
    for chunk in chunks:
        key = clean_text(chunk)
        if not key or key in labeled:
            continue
        if is_order_issue(chunk):
            labeled[key] = (ORDER_ISSUE_CATEGORY, "wongnai_pattern", chunk)
            continue
        cat = label_chunk(chunk)
        if cat is not None:
            labeled[key] = (cat, "wongnai_real", chunk)
    print(f"  ติดป้ายได้: {len(labeled)}")

    # --- เติมหมวดที่ยังไม่ถึง cap ด้วยท่อนจริงที่ pattern จับได้ ---
    counts = pd.Series([v[0] for v in labeled.values()]).value_counts()
    need = {c: MAX_PER_CATEGORY - int(counts.get(c, 0)) for c in ML_CATEGORY_CLASSES}
    need = {c: n for c, n in need.items() if n > 0}
    surplus = {c for c, n in counts.items() if n > 2 * MAX_PER_CATEGORY}
    print("  หมวดที่ยังขาด:", dict(sorted(need.items())))

    print("กำลังโหลด Wongnai corpus ฉบับเต็ม (แหล่งเสริม)...")
    extra_chunks = load_extra_corpus_chunks(exclude=chunks)
    print(f"  ท่อนใหม่ที่ไม่ซ้ำกับ data/wongnai.csv: {len(extra_chunks)}")

    pools = [
        ("wongnai_pattern", [ch for ch in chunks if clean_text(ch) not in labeled]),
        ("wongnai_corpus_pattern", extra_chunks),
        ("wongnai_pattern", [ch for ch in chunks if labeled.get(clean_text(ch), (None,))[0] in surplus]),
    ]
    added = {}
    for source, pool in pools:
        for chunk in pool:
            if not need:
                break
            cat = pattern_category(chunk)
            if cat not in need:
                continue
            key = clean_text(chunk)
            if not key or key in added or labeled.get(key, (None,))[0] == cat:
                continue
            added[key] = (cat, source, chunk)
            need[cat] -= 1
            if need[cat] == 0:
                del need[cat]
    labeled.update(added)
    print("  เติมด้วย pattern:", pd.Series([f"{c} <- {s}" for c, s, _ in added.values()]).value_counts().to_dict())

    df = pd.DataFrame(
        [{"clean_text": k, "Category": c, "source": s, "text": raw} for k, (c, s, raw) in labeled.items()]
    ).sort_values(["Category", "source", "clean_text"]).reset_index(drop=True)

    # จำกัดไม่เกิน cap ต่อหมวด — สุ่มแบบ random_state คงที่เพื่อทำซ้ำได้ แถวจาก pattern
    # (ที่เติมเพื่อให้หมวดนั้นถึง cap พอดี) จึงเก็บไว้ทั้งหมดก่อน แล้วค่อยสุ่มจาก keyword
    parts = []
    for cat, grp in df.groupby("Category"):
        if len(grp) <= MAX_PER_CATEGORY:
            parts.append(grp)
            continue
        keep = grp[grp["source"] != "wongnai_real"]
        rest = grp[grp["source"] == "wongnai_real"]
        parts.append(keep)
        parts.append(rest.sample(MAX_PER_CATEGORY - len(keep), random_state=42))
    df = pd.concat(parts).sort_values(["Category", "source", "clean_text"]).reset_index(drop=True)

    print(f"\nสรุปจำนวนต่อหมวด (cap {MAX_PER_CATEGORY}):")
    table = pd.crosstab(df["Category"], df["source"], margins=True, margins_name="รวม")
    print(table.to_string())
    counts = df["Category"].value_counts()
    print(f"\nอัตราส่วนหมวดใหญ่สุด:เล็กสุด = {counts.max() / counts.min():.2f}:1 "
          "(ส่วนที่เหลือชดเชยด้วย class_weight ตอนเทรน ดู 02_train_models.py)")

    # คอลัมน์ split (train/test) กำหนดในไฟล์ เพื่อให้ทุกโมเดล (TF-IDF+SVM, WangchanBERTa,
    # LLM) วัดผลบนแถวทดสอบชุดเดียวกันเป๊ะ — stratified ตามหมวด, random_state=42
    from sklearn.model_selection import train_test_split
    _, test_idx = train_test_split(df.index, test_size=0.20, random_state=42, stratify=df["Category"])
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"

    out_path = DATA / "category_dataset_v2.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "category_dataset_v2.csv", index=False, encoding="utf-8-sig")
    print(f"\nบันทึกไว้ที่: {out_path}  (รวม {len(df)} แถว)")


if __name__ == "__main__":
    main()
