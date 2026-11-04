# -*- coding: utf-8 -*-
"""
15_export_gold_standard.py
==========================
รวมป้ายจากผู้ติดป้าย 2 คน (หน้า "ติดป้ายคอมเมนต์ร้านอาหาร" บน claude.ai) เป็น data/gold_standard.csv
และคำนวณความสอดคล้องระหว่างผู้ติดป้าย (Cohen's kappa)

อินพุต: โฟลเดอร์ที่ export จากฐานข้อมูลของหน้า (Claude อ่านด้วย ArtifactData แล้วบันทึกเป็น JSON)
    <labels_dir>/<user_id>/answers/<gold_id>.json   {"sentiment", "category", "unsure", ...}
    และ/หรือ ไฟล์ที่ผู้ติดป้ายกด "คัดลอกผลลัพธ์" ส่งมา (ใช้เมื่อบัญชีเขาบันทึกลงระบบกลางไม่ได้)
    {"gold_labels": 1, "annotator": "...", "labels": {"G001": [sentiment, category, unsure]}}
ใช้งาน:
    python 15_export_gold_standard.py --labels-dir path/to/labels
    python 15_export_gold_standard.py --labels-dir path/to/labels --files somchai.json
    python 15_export_gold_standard.py --files a.json b.json

กติกาการรวม:
- ทั้งคู่เลือก "ไม่เกี่ยวข้อง" -> ตัดออก
- ป้ายตรงกัน -> ใช้ป้ายนั้น
- ไม่ตรงกัน (รวมกรณีคนเดียวเลือก "ไม่เกี่ยวข้อง") -> ปล่อยว่างไว้ และเขียนลง
  data/gold_disagreements.csv ให้ผู้วิจัยตัดสินชี้ขาด (adjudication) แล้วใส่ค่าใน
  คอลัมน์ final_sentiment / final_category จากนั้นรันสคริปต์ซ้ำด้วย --adjudicated
12_compare_models.py ใช้เฉพาะแถวที่มีป้ายสุดท้าย
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
SKIP = "ไม่เกี่ยวข้อง"


def load_labels(labels_dir: Path) -> dict:
    out = {}
    for user_dir in sorted(p for p in labels_dir.iterdir() if p.is_dir()):
        ans_dir = user_dir / "answers"
        if not ans_dir.is_dir():
            continue
        rows = {}
        for f in ans_dir.glob("*.json"):
            body = json.loads(f.read_text(encoding="utf-8"))
            body = body.get("data", body)  # รองรับทั้งไฟล์ที่ห่อด้วย {"data": ...} และไม่ห่อ
            rows[f.stem] = body
        if rows:
            out[user_dir.name] = rows
    return out


def load_pasted(files) -> dict:
    out = {}
    for f in files:
        body = json.loads(Path(f).read_text(encoding="utf-8-sig").strip())
        name = "pasted:" + (body.get("annotator") or Path(f).stem)
        out[name] = {gid: {"sentiment": v[0] or None, "category": v[1], "unsure": bool(v[2])}
                     for gid, v in body.get("labels", {}).items()}
    return out


def kappa(a, b):
    return round(float(cohen_kappa_score(a, b)), 4) if len(a) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", type=Path)
    ap.add_argument("--files", nargs="*", default=[], help="ไฟล์ JSON จากปุ่ม 'คัดลอกผลลัพธ์'")
    ap.add_argument("--adjudicated", type=Path, default=DATA / "gold_disagreements.csv",
                    help="ไฟล์ที่ผู้วิจัยใส่ final_sentiment/final_category แล้ว (ถ้ามี)")
    args = ap.parse_args()

    labels = load_labels(args.labels_dir) if args.labels_dir else {}
    labels.update(load_pasted(args.files))
    if len(labels) < 2:
        raise SystemExit(f"ต้องมีผู้ติดป้ายอย่างน้อย 2 คน พบ {len(labels)} คน")
    # ผู้ติดป้ายที่ทำมากสุด 2 คนแรก -> annotator_1 / annotator_2 (เรียงตามจำนวนข้อ แล้วตาม id)
    users = sorted(labels, key=lambda u: (-len(labels[u]), u))[:2]
    a1, a2 = labels[users[0]], labels[users[1]]
    print(f"annotator_1 = {users[0]} ({len(a1)} ข้อ), annotator_2 = {users[1]} ({len(a2)} ข้อ)")

    cand = pd.read_csv(DATA / "gold_candidates.csv", encoding="utf-8-sig")
    if "status" in cand.columns:  # ข้อที่ถูกแทนที่ด้วย --refine ไม่นับ
        cand = cand[cand["status"] == "active"]
    rows = []
    for r in cand.itertuples():
        x, y = a1.get(r.gold_id), a2.get(r.gold_id)
        if not x or not y:
            continue
        rows.append({
            "gold_id": r.gold_id, "message": r.message, "source": r.source,
            "annotator_1_sentiment": x.get("sentiment") or "", "annotator_2_sentiment": y.get("sentiment") or "",
            "annotator_1_category": x.get("category") or "", "annotator_2_category": y.get("category") or "",
            "unsure": bool(x.get("unsure")) or bool(y.get("unsure")),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("ยังไม่มีข้อความที่ทั้งสองคนติดป้ายแล้ว")

    both_skip = (df.annotator_1_category == SKIP) & (df.annotator_2_category == SKIP)
    one_skip = (df.annotator_1_category == SKIP) ^ (df.annotator_2_category == SKIP)
    rel = df[~both_skip & ~one_skip]

    report = {
        "n_double_labeled": int(len(df)), "n_both_irrelevant": int(both_skip.sum()),
        "n_one_irrelevant": int(one_skip.sum()), "n_relevant": int(len(rel)),
        "kappa_sentiment": kappa(rel.annotator_1_sentiment, rel.annotator_2_sentiment),
        "kappa_category": kappa(rel.annotator_1_category, rel.annotator_2_category),
        "kappa_relevance": kappa(df.annotator_1_category == SKIP, df.annotator_2_category == SKIP),
        "agreement_sentiment": round(float((rel.annotator_1_sentiment == rel.annotator_2_sentiment).mean()), 4) if len(rel) else None,
        "agreement_category": round(float((rel.annotator_1_category == rel.annotator_2_category).mean()), 4) if len(rel) else None,
        "by_source": {},
    }
    for src, g in rel.groupby("source"):
        report["by_source"][src] = {"n": int(len(g)),
                                    "kappa_sentiment": kappa(g.annotator_1_sentiment, g.annotator_2_sentiment),
                                    "kappa_category": kappa(g.annotator_1_category, g.annotator_2_category)}

    df = df[~both_skip].copy()
    agree_s = df.annotator_1_sentiment == df.annotator_2_sentiment
    agree_c = df.annotator_1_category == df.annotator_2_category
    df["sentiment"] = df.annotator_1_sentiment.where(agree_s & (df.annotator_1_category != SKIP), "")
    df["category"] = df.annotator_1_category.where(agree_c & (df.annotator_1_category != SKIP), "")

    # ผลชี้ขาดจากผู้วิจัย (ถ้ามี)
    if args.adjudicated.exists():
        adj = pd.read_csv(args.adjudicated, encoding="utf-8-sig", dtype=str).fillna("")
        if {"gold_id", "final_sentiment", "final_category"} <= set(adj.columns):
            adj = adj.set_index("gold_id")
            for i, r in df.iterrows():
                if r.gold_id in adj.index:
                    fs, fc = adj.at[r.gold_id, "final_sentiment"], adj.at[r.gold_id, "final_category"]
                    if fc == SKIP:
                        df.at[i, "sentiment"], df.at[i, "category"] = "", SKIP
                        continue
                    if fs:
                        df.at[i, "sentiment"] = fs
                    if fc:
                        df.at[i, "category"] = fc
    df = df[df.category != SKIP]

    need = df[(df.sentiment == "") | (df.category == "")]
    if not args.adjudicated.exists() or not len(pd.read_csv(args.adjudicated, encoding="utf-8-sig")):
        out = need[["gold_id", "message", "annotator_1_sentiment", "annotator_2_sentiment",
                    "annotator_1_category", "annotator_2_category"]].assign(final_sentiment="", final_category="")
        out.to_csv(DATA / "gold_disagreements.csv", index=False, encoding="utf-8-sig")
        print(f"ข้อที่ต้องชี้ขาด: {len(out)} ข้อ -> {DATA / 'gold_disagreements.csv'}")
    else:
        print(f"ยังต้องชี้ขาดอีก {len(need)} ข้อ (ดู {args.adjudicated})")

    cols = ["gold_id", "message", "sentiment", "category", "source", "annotator_1_sentiment", "annotator_2_sentiment",
            "annotator_1_category", "annotator_2_category", "unsure"]
    df[cols].to_csv(DATA / "gold_standard.csv", index=False, encoding="utf-8-sig")
    (BASE / "outputs").mkdir(exist_ok=True)
    (BASE / "outputs" / "gold_agreement.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"บันทึก {DATA / 'gold_standard.csv'} ({len(df)} แถว, ป้ายครบ {len(df) - len(need)} แถว)")


if __name__ == "__main__":
    main()
