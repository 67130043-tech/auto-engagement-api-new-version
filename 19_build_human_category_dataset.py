"""รวมป้ายข้อมูลฝึกที่คนติด (ผู้ติดป้าย 2 คน + การชี้ขาด) เป็นชุดข้อมูลหมวดหมู่ 8 คลาสจากคอมเมนต์จริง

กติกาการรวม:
  - ข้อที่ติดป้ายทั้ง 2 คนและตรงกัน -> ใช้ป้ายนั้น
  - ข้อที่ไม่ตรงกัน -> ใช้ผลชี้ขาด (data/train_labels/adjudication.json)
  - ข้อที่ติดป้ายคนเดียว -> ใช้ป้ายของคนนั้น
  - ตัดข้อที่สุดท้ายเป็น "ไม่เกี่ยวข้อง"
และรายงานความสอดคล้อง (Cohen's kappa) บนข้อที่ติดป้ายซ้ำ

    python 19_build_human_category_dataset.py -> data/category_human_v3.csv, outputs/train_label_agreement.json
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

BASE = Path(__file__).resolve().parent
SKIP = "ไม่เกี่ยวข้อง"


def main():
    items = pd.read_csv(BASE / "data/train_label_items.csv", encoding="utf-8-sig")
    A = json.load(open(BASE / "data/train_labels/annotator_A.json", encoding="utf-8"))["labels"]
    B = json.load(open(BASE / "data/train_labels/annotator_B.json", encoding="utf-8"))["labels"]
    adj = json.load(open(BASE / "data/train_labels/adjudication.json", encoding="utf-8"))

    rows, how = [], {}
    for r in items.itertuples():
        i = r.item_id
        a, b = A.get(i), B.get(i)
        if a and b:
            same = a[1] == b[1] and (a[1] == SKIP or a[0] == b[0])
            if same:
                s, c, src = a[0], a[1], "agreed"
            elif i in adj:
                s, c, src = adj[i]["sentiment"], adj[i]["category"], "adjudicated"
            else:
                raise SystemExit("missing adjudication for " + i)
            n = 2
        else:
            x = a or b
            if not x:
                raise SystemExit("unlabeled " + i)
            s, c, src, n = x[0], x[1], "single", 1
        how[src] = how.get(src, 0) + 1
        if c == SKIP:
            continue
        rows.append({"item_id": i, "text": r.message, "Category": c, "sentiment": s, "wisesight_sentiment": r.sentiment,
                     "label_source": src, "n_annotators": n, "source": "wisesight_human"})
    out = pd.DataFrame(rows)
    out.to_csv(BASE / "data/category_human_v3.csv", index=False, encoding="utf-8-sig")

    ov = [i for i in items.item_id if i in A and i in B]
    rel = cohen_kappa_score([A[i][1] != SKIP for i in ov], [B[i][1] != SKIP for i in ov])
    both = [i for i in ov if A[i][1] != SKIP and B[i][1] != SKIP]
    agg = {
        "n_items": len(items), "n_double": len(ov), "n_both_relevant": len(both),
        "kappa_relevance": round(rel, 3),
        "kappa_category": round(cohen_kappa_score([A[i][1] for i in both], [B[i][1] for i in both]), 3),
        "kappa_sentiment": round(cohen_kappa_score([A[i][0] for i in both], [B[i][0] for i in both]), 3),
        "relevant_rate_A": round(sum(A[i][1] != SKIP for i in A) / len(A), 3),
        "relevant_rate_B": round(sum(B[i][1] != SKIP for i in B) / len(B), 3),
        "A_saw_ai_hint": sum(v[3] for v in A.values()), "B_saw_ai_hint": sum(v[3] for v in B.values()),
        "label_source_counts": how, "n_final_relevant": len(out),
        "category_counts": out.Category.value_counts().to_dict(),
        "sentiment_agreement_with_wisesight": round(float((out.sentiment == out.wisesight_sentiment).mean()), 3),
    }
    (BASE / "outputs").mkdir(exist_ok=True)
    json.dump(agg, open(BASE / "outputs/train_label_agreement.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(agg, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
