# -*- coding: utf-8 -*-
"""
11_finetune_wangchanberta.py
============================
Fine-tune WangchanBERTa (airesearch/wangchanberta-base-att-spm-uncased) เป็นตัวจำแนก
sentiment หรือ category แล้ว export เป็น ONNX (int8) สำหรับรันบนเซิร์ฟเวอร์ที่ไม่มี GPU

รันบน Google Colab (GPU) ผ่าน 11_finetune_wangchanberta.ipynb หรือรันตรงๆ:
    python 11_finetune_wangchanberta.py --task sentiment
    python 11_finetune_wangchanberta.py --task category
    python 11_finetune_wangchanberta.py --task category --push-to-hub <user>/wangchanberta-category

ใช้ข้อมูลและ split เดียวกับ TF-IDF+SVM (คอลัมน์ split ใน data/*_dataset_v2.csv) ผลจึง
เทียบกันได้ตรงๆ — แบ่ง validation 10% จากชุด train สำหรับ early stopping และการสอบเทียบ

ขั้นตอน:
  1. Fine-tune ด้วย weighted cross-entropy (น้ำหนักกลับตามจำนวนตัวอย่าง = class_weight=
     "balanced" แบบเดียวกับ LinearSVC) เลือก epoch ที่ macro-F1 บน validation ดีที่สุด
  2. สอบเทียบค่าความมั่นใจด้วย Temperature Scaling (Guo et al., 2017) บน validation —
     ทำหน้าที่เดียวกับ Platt scaling ของ CalibratedClassifierCV ฝั่ง TF-IDF
  3. วัดผลบนชุดทดสอบ: accuracy, macro-F1, F1 รายหมวด, ผลแยกตาม source, Brier, ECE
  4. Export ONNX + quantize แบบ dynamic int8 แล้วเช็คว่าผลทำนายตรงกับ PyTorch

ผลลัพธ์ (โฟลเดอร์ --out, ค่าเริ่มต้น models/wangchanberta_<task>/):
  model.onnx          โมเดล int8 สำหรับ app/transformer_model.py
  tokenizer files     จาก tokenizer.save_pretrained()
  labels.json         ลำดับคลาส, temperature, max_length, ชื่อโมเดลต้นทาง
  metrics.json        ผลวัดทั้งหมด (ใช้ใน 12_compare_models.py)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from app.transformer_model import prepare_text  # noqa: E402  (preprocessing ชุดเดียวกับตอนใช้งานจริง)

DEFAULT_MODEL = "airesearch/wangchanberta-base-att-spm-uncased"
TASKS = {
    "sentiment": {"csv": "data/sentiment_dataset_v2.csv", "label": "sentiment", "max_len": 128},
    "category": {"csv": "data/category_dataset_v2.csv", "label": "Category", "max_len": 128},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--model-name", default=DEFAULT_MODEL)
    p.add_argument("--out", default=None)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-len", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="ใช้แค่ N แถวแรกของ train/test (ไว้ทดสอบโค้ด)")
    p.add_argument("--push-to-hub", default=None, help="repo id บน HuggingFace Hub เช่น user/wangchanberta-category")
    return p.parse_args()


# ---------------------------------------------------------------------------
# ตัวชี้วัด
# ---------------------------------------------------------------------------
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def brier_and_ece(y_true_idx, probs, n_bins=10):
    """Brier score และ ECE ของความมั่นใจในคลาสที่ทำนาย (นิยามเดียวกับ 08_calibration_check.py)"""
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (pred == y_true_idx).astype(float)
    brier = float(np.mean((conf - correct) ** 2))
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf >= lo) & ((conf <= hi) if i == n_bins - 1 else (conf < hi))
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return brier, float(ece)


def fit_temperature(logits, y_idx):
    """หา temperature T > 0 ที่ทำให้ NLL บน validation ต่ำสุด (Temperature Scaling)"""
    logits_t = torch.tensor(logits, dtype=torch.float32)
    y_t = torch.tensor(y_idx, dtype=torch.long)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)
    loss_fn = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = loss_fn(logits_t / log_t.exp(), y_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


# ---------------------------------------------------------------------------
# ข้อมูล
# ---------------------------------------------------------------------------
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, enc, labels):
        self.enc, self.labels = enc, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


def load_splits(task, limit, seed):
    cfg = TASKS[task]
    df = pd.read_csv(BASE / cfg["csv"], encoding="utf-8-sig")
    df["label"] = df[cfg["label"]].astype(str)
    if "source" not in df.columns:
        df["source"] = "unknown"
    train_all = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    if limit:
        train_all = train_all.groupby("label", group_keys=False).head(max(2, limit // df["label"].nunique()))
        test = test.groupby("label", group_keys=False).head(max(2, limit // (4 * df["label"].nunique())))
    train, val = train_test_split(train_all, test_size=0.10, random_state=seed, stratify=train_all["label"])
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ---------------------------------------------------------------------------
# หลัก
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)
    set_seed(args.seed)
    cfg = TASKS[args.task]
    max_len = args.max_len or cfg["max_len"]
    out_dir = Path(args.out or BASE / "models" / f"wangchanberta_{args.task}")
    out_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = load_splits(args.task, args.limit, args.seed)
    classes = sorted(train["label"].unique())
    label2id = {c: i for i, c in enumerate(classes)}
    print(f"task={args.task} classes={classes}")
    print(f"train={len(train)} val={len(val)} test={len(test)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def encode(frame):
        texts = [prepare_text(t) for t in frame["clean_text"].astype(str)]
        return tokenizer(texts, truncation=True, max_length=max_len)

    ds_train = TextDataset(encode(train), [label2id[c] for c in train["label"]])
    ds_val = TextDataset(encode(val), [label2id[c] for c in val["label"]])
    ds_test = TextDataset(encode(test), [label2id[c] for c in test["label"]])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(classes),
        id2label={i: c for c, i in label2id.items()}, label2id=label2id,
        ignore_mismatched_sizes=True)  # กันกรณี checkpoint ต้นทางมีหัวจำแนกจำนวนคลาสอื่นอยู่แล้ว

    # น้ำหนักคลาสแบบ "balanced": n_samples / (n_classes * n_class)
    counts = np.bincount(ds_train.labels, minlength=len(classes))
    class_weights = torch.tensor(len(ds_train) / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits, labels, weight=class_weights.to(outputs.logits.device))
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        pred = logits.argmax(axis=1)
        return {"accuracy": accuracy_score(labels, pred),
                "macro_f1": f1_score(labels, pred, average="macro", zero_division=0)}

    targs = TrainingArguments(
        output_dir=str(out_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to=[],
        seed=args.seed,
        logging_steps=50,
    )
    trainer = WeightedTrainer(model=model, args=targs, train_dataset=ds_train, eval_dataset=ds_val,
                              data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute_metrics)
    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0

    # --- สอบเทียบ + วัดผล ---
    val_logits = trainer.predict(ds_val).predictions
    temperature = fit_temperature(val_logits, np.array(ds_val.labels))
    test_logits = trainer.predict(ds_test).predictions
    y_test = np.array(ds_test.labels)
    probs_raw = softmax(test_logits)
    probs_cal = softmax(test_logits / temperature)
    pred = probs_cal.argmax(axis=1)
    pred_labels = [classes[i] for i in pred]
    brier_raw, ece_raw = brier_and_ece(y_test, probs_raw)
    brier_cal, ece_cal = brier_and_ece(y_test, probs_cal)

    by_source = {}
    for source in sorted(test["source"].unique()):
        m = (test["source"] == source).values
        by_source[source] = {
            "n_test": int(m.sum()),
            "accuracy": float(accuracy_score(y_test[m], pred[m])),
            "macro_f1": float(f1_score(y_test[m], pred[m], average="macro", zero_division=0)),
        }

    metrics = {
        "task": args.task,
        "base_model": args.model_name,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size, "max_length": max_len,
        "train_seconds": round(train_seconds, 1),
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "report": classification_report(test["label"], pred_labels, output_dict=True, zero_division=0),
        "by_source": by_source,
        "temperature": temperature,
        "calibration": {"uncalibrated": {"brier": brier_raw, "ece": ece_raw},
                        "temperature_scaled": {"brier": brier_cal, "ece": ece_cal}},
    }
    print(json.dumps({k: metrics[k] for k in ("accuracy", "macro_f1", "temperature", "calibration", "by_source")},
                     ensure_ascii=False, indent=2))

    # --- export ONNX (int8) ---
    export_onnx(trainer.model, tokenizer, out_dir, max_len)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "labels.json").write_text(json.dumps({
        "classes": classes, "temperature": temperature, "max_length": max_len,
        "base_model": args.model_name, "task": args.task,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- เช็คว่า ONNX ให้ผลตรงกับ PyTorch ---
    from app.transformer_model import OnnxTextClassifier
    onnx_model = OnnxTextClassifier(out_dir)
    sample = test["clean_text"].astype(str).tolist()[:200]
    onnx_pred = onnx_model.predict(sample)
    agree = float(np.mean([a == b for a, b in zip(onnx_pred, pred_labels[:len(sample)])]))
    print(f"ONNX int8 vs PyTorch agreement on {len(sample)} test rows: {agree:.3f}")
    metrics["onnx_int8_agreement"] = agree
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    import shutil
    shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)
    if args.push_to_hub:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push_to_hub, exist_ok=True)
        api.upload_folder(folder_path=str(out_dir), repo_id=args.push_to_hub)
        print(f"อัปโหลดไปที่ https://huggingface.co/{args.push_to_hub}")
    print(f"บันทึกไว้ที่: {out_dir}")


def export_onnx(model, tokenizer, out_dir: Path, max_len: int):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    model = model.to("cpu").eval()
    dummy = tokenizer(["ทดสอบ"], return_tensors="pt", padding="max_length", max_length=16, truncation=True)
    fp32_path = out_dir / "model_fp32.onnx"
    torch.onnx.export(
        model, (dummy["input_ids"], dummy["attention_mask"]), str(fp32_path),
        input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                      "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    quantize_dynamic(str(fp32_path), str(out_dir / "model.onnx"), weight_type=QuantType.QInt8)
    fp32_path.unlink()


if __name__ == "__main__":
    main()
