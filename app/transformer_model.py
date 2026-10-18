# -*- coding: utf-8 -*-
"""
transformer_model.py
====================
รันโมเดล WangchanBERTa ที่ fine-tune แล้ว (ONNX int8 จาก 11_finetune_wangchanberta.py)
บน CPU ด้วย onnxruntime — ไม่ต้องติดตั้ง PyTorch บนเซิร์ฟเวอร์

OnnxTextClassifier มี interface แบบเดียวกับโมเดล scikit-learn (predict / predict_proba /
classes_) engine.predict_with_confidence() จึงใช้ได้ทั้งสองแบบโดยไม่ต้องแก้

หาโมเดลจาก (เรียงตามลำดับ):
  1. env WANGCHAN_<TASK>_DIR       โฟลเดอร์ในเครื่อง
  2. env WANGCHAN_<TASK>_REPO      repo บน HuggingFace Hub (ดาวน์โหลดครั้งแรกแล้ว cache)
  3. models/wangchanberta_<task>/   ค่าเริ่มต้น
ถ้าไม่พบ load_transformer_model() คืน None แล้ว engine ใช้ TF-IDF+SVM แทน
"""
import json
import os
import re
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1]

# WangchanBERTa (thai2transformers) แทนช่องว่างด้วย token "<_>" และเป็นโมเดลแบบ uncased
SPACE_TOKEN = "<_>"


def prepare_text(text: str) -> str:
    """เตรียมข้อความก่อนเข้า WangchanBERTa — ใช้ชุดเดียวกันทั้งตอนเทรนและตอนใช้งาน
    รับข้อความที่ผ่าน preprocess.clean_text() แล้ว (หรือข้อความดิบก็ได้)"""
    text = str(text or "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace(" ", SPACE_TOKEN)


class OnnxTextClassifier:
    def __init__(self, model_dir):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
        self.classes_ = np.array(meta["classes"])
        self.temperature = float(meta.get("temperature", 1.0))
        self.max_length = int(meta.get("max_length", 128))
        self.task = meta.get("task", "")
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("ONNX_THREADS", "1"))
        self.session = ort.InferenceSession(str(model_dir / "model.onnx"), sess_options=opts,
                                            providers=["CPUExecutionProvider"])

    def _logits(self, texts, batch_size=32):
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = [prepare_text(t) for t in texts[i:i + batch_size]]
            enc = self.tokenizer(chunk, truncation=True, max_length=self.max_length,
                                 padding=True, return_tensors="np")
            logits = self.session.run(["logits"], {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": enc["attention_mask"].astype(np.int64),
            })[0]
            out.append(logits)
        return np.concatenate(out, axis=0) if out else np.zeros((0, len(self.classes_)))

    def predict_proba(self, texts):
        z = self._logits(list(texts)) / self.temperature
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, texts):
        return self.classes_[self.predict_proba(texts).argmax(axis=1)]


def resolve_model_dir(task: str):
    key = task.upper()
    local = os.environ.get(f"WANGCHAN_{key}_DIR")
    if local and (Path(local) / "model.onnx").exists():
        return Path(local)
    repo = os.environ.get(f"WANGCHAN_{key}_REPO")
    if repo:
        try:
            from huggingface_hub import snapshot_download
            return Path(snapshot_download(repo_id=repo, token=os.environ.get("HF_TOKEN") or None))
        except Exception as e:  # ดาวน์โหลดไม่ได้ -> ใช้โมเดลสำรอง ไม่ให้ API ล่ม
            print(f"[transformer_model] ดาวน์โหลด {repo} ไม่สำเร็จ: {e}")
    default = BASE / "models" / f"wangchanberta_{task}"
    if (default / "model.onnx").exists():
        return default
    return None


def load_transformer_model(task: str):
    """คืน OnnxTextClassifier ของ task (sentiment/category) หรือ None ถ้าไม่มีโมเดล/โหลดไม่ได้"""
    if os.environ.get("DISABLE_TRANSFORMER", "").strip().lower() == "true":
        return None
    model_dir = resolve_model_dir(task)
    if model_dir is None:
        return None
    try:
        return OnnxTextClassifier(model_dir)
    except Exception as e:
        print(f"[transformer_model] โหลดโมเดล {model_dir} ไม่สำเร็จ: {e}")
        return None
