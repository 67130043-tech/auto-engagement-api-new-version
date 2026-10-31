# -*- coding: utf-8 -*-
"""
13_deploy_hf_space.py
=====================
นำ API ขึ้น HuggingFace Spaces (Docker, CPU basic ฟรี: 2 vCPU / 16 GB RAM)

เหตุผล: WangchanBERTa ONNX fp32 ใช้ ~420 MB ต่อโมเดล รวม 2 โมเดลต้องการ RAM > 1 GB
Render แผนฟรี (512 MB) รันไม่ไหว Spaces แผนฟรีมี RAM พอ

ใช้งาน (หลัง login HuggingFace ด้วย Write token แล้ว เช่น notebook_login() ใน Colab):
    python 13_deploy_hf_space.py --user <hf_username>

สิ่งที่สคริปต์ทำ:
  1. สร้าง Space <user>/auto-engagement-api (sdk=docker) ถ้ายังไม่มี
  2. อัปโหลดโค้ด (app/, models/*.joblib, Dockerfile, requirements.txt) — ไม่รวม data/
  3. ตั้ง Variables: WANGCHAN_SENTIMENT_REPO, WANGCHAN_CATEGORY_REPO
คีย์ลับ (OPENAI_API_KEY ฯลฯ) ให้เจ้าของบัญชีตั้งเองที่ Space -> Settings -> Secrets
"""
import argparse
import io
from pathlib import Path

from huggingface_hub import HfApi

BASE = Path(__file__).resolve().parent

SPACE_README = """---
title: Auto Engagement API
emoji: 💬
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

User Behavior-based Auto Engagement API (FastAPI) — โค้ดหลักอยู่ที่
https://github.com/67130043-tech/auto-engagement-api-new-version
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="ชื่อบัญชี HuggingFace")
    ap.add_argument("--space", default="auto-engagement-api")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    space_id = f"{args.user}/{args.space}"
    api.create_repo(space_id, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    api.upload_folder(
        repo_id=space_id, repo_type="space", folder_path=str(BASE),
        allow_patterns=["app/**", "models/*.joblib", "Dockerfile", ".dockerignore", "requirements.txt"],
        ignore_patterns=["**/__pycache__/**", "*.pyc"],
        commit_message="Deploy API",
    )
    api.upload_file(path_or_fileobj=io.BytesIO(SPACE_README.encode("utf-8")), path_in_repo="README.md",
                    repo_id=space_id, repo_type="space", commit_message="Space config")
    for task in ("sentiment", "category"):
        api.add_space_variable(space_id, f"WANGCHAN_{task.upper()}_REPO",
                               f"{args.user}/wangchanberta-autoengage-{task}")

    host = f"{args.user}-{args.space}".lower().replace("_", "-")
    print(f"Space: https://huggingface.co/spaces/{space_id}")
    print(f"API  : https://{host}.hf.space   (Make.com -> https://{host}.hf.space/make/predict)")
    print("build ครั้งแรกใช้เวลา ~5-10 นาที ดูสถานะได้ที่แท็บ Logs ของ Space")


if __name__ == "__main__":
    main()
