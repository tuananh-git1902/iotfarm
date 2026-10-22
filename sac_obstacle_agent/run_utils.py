# -*- coding: utf-8 -*-
"""Tiện ích dùng chung cho train / evaluate / drive."""
import json
import os

from config import Config


def load_run_config(model_path):
    """Đọc config.json cạnh file model. Lúc chạy agent PHẢI dùng đúng cấu hình observation lúc train."""
    run_dir = os.path.dirname(os.path.abspath(model_path))
    path = os.path.join(run_dir, "config.json")
    if os.path.exists(path):
        print(f"[CONFIG] Dùng cấu hình lúc huấn luyện: {path}")
        return Config.from_json(path)
    print("[CONFIG] Không thấy config.json cạnh model → dùng cấu hình mặc định")
    return Config()


def save_curriculum_state(run_dir, level):
    with open(os.path.join(run_dir, "curriculum_state.json"), "w", encoding="utf-8") as f:
        json.dump({"level": level}, f)


def load_curriculum_state(run_dir):
    path = os.path.join(run_dir, "curriculum_state.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("level")
    return None
