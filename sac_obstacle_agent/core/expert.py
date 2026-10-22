# -*- coding: utf-8 -*-
"""Bộ lái theo luật (pure pursuit + điều tốc P) — KHÔNG học.

Dùng để: (1) kiểm tra môi trường chạy đúng trước khi train,
(2) làm mốc so sánh (baseline) trong báo cáo: SAC phải lái tốt hơn bộ lái luật này.
Chỉ đọc từ observation, giống hệt thông tin agent SAC nhận được.
"""
import math

import numpy as np


def pure_pursuit(obs, lookahead_idx=3, wheelbase=2.875, max_steer_deg=70.0,
                 v_normal=28.0, v_avoid=14.0, k_speed=0.15):
    wps = np.asarray(obs["waypoints"]).reshape(-1, 2) * 50.0      # mét, hệ tọa độ xe
    tx, ty = wps[min(lookahead_idx, len(wps) - 1)]
    ld = max(math.hypot(tx, ty), 1.0)
    alpha = math.atan2(ty, tx)                                       # góc tới điểm mục tiêu (phải = dương)
    delta = math.atan2(2.0 * wheelbase * math.sin(alpha), ld)        # góc bánh xe cần đánh
    steer = float(np.clip(delta / math.radians(max_steer_deg), -1.0, 1.0))

    speed = float(obs["vehicle"][2]) * 50.0
    avoiding = bool(np.any(np.asarray(obs["avoid_phase"])[1:4] > 0.5))   # tiếp cận / chuyển làn / vượt
    near = obs["obstacle"][0] > 0.5 and obs["obstacle"][1] < 0.3 and abs(obs["obstacle"][2]) < 0.4
    v_ref = 0.0 if near else (v_avoid if avoiding else v_normal)          # vật cản sát trước mặt → phanh
    accel = float(np.clip(k_speed * (v_ref - speed), -1.0, 1.0))
    return np.array([steer, accel], dtype=np.float32)
