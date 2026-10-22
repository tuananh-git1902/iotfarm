# -*- coding: utf-8 -*-
"""Hàm phần thưởng (reward) và điều kiện kết thúc episode.

reward mỗi bước = r_tốc_độ × r_giữ_tâm × r_góc           (liên tục, như đồ án)
                + w_progress × (số mét tiến được)           (MỚI: sửa lỗi xe chạy 12–15 km/h)
                − w_steer_rate × |Δ góc lái|                 (MỚI: lái êm)
                + 0.3 khi xong 1 giai đoạn né                (như đồ án)
                − 0.05 khi đang né mà chạy > 15 km/h         (như đồ án)
Sự kiện kết thúc: va chạm −10, lệch tuyến −10, kẹt −5, tới đích +20.
"""
import math
from dataclasses import dataclass

from .avoidance import PH_APPROACH, PH_NORMAL, PH_PASS, PH_RETURN, PH_SHIFT


def speed_reward(v_kmh, v_target, v_max):
    """Tăng tuyến tính tới đỉnh 1 ở v_target, giảm về 0 ở v_max, âm khi vượt v_max.

    Đồ án dùng vùng phẳng [min, target] = 1 nên chạy 15 km/h cũng 'đủ điểm' → xe chạy chậm.
    """
    if v_kmh <= v_target:
        return max(v_kmh, 0.0) / v_target
    if v_kmh <= v_max:
        return 1.0 - (v_kmh - v_target) / (v_max - v_target)
    return max(-1.0, -(v_kmh - v_max) / (v_max - v_target))


@dataclass
class StepMetrics:
    speed_kmh: float
    lat_dev: float            # m, lệch so với tuyến dự kiến (có dấu)
    heading_err: float        # rad
    progress_m: float         # m tiến được trong bước này
    steer: float
    prev_steer: float
    phase: int
    prev_phase: int
    collision: bool
    goal_reached: bool
    stuck: bool
    timeout: bool


def compute_reward(m, cfg):
    """Trả về (reward, terminated, truncated, event, terms)."""
    avoiding = m.phase in (PH_APPROACH, PH_SHIFT, PH_PASS)
    v_target = cfg.avoid_target_speed if avoiding else cfg.target_speed
    r_speed = speed_reward(m.speed_kmh, v_target, cfg.max_speed)
    r_center = max(1.0 - abs(m.lat_dev) / cfg.center_norm, 0.0)
    r_angle = max(1.0 - abs(m.heading_err) / math.radians(cfg.max_angle_deg), 0.0)
    terms = {
        "continuous": r_speed * r_center * r_angle,
        "progress": cfg.w_progress * max(-cfg.progress_clip, min(cfg.progress_clip, m.progress_m)),
        "steer_rate": -cfg.w_steer_rate * abs(m.steer - m.prev_steer),
        "phase_bonus": 0.0,
        "avoid_overspeed": 0.0,
    }
    if m.prev_phase == PH_SHIFT and m.phase == PH_PASS:                  # đồ án: pha 1 → 2
        terms["phase_bonus"] = cfg.phase_bonus
    if m.prev_phase == PH_RETURN and m.phase in (PH_NORMAL, PH_APPROACH):  # đồ án: pha 3 → 0
        terms["phase_bonus"] = cfg.phase_bonus
    if avoiding and m.speed_kmh > cfg.avoid_overspeed_kmh:
        terms["avoid_overspeed"] = -cfg.w_avoid_overspeed
    reward = sum(terms.values())

    terminated, truncated, event = False, False, ""
    if m.collision:
        reward, terminated, event = cfg.r_collision, True, "collision"
    elif abs(m.lat_dev) > cfg.max_center_dev:
        reward, terminated, event = cfg.r_off_route, True, "off_route"
    elif m.goal_reached:
        reward, terminated, event = reward + cfg.r_goal, True, "success"
    elif m.stuck:
        reward, terminated, event = cfg.r_stuck, True, "stuck"
    elif m.timeout:
        truncated, event = True, "timeout"      # hết giờ = cắt ngang, không phải lỗi của trạng thái
    terms["event"] = cfg.r_goal if event == "success" else (reward if terminated else 0.0)
    return float(reward), terminated, truncated, event, terms
