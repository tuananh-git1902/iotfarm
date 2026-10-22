# -*- coding: utf-8 -*-
"""Lập tuyến né vật cản và xác định các pha né (giống logic đồ án, Hình 4.11–4.12).

Tuyến né = tuyến gốc được DỜI NGANG MƯỢT sang làn bên cạnh quanh vật cản:

  độ lệch ngang
      ▲            ┌───────────── giữ làn bên cạnh ─────────────┐
      │           /                                              \
      │          / chuyển làn                          về làn gốc \
  0 ──┼─────────┘                [ VẬT CẢN ]                       └──────── s
               a0     a1     s_near   s_near+L     b0             b1

Các pha (đánh số như đồ án): 0 bình thường → 0.5 tiếp cận → 1 chuyển làn → 2 vượt qua → 3 về làn.
"""
from dataclasses import dataclass

import numpy as np

from .geometry import smoothstep

PH_NORMAL, PH_APPROACH, PH_SHIFT, PH_PASS, PH_RETURN = range(5)
PHASE_NAMES = ("0 bình thường", "0.5 tiếp cận", "1 chuyển làn", "2 vượt qua", "3 về làn")
N_PHASES = 5

# Thứ tự ưu tiên làn né: về làn gốc > làn cùng chiều > làn ngược chiều
KIND_RANK = {"base": 0.0, "left_same": 1.0, "right_same": 1.0, "left_opp": 2.0}


@dataclass
class Bump:
    a0: float          # bắt đầu chuyển làn
    a1: float          # đã sang hẳn làn mới
    b0: float          # bắt đầu về
    b1: float          # đã về hẳn
    target: float      # độ lệch ngang của làn mới so với làn gốc (m)
    kind: str          # base / left_same / right_same / left_opp
    track_id: int


def bump_weight(s, b):
    s = np.asarray(s, dtype=np.float64)
    up = smoothstep((s - b.a0) / max(b.a1 - b.a0, 1e-6))
    down = 1.0 - smoothstep((s - b.b0) / max(b.b1 - b.b0, 1e-6))
    return np.where(s < b.a1, up, np.where(s <= b.b0, 1.0, down))


def compose_offsets(s, bumps):
    """Ghép lần lượt các tuyến né (vật cản sau đè lên vật cản trước)."""
    off = np.zeros(len(s))
    for b in bumps:
        w = bump_weight(s, b)
        off = off * (1.0 - w) + b.target * w
    return off


def plan_bump(path, lane_opts, s_ego, s_near, lat_base, current_offsets, cfg, track_id):
    """Chọn làn né và đoạn chuyển làn cho 1 vật cản. Không có làn nào an toàn → None.

    lane_opts: dict kind → mảng độ lệch tâm làn tại mỗi chỉ số tuyến (NaN nếu không có làn).
    lat_base:  độ lệch ngang của vật cản so với làn gốc (dương = lệch phải).
    """
    a1 = s_near - cfg.lead
    a0 = a1 - cfg.ramp
    b0 = s_near + cfg.obstacle_length + cfg.clearance_after
    b1 = b0 + cfg.ramp
    a0 = max(a0, s_ego + 1.0)                 # phát hiện muộn → chuyển làn gắt hơn
    a1 = max(a1, a0 + cfg.min_ramp)
    b0 = max(b0, a1 + 1.0)
    b1 = max(b1, b0 + cfg.min_ramp)

    check = [path.index_at_s(v) for v in (a0, a1, s_near, b0, b1)]
    i_obs = path.index_at_s(s_near)
    cur = float(current_offsets[i_obs])

    # vật cản lệch phải → ưu tiên né trái và ngược lại; ở giữa làn → vượt bên trái (luật giao thông)
    side_pref = {"left_same": 0.0, "right_same": 0.1} if lat_base >= -0.3 else {"left_same": 0.1, "right_same": 0.0}

    best, best_key = None, None
    for kind, arr in lane_opts.items():
        if kind == "left_opp" and not cfg.allow_opposite_lane:
            continue
        vals = arr[check]
        if not np.all(np.isfinite(vals)):
            continue                          # làn không tồn tại suốt đoạn né
        target = float(arr[i_obs])
        if abs(target - lat_base) < cfg.min_lat_clear:
            continue                          # làn này chứa vật cản
        key = (KIND_RANK[kind] + side_pref.get(kind, 0.0), abs(target - cur))
        if best_key is None or key < best_key:
            best, best_key = (kind, target), key
    if best is None:
        return None
    return Bump(a0, a1, b0, b1, best[1], best[0], track_id)


def phase_at(s_ego, bumps):
    """Pha né hiện tại theo vị trí xe và tuyến né đầu tiên chưa hoàn thành."""
    for b in sorted(bumps, key=lambda x: x.a0):
        if s_ego >= b.b1:
            continue
        if s_ego < b.a0:
            return PH_APPROACH
        if s_ego < b.a1:
            return PH_SHIFT
        if s_ego < b.b0:
            return PH_PASS
        return PH_RETURN
    return PH_NORMAL
