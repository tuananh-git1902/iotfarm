# -*- coding: utf-8 -*-
"""Hình học dùng chung: hệ tọa độ, tuyến đường (path), chiếu điểm lên tuyến.

Quy ước tọa độ của CARLA/Unreal: x tiến, y sang PHẢI, z lên; yaw tính bằng độ,
tăng khi quay sang phải. Vector "phải" của hướng ψ là (-sin ψ, cos ψ).
Module chỉ dùng NumPy nên kiểm thử được mà không cần CARLA.
"""
import math

import numpy as np


def wrap_angle(a):
    """Đưa góc (rad) về khoảng [-π, π)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def transform_matrix(x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
    """Ma trận 4×4 từ hệ cục bộ sang hệ thế giới, đúng công thức carla.Transform.get_matrix()."""
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    return np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, x],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, y],
        [sp, -cp * sr, cp * cr, z],
        [0.0, 0.0, 0.0, 1.0],
    ])


def fwd_right(yaw_rad):
    """Vector tiến và vector phải (2D) của hướng yaw (rad). Chấp nhận mảng."""
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.stack([c, s], -1), np.stack([-s, c], -1)


def to_local(points_xy, ego_xy, ego_yaw_rad):
    """Đổi điểm thế giới (N,2) sang hệ xe: cột 0 = phía trước, cột 1 = sang phải."""
    d = np.asarray(points_xy, dtype=np.float64) - np.asarray(ego_xy, dtype=np.float64)
    f, r = fwd_right(ego_yaw_rad)
    return np.stack([d @ f, d @ r], -1)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def path_yaws(xy):
    d = np.diff(xy, axis=0)
    yaw = np.arctan2(d[:, 1], d[:, 0])
    return np.append(yaw, yaw[-1]) if len(yaw) else np.zeros(len(xy))


def cumulative_s(xy):
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


class Path:
    """Tuyến gốc (theo tim làn) + độ lệch ngang offsets[i] → tuyến dự kiến (planned).

    offsets > 0 nghĩa là dời sang phải. Tuyến né vật cản = tuyến gốc dời ngang mượt.
    """

    def __init__(self, base_xy, lane_width=None):
        self.base_xy = np.asarray(base_xy, dtype=np.float64)
        n = len(self.base_xy)
        if n < 3:
            raise ValueError("Tuyến quá ngắn")
        self.base_yaw = path_yaws(self.base_xy)
        self.s = cumulative_s(self.base_xy)
        self.length = float(self.s[-1])
        self.lane_width = np.full(n, 3.5) if lane_width is None else np.asarray(lane_width, dtype=np.float64)
        _, self.base_right = fwd_right(self.base_yaw)
        self.set_offsets(np.zeros(n))

    def __len__(self):
        return len(self.base_xy)

    def set_offsets(self, offsets):
        self.offsets = np.asarray(offsets, dtype=np.float64)
        self.xy = self.base_xy + self.offsets[:, None] * self.base_right
        self.yaw = path_yaws(self.xy)

    def index_at_s(self, s):
        return int(np.clip(np.searchsorted(self.s, s), 0, len(self) - 1))

    def project(self, p, idx, back=2, ahead=25):
        """Chiếu 1 điểm lên tuyến dự kiến trong cửa sổ quanh idx.

        Trả về (k, s, lat, seg_yaw): đoạn k, quãng đường s dọc tuyến gốc,
        độ lệch ngang có dấu (dương = bên phải tuyến), hướng của đoạn.
        """
        i0 = max(0, idx - back)
        i1 = min(len(self) - 1, idx + ahead)
        a, b = self.xy[i0:i1], self.xy[i0 + 1:i1 + 1]
        ab = b - a
        L2 = np.maximum((ab ** 2).sum(1), 1e-9)
        t = np.clip(((np.asarray(p) - a) * ab).sum(1) / L2, 0.0, 1.0)
        proj = a + t[:, None] * ab
        dist = np.linalg.norm(np.asarray(p) - proj, axis=1)
        j = int(np.argmin(dist))
        k = i0 + j
        seg_len = math.sqrt(L2[j])
        right = np.array([-ab[j, 1], ab[j, 0]]) / seg_len
        lat = float((np.asarray(p) - proj[j]) @ right)
        s = float(self.s[k] + t[j] * (self.s[k + 1] - self.s[k]))
        return k, s, lat, math.atan2(ab[j, 1], ab[j, 0])

    def project_many(self, pts, i0, i1, use_base=False):
        """Chiếu nhiều điểm (N,2) lên đoạn tuyến [i0, i1]. Trả về (s, lat) cho từng điểm."""
        xy = self.base_xy if use_base else self.xy
        i0 = int(np.clip(i0, 0, len(self) - 2))
        i1 = int(np.clip(i1, i0 + 1, len(self) - 1))
        a, b = xy[i0:i1], xy[i0 + 1:i1 + 1]                    # (K,2)
        ab = b - a
        L2 = np.maximum((ab ** 2).sum(1), 1e-9)                 # (K,)
        d = pts[:, None, :] - a[None, :, :]                     # (N,K,2)
        t = np.clip((d * ab[None]).sum(2) / L2[None], 0.0, 1.0)  # (N,K)
        proj = a[None] + t[..., None] * ab[None]
        dist = np.linalg.norm(pts[:, None, :] - proj, axis=2)
        j = np.argmin(dist, axis=1)
        n = np.arange(len(pts))
        abj = ab[j]
        right = np.stack([-abj[:, 1], abj[:, 0]], 1) / np.sqrt(L2[j])[:, None]
        lat = ((pts - proj[n, j]) * right).sum(1)
        k = i0 + j
        s = self.s[k] + t[n, j] * (self.s[k + 1] - self.s[k])
        return s, lat
