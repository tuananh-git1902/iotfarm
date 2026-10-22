# -*- coding: utf-8 -*-
"""Nhận diện vật cản tĩnh từ camera semantic + camera depth.

Cải tiến so với đồ án gốc (đếm >8% pixel "xe" trên cả ảnh rồi chia đôi trái/phải):
  1. Mỗi pixel "xe" được chiếu ngược ra không gian 3D nhờ ảnh depth.
  2. Chỉ giữ điểm nằm trong HÀNH LANG của tuyến dự kiến (bề rộng 1 làn, dọc theo tuyến),
     nên vật cản sau khúc cua vẫn được bắt, còn xe ở làn khác/đỗ ven đường thì bỏ qua.
  3. Trả về khoảng cách thật (m) dọc tuyến và độ lệch ngang của vật cản.
  4. Theo dõi vật cản qua nhiều khung hình để biết nó đứng yên (tĩnh) hay đang chạy.
Chỉ dùng NumPy → kiểm thử được không cần CARLA.
"""
import math
from dataclasses import dataclass, field

import numpy as np

# Bảng màu CityScapes theo nhãn semantic của CARLA >= 0.9.14 (nhãn 0..28)
CITYSCAPES = np.array([
    (0, 0, 0), (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156),
    (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35),
    (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142),
    (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
    (110, 190, 160), (170, 120, 50), (55, 90, 80), (45, 60, 150), (157, 234, 50),
    (81, 0, 81), (150, 100, 100), (230, 150, 140), (180, 165, 180),
], dtype=np.uint8)
ROAD_TAGS = (1, 24)        # Roads, RoadLine


# ----------------------------------------------------------------- giải mã ảnh CARLA
def bgra_array(raw_data, height, width):
    return np.frombuffer(raw_data, dtype=np.uint8).reshape(height, width, 4)


def tags_from_bgra(bgra):
    """Camera semantic: nhãn nằm ở kênh R (chỉ số 2 trong BGRA)."""
    return bgra[:, :, 2].copy()


def depth_from_bgra(bgra):
    """Camera depth: (R + G·256 + B·256²) / (256³ − 1) × 1000 m."""
    b = bgra[:, :, 0].astype(np.float32)
    g = bgra[:, :, 1].astype(np.float32)
    r = bgra[:, :, 2].astype(np.float32)
    return (r + g * 256.0 + b * 65536.0) / 16777215.0 * 1000.0


def tags_to_rgb(tags):
    return CITYSCAPES[np.clip(tags, 0, len(CITYSCAPES) - 1)]


def semantic_grid(tags, rows=8, cols=16, vehicle_tags=(14, 15, 16, 18, 19)):
    """Lưới thô 2 kênh (tỉ lệ pixel 'đường' và 'xe' trong từng ô) → vector 2·rows·cols.

    Giữ được thông tin VỊ TRÍ (thứ mà average-pooling của ResNet làm mất).
    """
    h, w = tags.shape
    ch, cw = h // rows, w // cols
    t = tags[:rows * ch, :cols * cw]
    road = np.isin(t, ROAD_TAGS).reshape(rows, ch, cols, cw).mean((1, 3))
    veh = np.isin(t, vehicle_tags).reshape(rows, ch, cols, cw).mean((1, 3))
    return np.concatenate([road.ravel(), veh.ravel()]).astype(np.float32)


# ----------------------------------------------------------------- hình học camera
def intrinsics(width, height, fov_deg):
    f = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    return f, width / 2.0, height / 2.0


def backproject(us, vs, depth, f, cx, cy):
    """Pixel (u, v) + độ sâu → điểm trong hệ camera CARLA (x tiến, y phải, z lên)."""
    x = depth
    y = (us - cx) * depth / f
    z = -(vs - cy) * depth / f
    return np.stack([x, y, z], 1)


def project(points_cam, f, cx, cy):
    """Ngược lại: điểm hệ camera → pixel (u, v). Dùng cho kiểm thử và vẽ debug."""
    x = points_cam[:, 0]
    return np.stack([f * points_cam[:, 1] / x + cx, -f * points_cam[:, 2] / x + cy], 1)


def cam_to_world(points_cam, cam_matrix):
    hom = np.concatenate([points_cam, np.ones((len(points_cam), 1))], 1)
    return (np.asarray(cam_matrix) @ hom.T).T[:, :3]


# ----------------------------------------------------------------- phát hiện
@dataclass
class Detection:
    xy: np.ndarray            # vị trí thế giới (điểm gần nhất của vật cản)
    s_near: float             # quãng đường dọc tuyến gốc tới mép gần của vật cản
    dist: float               # khoảng cách từ xe tới vật cản dọc tuyến (m)
    lat_planned: float        # lệch ngang so với tâm tuyến dự kiến (m, dương = bên phải)
    n_pixels: int


def detect_obstacle(tags, depth, cam_matrix, intr, path, idx, s_ego, ground_z, cfg):
    """Tìm vật cản GẦN NHẤT nằm trong hành lang tuyến dự kiến. Không có → None."""
    mask = np.isin(tags, cfg.vehicle_tags) & (depth > 0.5) & (depth < cfg.detect_range)
    if int(mask.sum()) < cfg.min_pixels:
        return None
    vs, us = np.nonzero(mask)
    f, cx, cy = intr
    pts_w = cam_to_world(backproject(us.astype(np.float64), vs.astype(np.float64),
                                     depth[vs, us].astype(np.float64), f, cx, cy), cam_matrix)
    keep = (pts_w[:, 2] - ground_z) < cfg.max_height
    pts_w = pts_w[keep]
    if len(pts_w) < cfg.min_pixels:
        return None

    i1 = path.index_at_s(s_ego + cfg.lookahead)
    s_pts, lat_pts = path.project_many(pts_w[:, :2], idx, i1)
    hw = path.lane_width[np.clip(np.searchsorted(path.s, s_pts), 0, len(path) - 1)] / 2.0 + cfg.corridor_margin
    inside = (np.abs(lat_pts) <= hw) & (s_pts > s_ego) & (s_pts < s_ego + cfg.lookahead)
    if int(inside.sum()) < cfg.min_pixels:
        return None

    s_in, lat_in, xy_in = s_pts[inside], lat_pts[inside], pts_w[inside, :2]
    s_near = float(np.percentile(s_in, 5))
    cluster = s_in < s_near + 4.0                       # chỉ lấy cụm gần nhất
    return Detection(xy=np.median(xy_in[cluster], axis=0), s_near=s_near, dist=s_near - s_ego,
                     lat_planned=float(np.median(lat_in[cluster])), n_pixels=int(inside.sum()))


# ----------------------------------------------------------------- theo dõi qua nhiều khung
@dataclass
class Track:
    tid: int
    xy: np.ndarray
    s_near: float
    lat_base: float
    first_step: int
    last_step: int
    first_dist: float          # khoảng cách lúc phát hiện lần đầu (để đánh giá nhận diện)
    hits: int = 1
    speed: float = 0.0
    bump: object = None        # tuyến né đã lập cho vật cản này

    @property
    def is_static(self):
        return self.hits >= 3 and self.speed < self._tol

    _tol: float = field(default=1.5, repr=False)


class ObstacleTracker:
    """Ghi nhớ vật cản đã thấy. Nhờ đó xe 'biết' vật cản vẫn ở đó khi đã ra khỏi khung hình."""

    def __init__(self, assoc_radius=3.5, static_speed_tol=1.5):
        self.assoc_radius = assoc_radius
        self.tol = static_speed_tol
        self.tracks = []
        self._next = 0

    def reset(self):
        self.tracks, self._next = [], 0

    def update(self, det, lat_base, step, dt):
        """Gộp phát hiện mới vào vật cản đã biết, hoặc tạo vật cản mới. Trả về (track, is_new)."""
        for tr in self.tracks:
            if np.linalg.norm(tr.xy - det.xy) < self.assoc_radius:
                gap = max(step - tr.last_step, 1) * dt
                v = float(np.linalg.norm(det.xy - tr.xy) / gap)
                tr.speed = 0.7 * tr.speed + 0.3 * v if tr.hits > 1 else v
                tr.xy = 0.5 * tr.xy + 0.5 * det.xy
                tr.s_near = min(tr.s_near, det.s_near) if tr.bump is not None else det.s_near
                tr.lat_base = 0.5 * tr.lat_base + 0.5 * lat_base
                tr.hits += 1
                tr.last_step = step
                return tr, False
        tr = Track(tid=self._next, xy=det.xy.copy(), s_near=det.s_near, lat_base=lat_base,
                   first_step=step, last_step=step, first_dist=det.dist, _tol=self.tol)
        self._next += 1
        self.tracks.append(tr)
        return tr, True

    def nearest_ahead(self, s_ego, obstacle_length):
        """Vật cản gần nhất chưa vượt qua (dùng cho observation)."""
        ahead = [t for t in self.tracks if t.s_near + obstacle_length + 1.0 > s_ego]
        return min(ahead, key=lambda t: t.s_near) if ahead else None
