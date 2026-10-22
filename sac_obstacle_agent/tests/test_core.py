# -*- coding: utf-8 -*-
"""Kiểm thử các module lõi KHÔNG cần CARLA (chỉ cần NumPy).

Chạy:  python tests/test_core.py      (hoặc: python -m pytest tests)
Có một "máy dựng ảnh" nhỏ bắn tia để tạo ảnh semantic + depth giả, giống camera CARLA.
"""
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config                                            # noqa: E402
from core.avoidance import (PH_APPROACH, PH_NORMAL, PH_PASS, PH_RETURN,  # noqa: E402
                            PH_SHIFT, compose_offsets, phase_at, plan_bump)
from core.curriculum import AdaptiveCurriculum, FailureReplay, ScenarioScheduler  # noqa: E402
from core.geometry import Path, to_local, transform_matrix, wrap_angle  # noqa: E402
from core.perception import (ObstacleTracker, backproject, cam_to_world,  # noqa: E402
                             depth_from_bgra, detect_obstacle, intrinsics, project,
                             semantic_grid)
from core.reward import StepMetrics, compute_reward, speed_reward    # noqa: E402

CFG = Config()


# ---------------------------------------------------------------- tiện ích dựng cảnh giả
def render(cam_M, boxes, w=160, h=80, fov=90.0):
    """Bắn tia từ camera; trả về (tags, depth) giống camera semantic + depth của CARLA."""
    f, cx, cy = intrinsics(w, h, fov)
    vs, us = np.mgrid[0:h, 0:w]
    d_cam = np.stack([np.ones(h * w), (us.ravel() + 0.5 - cx) / f, -(vs.ravel() + 0.5 - cy) / f], 1)
    R, o = cam_M[:3, :3], cam_M[:3, 3]
    D = d_cam @ R.T
    depth = np.full(h * w, 1000.0)
    tags = np.full(h * w, 11, dtype=np.uint8)                       # trời
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = np.where(D[:, 2] < 0, -o[2] / D[:, 2], np.inf)          # mặt đường z = 0
    hit = np.isfinite(tg) & (tg < depth)
    depth[hit], tags[hit] = tg[hit], 1
    for lo, hi in boxes:
        lo, hi = np.array(lo, float), np.array(hi, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - o) / D, (hi - o) / D
        tmin = np.nanmax(np.minimum(t1, t2), 1)
        tmax = np.nanmin(np.maximum(t1, t2), 1)
        hit = (tmax >= np.maximum(tmin, 0)) & (tmin > 0) & (tmin < depth)
        depth[hit], tags[hit] = tmin[hit], 14                       # ô tô
    return tags.reshape(h, w), depth.reshape(h, w).astype(np.float32)


def straight_path(length=200.0, step=2.0):
    xs = np.arange(0.0, length + step, step)
    return Path(np.stack([xs, np.zeros_like(xs)], 1))


def curved_path(R=30.0):
    pts = [(s, 0.0) for s in np.arange(0, 5, 2.0)]
    for s in np.arange(5, 80, 2.0):
        th = (s - 5) / R
        pts.append((5 + R * math.sin(th), R * (1 - math.cos(th))))   # rẽ phải (y dương)
    return Path(np.array(pts))


CAM_M = transform_matrix(1.5, 0.0, 1.7, pitch=-8.0)                  # xe ở gốc tọa độ, hướng +x
INTR = intrinsics(160, 80, 90.0)


# ---------------------------------------------------------------- hình học
def test_wrap_and_local():
    assert abs(wrap_angle(3 * math.pi) - (-math.pi)) < 1e-9
    loc = to_local([[10.0, 0.0], [0.0, 2.0]], [0.0, 0.0], 0.0)
    assert np.allclose(loc, [[10, 0], [0, 2]])                       # y dương = bên phải
    loc = to_local([[0.0, 10.0]], [0.0, 0.0], math.pi / 2)           # xe quay 90° sang phải
    assert np.allclose(loc, [[10, 0]], atol=1e-9)


def test_path_projection_sign():
    p = straight_path()
    k, s, lat, yaw = p.project(np.array([10.3, 1.0]), 0)
    assert abs(s - 10.3) < 1e-6 and abs(lat - 1.0) < 1e-6 and abs(yaw) < 1e-9
    s2, lat2 = p.project_many(np.array([[20.0, -2.0], [30.0, 0.5]]), 0, 40)
    assert np.allclose(s2, [20, 30]) and np.allclose(lat2, [-2.0, 0.5])


def test_camera_roundtrip():
    pts = np.array([[10.0, 1.0, -1.0], [25.0, -3.0, 0.5]])
    uv = project(pts, *INTR)
    back = backproject(uv[:, 0], uv[:, 1], pts[:, 0], *INTR)
    assert np.allclose(back, pts)
    w = cam_to_world(np.array([[1.0, 0.0, 0.0]]), CAM_M)             # 1 m trước camera, chúc xuống 8°
    assert abs(w[0, 2] - (1.7 - math.sin(math.radians(8)))) < 1e-9


def test_depth_decode():
    bgra = np.zeros((1, 1, 4), np.uint8)
    bgra[0, 0, :3] = (0, 39, 16)                                     # B, G, R → 16 + 39·256 = 10000
    assert abs(depth_from_bgra(bgra)[0, 0] - 10000 / 16777215 * 1000) < 1e-6


# ---------------------------------------------------------------- nhận diện vật cản
def test_detect_obstacle_ahead():
    car = ((20.0, -0.9, 0.0), (24.5, 0.9, 1.5))
    tags, depth = render(CAM_M, [car])
    det = detect_obstacle(tags, depth, CAM_M, INTR, straight_path(), 0, 0.0, 0.0, CFG.detection)
    assert det is not None, "phải thấy xe nằm giữa làn ở 20 m"
    assert abs(det.dist - 20.0) < 1.0 and abs(det.lat_planned) < 0.6, det


def test_ignore_car_in_other_lane():
    car = ((18.0, -4.4, 0.0), (22.5, -2.6, 1.5))                     # làn bên trái
    tags, depth = render(CAM_M, [car])
    assert (tags == 14).sum() > 20                                   # có thấy xe trong ảnh …
    det = detect_obstacle(tags, depth, CAM_M, INTR, straight_path(), 0, 0.0, 0.0, CFG.detection)
    assert det is None                                               # … nhưng không nằm trên tuyến


def test_detect_on_curve_and_ignore_straight_ahead():
    path = curved_path()
    on_curve = ((22.0, 5.0, 0.0), (25.0, 8.0, 1.5))                  # nằm trên khúc cua
    straight = ((25.0, -1.0, 0.0), (29.0, 1.0, 1.5))                 # thẳng phía trước, ngoài tuyến
    tags, depth = render(CAM_M, [straight])
    assert detect_obstacle(tags, depth, CAM_M, INTR, path, 0, 0.0, 0.0, CFG.detection) is None
    tags, depth = render(CAM_M, [on_curve])
    det = detect_obstacle(tags, depth, CAM_M, INTR, path, 0, 0.0, 0.0, CFG.detection)
    assert det is not None and 18 < det.dist < 30, det


def test_out_of_range():
    car = ((45.0, -0.9, 0.0), (49.5, 0.9, 1.5))
    tags, depth = render(CAM_M, [car])
    assert detect_obstacle(tags, depth, CAM_M, INTR, straight_path(), 0, 0.0, 0.0, CFG.detection) is None


def test_tracker_static():
    tr = ObstacleTracker(3.5, 1.5)
    tags, depth = render(CAM_M, [((20.0, -0.9, 0.0), (24.5, 0.9, 1.5))])
    det = detect_obstacle(tags, depth, CAM_M, INTR, straight_path(), 0, 0.0, 0.0, CFG.detection)
    t, new = tr.update(det, 0.0, step=0, dt=0.05)
    assert new
    for k in range(1, 4):
        t, new = tr.update(det, 0.0, step=k, dt=0.05)
        assert not new
    assert t.is_static and t.hits == 4


def test_semantic_grid_shape():
    tags, _ = render(CAM_M, [((20.0, -0.9, 0.0), (24.5, 0.9, 1.5))])
    g = semantic_grid(tags)
    assert g.shape == (256,) and g[:128].max() > 0.5 and g[128:].max() > 0


# ---------------------------------------------------------------- lập tuyến né
def lane_opts(n, left=-3.5, right=np.nan, opp=np.nan):
    return {"base": np.zeros(n), "left_same": np.full(n, left),
            "right_same": np.full(n, right), "left_opp": np.full(n, opp)}


def test_bump_and_phases():
    p = straight_path()
    b = plan_bump(p, lane_opts(len(p)), 0.0, 60.0, 0.0, p.offsets, CFG.avoid, 0)
    assert b.kind == "left_same" and b.target == -3.5
    assert (b.a0, b.a1, b.b0, b.b1) == (42.0, 54.0, 71.0, 83.0)
    off = compose_offsets(p.s, [b])
    assert off[p.index_at_s(40)] == 0 and off[p.index_at_s(60)] == -3.5 and off[p.index_at_s(90)] == 0
    seq = [phase_at(s, [b]) for s in (30, 45, 60, 75, 90)]
    assert seq == [PH_APPROACH, PH_SHIFT, PH_PASS, PH_RETURN, PH_NORMAL]


def test_lane_choice():
    p = straight_path()
    both = lane_opts(len(p), right=3.5)
    assert plan_bump(p, both, 0, 60, +0.4, p.offsets, CFG.avoid, 0).kind == "left_same"
    assert plan_bump(p, both, 0, 60, -0.5, p.offsets, CFG.avoid, 0).kind == "right_same"
    none = lane_opts(len(p), left=np.nan)
    assert plan_bump(p, none, 0, 60, 0.0, p.offsets, CFG.avoid, 0) is None
    opp = lane_opts(len(p), left=np.nan, opp=-3.5)
    assert plan_bump(p, opp, 0, 60, 0.0, p.offsets, CFG.avoid, 0).kind == "left_opp"


def test_second_obstacle_in_side_lane_returns_to_base():
    p = straight_path()
    opts = lane_opts(len(p))
    b1 = plan_bump(p, opts, 0, 60, 0.0, p.offsets, CFG.avoid, 0)
    p.set_offsets(compose_offsets(p.s, [b1]))
    b2 = plan_bump(p, opts, 50, 65, -3.5, p.offsets, CFG.avoid, 1)   # vật cản thứ 2 ở làn trái
    assert b2.kind == "base" and b2.target == 0.0


def test_late_detection_still_plans():
    p = straight_path()
    b = plan_bump(p, lane_opts(len(p)), 50.0, 55.0, 0.0, p.offsets, CFG.avoid, 0)
    assert b.a0 == 51.0 and b.a1 == 56.0


# ---------------------------------------------------------------- phần thưởng
def metrics(**kw):
    base = dict(speed_kmh=30, lat_dev=0, heading_err=0, progress_m=30 / 3.6 * 0.05, steer=0,
                prev_steer=0, phase=PH_NORMAL, prev_phase=PH_NORMAL, collision=False,
                goal_reached=False, stuck=False, timeout=False)
    base.update(kw)
    return StepMetrics(**base)


def test_speed_reward_shape():
    rc = CFG.reward
    assert speed_reward(30, 30, 40) == 1.0 and speed_reward(15, 30, 40) == 0.5
    assert speed_reward(40, 30, 40) == 0.0 and speed_reward(50, 30, 40) < 0
    r30 = compute_reward(metrics(), rc)[0]
    r15 = compute_reward(metrics(speed_kmh=15, progress_m=15 / 3.6 * 0.05), rc)[0]
    assert r30 > r15, "chạy 30 km/h phải được thưởng nhiều hơn 15 km/h (lỗi của đồ án gốc)"


def test_events():
    rc = CFG.reward
    r, term, trunc, ev, _ = compute_reward(metrics(collision=True), rc)
    assert (r, term, ev) == (-10.0, True, "collision")
    r, term, trunc, ev, _ = compute_reward(metrics(lat_dev=3.5), rc)
    assert ev == "off_route" and term
    r, term, trunc, ev, _ = compute_reward(metrics(goal_reached=True), rc)
    assert ev == "success" and term and r > 20
    r, term, trunc, ev, _ = compute_reward(metrics(timeout=True), rc)
    assert ev == "timeout" and trunc and not term


def test_phase_bonus_and_overspeed():
    rc = CFG.reward
    base = compute_reward(metrics(phase=PH_PASS, prev_phase=PH_PASS, speed_kmh=15, progress_m=0.2), rc)[0]
    bonus = compute_reward(metrics(phase=PH_PASS, prev_phase=PH_SHIFT, speed_kmh=15, progress_m=0.2), rc)[0]
    assert abs(bonus - base - 0.3) < 1e-9
    slow = compute_reward(metrics(phase=PH_APPROACH, prev_phase=PH_APPROACH, speed_kmh=15), rc)[4]
    fast = compute_reward(metrics(phase=PH_APPROACH, prev_phase=PH_APPROACH, speed_kmh=25), rc)[4]
    assert slow["avoid_overspeed"] == 0.0 and fast["avoid_overspeed"] == -0.05
    ret = compute_reward(metrics(phase=PH_NORMAL, prev_phase=PH_RETURN), rc)[4]
    assert ret["phase_bonus"] == 0.3


# ---------------------------------------------------------------- curriculum
def test_curriculum_promotes():
    c = AdaptiveCurriculum((0, 1, 2), window=10, promote_success=0.8)
    for ep in range(10):
        c.report(True, 0, ep)
    assert c.n_obstacles == 1


def test_failure_replay_and_scheduler():
    fr = FailureReplay(capacity=5, p_replay=1.0, max_replays=2)
    fr.add(123, 2)
    rng = random.Random(0)
    assert fr.sample(rng) == (123, 2) and fr.sample(rng) == (123, 2) and fr.sample(rng) is None
    sch = ScenarioScheduler(CFG.curriculum, seed=1)
    seed, n, src = sch.next()
    assert n == 0 and src in ("new", "replay")
    sch.set_eval([(7, 1), (8, 2)])
    assert sch.next()[:2] == (7, 1) and sch.next()[:2] == (8, 2)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ĐẠT   {name}")
        except AssertionError as e:
            failed += 1
            print(f"  LỖI   {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} bài kiểm thử đạt")
    sys.exit(1 if failed else 0)
