# -*- coding: utf-8 -*-
"""Mọi thao tác trực tiếp với thế giới CARLA: kết nối, chế độ đồng bộ, tuyến đường, sinh vật cản."""
import math

import carla
import numpy as np
from agents.navigation.global_route_planner import GlobalRoutePlanner

from core.avoidance import plan_bump
from core.geometry import Path

ROLE_EGO, ROLE_OBSTACLE, ROLE_SENSOR = "hero", "sac_obstacle", "sac_sensor"


# ----------------------------------------------------------------- kết nối & cài đặt
def connect(sim):
    client = carla.Client(sim.host, sim.port)
    client.set_timeout(60.0)
    world = client.get_world()
    if sim.town not in world.get_map().name:
        print(f"[CARLA] Đang tải bản đồ {sim.town} …")
        world = client.load_world(sim.town)
    return client, world


def enable_sync(world, dt):
    """Bật chế độ đồng bộ. Trả về cài đặt cũ để khôi phục khi đóng."""
    s = world.get_settings()
    orig = (s.synchronous_mode, s.fixed_delta_seconds)
    s.synchronous_mode = True
    s.fixed_delta_seconds = dt
    world.apply_settings(s)
    return orig


def restore_settings(world, orig):
    s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds = orig
    world.apply_settings(s)


def unload_layers(world, names):
    for name in names:
        layer = getattr(carla.MapLayer, name, None)
        if layer is None:
            continue
        try:
            world.unload_map_layer(layer)
        except RuntimeError:
            pass   # bản đồ không phải *_Opt thì bỏ qua


def destroy_actors(client, actors):
    ids = []
    for a in actors:
        if a is None:
            continue
        try:
            if isinstance(a, carla.Sensor) and a.is_listening:
                a.stop()
            ids.append(a.id)
        except RuntimeError:
            pass
    if ids:
        client.apply_batch_sync([carla.command.DestroyActor(i) for i in ids], False)


def cleanup_leftovers(client, world):
    """Xoá xe/cảm biến còn sót lại từ lần chạy trước bị tắt ngang."""
    left = [a for a in world.get_actors() if a.attributes.get("role_name") in (ROLE_EGO, ROLE_OBSTACLE, ROLE_SENSOR)]
    destroy_actors(client, left)
    if left:
        print(f"[CARLA] Đã dọn {len(left)} actor còn sót lại")


def follow_spectator(world, ego):
    tf = ego.get_transform()
    yaw = math.radians(tf.rotation.yaw)
    loc = tf.location + carla.Location(x=-9 * math.cos(yaw), y=-9 * math.sin(yaw), z=5)
    world.get_spectator().set_transform(carla.Transform(loc, carla.Rotation(pitch=-18, yaw=tf.rotation.yaw)))


# ----------------------------------------------------------------- tuyến đường
class RouteBuilder:
    def __init__(self, cmap, sampling):
        self.map = cmap
        self.grp = GlobalRoutePlanner(cmap, sampling)
        self.spawn_points = cmap.get_spawn_points()

    def build(self, rng, min_len, max_len, tries=80):
        """Chọn ngẫu nhiên cặp spawn point → tuyến dài trong [min_len, max_len] (A* của CARLA)."""
        best = None
        for _ in range(tries):
            a, b = rng.sample(self.spawn_points, 2)
            route = self.grp.trace_route(a.location, b.location)
            route = _dedupe(route)
            if len(route) < 10:
                continue
            xy = np.array([[w.transform.location.x, w.transform.location.y] for w, _ in route])
            length = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
            if min_len <= length <= max_len:
                return a, route
            if best is None or abs(length - min_len) < abs(best[2] - min_len):
                best = (a, route, length)
        return best[0], best[1]


def _dedupe(route, eps=0.1):
    out = []
    for wp, opt in route:
        if out:
            p, q = out[-1][0].transform.location, wp.transform.location
            if math.hypot(p.x - q.x, p.y - q.y) < eps:
                continue
        out.append((wp, opt))
    return out


def smooth_xy(xy, passes=3):
    """Làm mượt nhẹ tuyến: chỗ GlobalRoutePlanner đổi làn đột ngột (nhảy 3,5 m) thành đoạn chuyển ~12 m."""
    out = xy.copy()
    for _ in range(passes):
        out[1:-1] = 0.25 * out[:-2] + 0.5 * out[1:-1] + 0.25 * out[2:]
    return out


def route_to_path(route):
    wps = [w for w, _ in route]
    xy = np.array([[w.transform.location.x, w.transform.location.y] for w in wps])
    return Path(smooth_xy(xy), lane_width=[w.lane_width for w in wps]), wps, [o for _, o in route]


def lane_options(wps):
    """Với mỗi waypoint của tuyến: độ lệch tới tâm làn bên cạnh (NaN nếu không có / trong giao lộ)."""
    n = len(wps)
    opts = {"base": np.zeros(n), "left_same": np.full(n, np.nan),
            "right_same": np.full(n, np.nan), "left_opp": np.full(n, np.nan)}
    for i, w in enumerate(wps):
        if w.is_junction:
            opts["base"][i] = np.nan
            continue
        r = w.get_right_lane()
        if r is not None and r.lane_type == carla.LaneType.Driving and r.lane_id * w.lane_id > 0:
            opts["right_same"][i] = (w.lane_width + r.lane_width) / 2.0
        lft = w.get_left_lane()
        if lft is not None and lft.lane_type == carla.LaneType.Driving:
            key = "left_same" if lft.lane_id * w.lane_id > 0 else "left_opp"
            opts[key][i] = -(w.lane_width + lft.lane_width) / 2.0
    return opts


# ----------------------------------------------------------------- xe & vật cản
def spawn_ego(world, blueprint_id, transform):
    bp = world.get_blueprint_library().find(blueprint_id)
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", ROLE_EGO)
    return world.try_spawn_actor(bp, transform)


def obstacle_slots(path, wps, lane_opts, n, rng, ocfg, acfg, opts=None):
    """Chọn vị trí (quãng đường s) đặt vật cản: xa điểm đầu/cuối, ngoài giao lộ và đoạn đổi làn,
    có làn để né."""
    cand = []
    zeros = np.zeros(len(path))
    for i in range(len(path)):
        s = path.s[i]
        if s < ocfg.start_clearance or s > path.length - ocfg.goal_clearance:
            continue
        j0, j1 = path.index_at_s(s - 20.0), path.index_at_s(s + 25.0)
        if any(w.is_junction for w in wps[j0:j1 + 1]):
            continue
        if opts is not None and any(o.name.startswith("CHANGELANE") for o in opts[j0:j1 + 1]):
            continue
        if plan_bump(path, lane_opts, 0.0, s, 0.0, zeros, acfg, -1) is None:
            continue
        cand.append(i)
    rng.shuffle(cand)
    chosen = []
    for i in cand:
        if len(chosen) >= n:
            break
        if all(abs(path.s[i] - path.s[j]) >= ocfg.min_gap for j in chosen):
            chosen.append(i)
    return sorted(chosen)


def spawn_static_obstacles(world, path, wps, slots, rng, ocfg):
    bps = [b for b in world.get_blueprint_library().filter(ocfg.blueprint_filter)
           if b.has_attribute("number_of_wheels") and int(b.get_attribute("number_of_wheels")) == 4]
    out = []
    for i in slots:
        tf = wps[i].transform
        right = tf.get_right_vector()
        jitter = rng.uniform(-ocfg.lateral_jitter, ocfg.lateral_jitter)
        loc = carla.Location(tf.location.x + right.x * jitter, tf.location.y + right.y * jitter, tf.location.z + 0.3)
        rot = carla.Rotation(yaw=tf.rotation.yaw + rng.uniform(-ocfg.yaw_jitter_deg, ocfg.yaw_jitter_deg))
        bp = rng.choice(bps)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", ROLE_OBSTACLE)
        if bp.has_attribute("color"):
            bp.set_attribute("color", rng.choice(bp.get_attribute("color").recommended_values))
        actor = world.try_spawn_actor(bp, carla.Transform(loc, rot))
        if actor is None:
            continue
        actor.apply_control(carla.VehicleControl(hand_brake=True, brake=1.0))   # đứng yên = vật cản tĩnh
        out.append({"actor": actor, "id": actor.id, "s": float(path.s[i]), "lat": jitter,
                    "xy": np.array([loc.x, loc.y]), "first_det": None, "passed": False})
    return out
