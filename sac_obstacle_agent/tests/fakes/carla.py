# -*- coding: utf-8 -*-
"""CARLA GIẢ — chỉ dùng cho kiểm thử khi không có CARLA thật.

Mô phỏng: một đường thẳng dọc trục x có 2 làn cùng chiều + 1 làn ngược chiều, xe theo mô hình
xe đạp (kinematic bicycle), camera semantic/depth dựng bằng bắn tia, cảm biến va chạm.
Chỉ cài đúng những hàm mà code của dự án gọi tới.
"""
import fnmatch
import math
import types

import numpy as np

LANE_W = 3.5
LANES = {-1: 0.0, -2: -3.5, 1: -7.0}       # lane_id → y tâm làn (âm = bên trái hướng +x)
ROAD_LEN = 700.0


# ----------------------------------------------------------------- kiểu dữ liệu cơ bản
class Vector3D:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    def length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)


class Location(Vector3D):
    def __add__(self, o):
        return Location(self.x + o.x, self.y + o.y, self.z + o.z)

    def distance(self, o):
        return math.sqrt((self.x - o.x) ** 2 + (self.y - o.y) ** 2 + (self.z - o.z) ** 2)


class Rotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch, self.yaw, self.roll = float(pitch), float(yaw), float(roll)


def _matrix(loc, rot):
    cy, sy = math.cos(math.radians(rot.yaw)), math.sin(math.radians(rot.yaw))
    cr, sr = math.cos(math.radians(rot.roll)), math.sin(math.radians(rot.roll))
    cp, sp = math.cos(math.radians(rot.pitch)), math.sin(math.radians(rot.pitch))
    return np.array([
        [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, loc.x],
        [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, loc.y],
        [sp, -cp * sr, cp * cr, loc.z],
        [0, 0, 0, 1.0]])


class Transform:
    def __init__(self, location=None, rotation=None, matrix=None):
        self.location = location or Location()
        self.rotation = rotation or Rotation()
        self._m = matrix

    def get_matrix(self):
        m = self._m if self._m is not None else _matrix(self.location, self.rotation)
        return m.tolist()

    def get_right_vector(self):
        y = math.radians(self.rotation.yaw)
        return Vector3D(-math.sin(y), math.cos(y), 0)

    def get_forward_vector(self):
        y = math.radians(self.rotation.yaw)
        return Vector3D(math.cos(y), math.sin(y), 0)


class Color:
    def __init__(self, r=0, g=0, b=0, a=255):
        self.r, self.g, self.b = r, g, b


class VehicleControl:
    def __init__(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False):
        self.throttle, self.steer, self.brake, self.hand_brake = throttle, steer, brake, hand_brake


LaneType = types.SimpleNamespace(Driving="Driving", Sidewalk="Sidewalk")
MapLayer = types.SimpleNamespace(ParkedVehicles=1, Particles=2, Foliage=3)
AttachmentType = types.SimpleNamespace(Rigid=0, SpringArm=1)
command = types.SimpleNamespace(DestroyActor=lambda i: ("destroy", i))


# ----------------------------------------------------------------- bản đồ
class Waypoint:
    def __init__(self, x, lane_id):
        self.lane_id = lane_id
        self.lane_width = LANE_W
        self.lane_type = LaneType.Driving
        self.is_junction = False
        yaw = 0.0 if lane_id < 0 else 180.0
        self.transform = Transform(Location(x, LANES[lane_id], 0.0), Rotation(yaw=yaw))

    def get_left_lane(self):
        order = {-1: -2, -2: 1}
        nxt = order.get(self.lane_id)
        return Waypoint(self.transform.location.x, nxt) if nxt is not None else None

    def get_right_lane(self):
        order = {-2: -1}
        nxt = order.get(self.lane_id)
        return Waypoint(self.transform.location.x, nxt) if nxt is not None else None


class Map:
    name = "Carla/Maps/Town10HD_Opt"

    def get_spawn_points(self):
        return [Transform(Location(x, 0.0, 0.3), Rotation(yaw=0.0)) for x in (10.0, 30.0, 220.0, 340.0, 480.0)]


class _Opt:
    name = "LANEFOLLOW"


class GlobalRoutePlanner:
    def __init__(self, cmap, sampling):
        self.step = sampling

    def trace_route(self, a, b):
        if b.x <= a.x:
            return []
        xs = np.arange(a.x, b.x, self.step)
        return [(Waypoint(float(x), -1), _Opt()) for x in xs]


# ----------------------------------------------------------------- blueprint & actor
class _Attr:
    def __init__(self, v, rec=None):
        self.v = v
        self.recommended_values = rec or [v]

    def __int__(self):
        return int(self.v)

    def __str__(self):
        return str(self.v)


class Blueprint:
    def __init__(self, bid, attrs):
        self.id = bid
        self.attrs = dict(attrs)

    def has_attribute(self, k):
        return k in self.attrs

    def set_attribute(self, k, v):
        self.attrs[k] = v

    def get_attribute(self, k):
        v = self.attrs[k]
        return v if isinstance(v, _Attr) else _Attr(v)


class BlueprintLibrary:
    def __init__(self):
        car = {"number_of_wheels": "4", "role_name": "", "color": _Attr("255,0,0", ["255,0,0", "0,0,255"])}
        self.bps = [Blueprint(i, car) for i in ("vehicle.tesla.model3", "vehicle.audi.a2", "vehicle.lincoln.mkz_2020")]
        cam = {"image_size_x": "800", "image_size_y": "600", "fov": "90", "role_name": ""}
        self.bps += [Blueprint("sensor.camera.semantic_segmentation", cam), Blueprint("sensor.camera.depth", cam),
                     Blueprint("sensor.camera.rgb", cam), Blueprint("sensor.other.collision", {"role_name": ""})]

    def find(self, bid):
        for b in self.bps:
            if b.id == bid:
                return Blueprint(b.id, b.attrs)
        raise IndexError(bid)

    def filter(self, pattern):
        return [Blueprint(b.id, b.attrs) for b in self.bps if fnmatch.fnmatch(b.id, pattern)]


class Actor:
    _next_id = 100

    def __init__(self, world, bp, transform):
        Actor._next_id += 1
        self.id = Actor._next_id
        self.world = world
        self.type_id = bp.id
        self.attributes = {k: str(v) for k, v in bp.attrs.items()}
        self._tf = transform
        self.alive = True

    def get_transform(self):
        t = self._tf
        return Transform(Location(t.location.x, t.location.y, t.location.z), Rotation(t.rotation.pitch, t.rotation.yaw))

    def get_location(self):
        return self.get_transform().location

    def destroy(self):
        self.alive = False
        self.world._actors.pop(self.id, None)


class Vehicle(Actor):
    L, HALF_LEN, HALF_W, MAX_STEER = 2.875, 2.35, 0.95, math.radians(70.0)

    def __init__(self, world, bp, transform):
        super().__init__(world, bp, transform)
        self.v = 0.0
        self.control = VehicleControl()
        self._tf.location.z = 0.0                # "rơi" xuống mặt đường

    def apply_control(self, c):
        self.control = c

    def set_simulate_physics(self, on):
        pass

    def get_velocity(self):
        y = math.radians(self._tf.rotation.yaw)
        return Vector3D(self.v * math.cos(y), self.v * math.sin(y), 0)

    def _step(self, dt):
        c = self.control
        if c.hand_brake:
            self.v = 0.0
            return
        acc = 4.0 * c.throttle - 9.0 * c.brake - 0.02 * self.v * self.v / 10.0
        self.v = max(0.0, self.v + acc * dt)
        delta = float(np.clip(c.steer, -1, 1)) * self.MAX_STEER
        yaw = math.radians(self._tf.rotation.yaw)
        yaw += self.v / self.L * math.tan(delta) * dt
        self._tf.location.x += self.v * math.cos(yaw) * dt
        self._tf.location.y += self.v * math.sin(yaw) * dt
        self._tf.rotation.yaw = math.degrees(yaw)

    def corners(self, scale=1.0):
        y = math.radians(self._tf.rotation.yaw)
        f, r = np.array([math.cos(y), math.sin(y)]), np.array([-math.sin(y), math.cos(y)])
        c = np.array([self._tf.location.x, self._tf.location.y])
        pts = []
        for a in (-1, -0.5, 0, 0.5, 1):
            for b in (-1, 0, 1):
                pts.append(c + f * a * self.HALF_LEN * scale + r * b * self.HALF_W * scale)
        return np.array(pts)


class Sensor(Actor):
    def __init__(self, world, bp, transform, parent):
        super().__init__(world, bp, transform)
        self.parent = parent
        self.cb = None

    @property
    def is_listening(self):
        return self.cb is not None

    def listen(self, cb):
        self.cb = cb

    def stop(self):
        self.cb = None

    def world_matrix(self):
        rel = _matrix(self._tf.location, self._tf.rotation)
        return np.array(self.parent.get_transform().get_matrix()) @ rel


class Image:
    def __init__(self, frame, transform, w, h, raw):
        self.frame, self.transform, self.width, self.height, self.raw_data = frame, transform, w, h, raw


class Settings:
    def __init__(self):
        self.synchronous_mode, self.fixed_delta_seconds = False, None


# ----------------------------------------------------------------- thế giới
class World:
    def __init__(self):
        self._actors = {}
        self._settings = Settings()
        self.frame = 0
        self.bp_lib = BlueprintLibrary()
        self.debug = types.SimpleNamespace(draw_point=lambda *a, **k: None, draw_string=lambda *a, **k: None)
        self._spectator = types.SimpleNamespace(set_transform=lambda t: None)
        self.collisions_reported = 0

    def get_map(self):
        return Map()

    def get_settings(self):
        s = Settings()
        s.synchronous_mode, s.fixed_delta_seconds = self._settings.synchronous_mode, self._settings.fixed_delta_seconds
        return s

    def apply_settings(self, s):
        self._settings = s

    def unload_map_layer(self, layer):
        pass

    def get_blueprint_library(self):
        return self.bp_lib

    def get_actors(self):
        return list(self._actors.values())

    def get_spectator(self):
        return self._spectator

    def _vehicles(self):
        return [a for a in self._actors.values() if isinstance(a, Vehicle)]

    def try_spawn_actor(self, bp, tf):
        if not bp.id.startswith("vehicle."):
            return self.spawn_actor(bp, tf)
        cand = Vehicle(self, bp, Transform(Location(tf.location.x, tf.location.y, tf.location.z),
                                           Rotation(yaw=tf.rotation.yaw)))
        for v in self._vehicles():
            if np.linalg.norm(np.array([v._tf.location.x - tf.location.x, v._tf.location.y - tf.location.y])) < 4.0:
                return None
        self._actors[cand.id] = cand
        return cand

    def spawn_actor(self, bp, tf, attach_to=None):
        if bp.id.startswith("vehicle."):
            a = self.try_spawn_actor(bp, tf)
            if a is None:
                raise RuntimeError("Spawn failed because of collision at spawn position")
            return a
        a = Sensor(self, bp, Transform(Location(tf.location.x, tf.location.y, tf.location.z),
                                       Rotation(tf.rotation.pitch, tf.rotation.yaw)), attach_to)
        self._actors[a.id] = a
        return a

    def tick(self):
        dt = self._settings.fixed_delta_seconds or 0.05
        self.frame += 1
        for v in self._vehicles():
            v._step(dt)
        sensors = [a for a in self._actors.values() if isinstance(a, Sensor) and a.cb]
        cache = {}
        for s in sensors:
            if s.type_id == "sensor.other.collision":
                self._check_collision(s)
                continue
            w, h, fov = int(s.attributes["image_size_x"]), int(s.attributes["image_size_y"]), float(s.attributes["fov"])
            M = s.world_matrix()
            key = (round(float(M[0, 3]), 4), round(float(M[1, 3]), 4), w, h)
            if key not in cache:
                cache[key] = render(M, self._boxes(exclude=s.parent), w, h, fov)
            tags, depth = cache[key]
            bgra = np.zeros((h, w, 4), np.uint8)
            if s.type_id == "sensor.camera.depth":
                val = np.round(np.clip(depth / 1000.0, 0, 1) * 16777215).astype(np.int64)
                bgra[:, :, 2], bgra[:, :, 1], bgra[:, :, 0] = val & 255, (val >> 8) & 255, (val >> 16) & 255
            elif s.type_id == "sensor.camera.semantic_segmentation":
                bgra[:, :, 2] = tags
            s.cb(Image(self.frame, Transform(matrix=M), w, h, bgra.tobytes()))
        return self.frame

    def _boxes(self, exclude=None):
        out = []
        for v in self._vehicles():
            if v is exclude:
                continue
            x, y = v._tf.location.x, v._tf.location.y
            out.append(((x - 2.25, y - 0.9, 0.0), (x + 2.25, y + 0.9, 1.5)))
        return out

    def _check_collision(self, s):
        ego = s.parent
        pts = ego.corners()
        for v in self._vehicles():
            if v is ego:
                continue
            x, y = v._tf.location.x, v._tf.location.y
            inside = (np.abs(pts[:, 0] - x) <= 2.25) & (np.abs(pts[:, 1] - y) <= 0.9)
            if inside.any():
                s.cb(types.SimpleNamespace(frame=self.frame, other_actor=v, normal_impulse=Vector3D(100, 0, 0)))
                return


def render(cam_M, boxes, w, h, fov):
    f = w / (2.0 * math.tan(math.radians(fov) / 2.0))
    cx, cy = w / 2.0, h / 2.0
    vs, us = np.mgrid[0:h, 0:w]
    d_cam = np.stack([np.ones(h * w), (us.ravel() + 0.5 - cx) / f, -(vs.ravel() + 0.5 - cy) / f], 1)
    R, o = cam_M[:3, :3], cam_M[:3, 3]
    D = d_cam @ R.T
    depth = np.full(h * w, 1000.0)
    tags = np.full(h * w, 11, dtype=np.uint8)
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = np.where(D[:, 2] < 0, -o[2] / D[:, 2], np.inf)
    hit = np.isfinite(tg) & (tg < depth)
    gy = o[1] + tg * D[:, 1]
    road = hit & (gy > -8.75) & (gy < 1.75)
    depth[hit] = tg[hit]
    tags[hit] = 2
    tags[road] = 1
    for lo, hi in boxes:
        lo, hi = np.array(lo, float), np.array(hi, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - o) / D, (hi - o) / D
        tmin = np.nanmax(np.minimum(t1, t2), 1)
        tmax = np.nanmin(np.maximum(t1, t2), 1)
        hb = (tmax >= np.maximum(tmin, 0)) & (tmin > 0) & (tmin < depth)
        depth[hb], tags[hb] = tmin[hb], 14
    return tags.reshape(h, w), depth.reshape(h, w).astype(np.float32)


class Client:
    _world = None

    def __init__(self, host="localhost", port=2000):
        if Client._world is None:
            Client._world = World()

    def set_timeout(self, t):
        pass

    def get_world(self):
        return Client._world

    def load_world(self, town):
        return Client._world

    def apply_batch_sync(self, cmds, do_tick=False):
        for _, i in cmds:
            a = Client._world._actors.get(i)
            if a is not None:
                a.destroy()
        return []
