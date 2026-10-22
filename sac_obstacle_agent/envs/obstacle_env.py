# -*- coding: utf-8 -*-
"""Môi trường Gymnasium: Tesla Model 3 bám tuyến và né vật cản tĩnh trong CARLA.

Mỗi bước (0,05 s):
  1. Agent xuất action [lái, ga/phanh] → áp lên xe → CARLA tiến 1 tick.
  2. Đọc camera semantic + depth → NHẬN DIỆN vật cản trong hành lang tuyến (core/perception).
  3. Vật cản mới → lập tuyến né sang làn bên cạnh (core/avoidance) → waypoint trong
     observation đổi theo, agent học bám theo tuyến né.
  4. Tính reward + điều kiện kết thúc (core/reward).
  5. Hết episode → báo kết quả cho curriculum / failure replay (core/curriculum).
"""
import math
import random

import carla
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from carla_bridge.sensors import SensorRig
from carla_bridge.world import (RouteBuilder, cleanup_leftovers, connect, destroy_actors,
                                enable_sync, follow_spectator, lane_options, obstacle_slots,
                                restore_settings, route_to_path, spawn_ego,
                                spawn_static_obstacles, unload_layers)
from config import Config
from core.avoidance import N_PHASES, PH_NORMAL, PHASE_NAMES, compose_offsets, phase_at, plan_bump
from core.curriculum import ScenarioScheduler
from core.geometry import to_local, wrap_angle
from core.perception import (ObstacleTracker, bgra_array, depth_from_bgra, detect_obstacle,
                             intrinsics, semantic_grid, tags_from_bgra, tags_to_rgb)
from core.reward import StepMetrics, compute_reward

ROAD_OPTIONS = ("LEFT", "RIGHT", "STRAIGHT", "LANEFOLLOW", "CHANGELANELEFT", "CHANGELANERIGHT")


class ObstacleAvoidEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg=None, seed=0, fixed_obstacles=None, encoder=None):
        super().__init__()
        self.cfg = cfg or Config()
        c = self.cfg
        self.dt = c.sim.fixed_dt
        self.client, self.world = connect(c.sim)
        self.map = self.world.get_map()
        unload_layers(self.world, c.sim.unload_layers)
        self._orig_settings = enable_sync(self.world, self.dt)
        cleanup_leftovers(self.client, self.world)
        self.world.tick()
        self.routes = RouteBuilder(self.map, c.route.sampling)
        self.intr = intrinsics(c.camera.width, c.camera.height, c.camera.fov)
        self.scheduler = ScenarioScheduler(c.curriculum, seed, fixed_obstacles)
        self.tracker = ObstacleTracker(c.detection.assoc_radius, c.detection.static_speed_tol)

        if encoder is None and c.agent.use_resnet:
            from models.encoder import FrozenImageEncoder   # chỉ cần torch khi dùng ResNet
            encoder = FrozenImageEncoder()
        self.encoder = encoder

        inf = np.inf
        obs = {
            "vehicle": spaces.Box(-inf, inf, (5,), np.float32),        # lái, ga, tốc độ, góc lệch, lệch tâm
            "waypoints": spaces.Box(-1.0, 1.0, (2 * c.route.n_obs_waypoints,), np.float32),
            "maneuver": spaces.Box(0.0, 1.0, (len(ROAD_OPTIONS),), np.float32),
            "goal": spaces.Box(-inf, inf, (3,), np.float32),
            "obstacle": spaces.Box(-inf, inf, (4,), np.float32),       # có?, k/c, lệch ngang, tĩnh?
            "avoid_phase": spaces.Box(0.0, 1.0, (N_PHASES,), np.float32),
            "sem_grid": spaces.Box(0.0, 1.0, (256,), np.float32),
        }
        if self.encoder is not None:
            obs["img_latent"] = spaces.Box(-inf, inf, (self.encoder.out_dim,), np.float32)
        self.observation_space = spaces.Dict(obs)
        # action[0] = góc lái ∈ [-1, 1]; action[1] = ga (>0) hoặc phanh (<0) ∈ [-1, 1]
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)

        self.ego, self.rig, self.obstacles, self.extra_actors = None, None, [], []
        self.episode_count = 0
        self.last_summary = None

    # ================================================================== Gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.scheduler.rng.seed(seed)
        self._destroy_episode()
        c = self.cfg

        scen_seed, n_req, source = self.scheduler.next()
        if options and "scenario" in options:
            scen_seed, n_req = options["scenario"]
            source = "manual"
        rng = random.Random(scen_seed)
        for _ in range(20):
            start_tf, route = self.routes.build(rng, c.route.min_len, c.route.max_len)
            self.ego = spawn_ego(self.world, c.sim.vehicle, start_tf)
            if self.ego is not None:
                break
        if self.ego is None:
            raise RuntimeError("Không spawn được xe sau 20 lần thử")

        self.path, self.route_wps, self.route_opts = route_to_path(route)
        self.lane_opts = lane_options(self.route_wps)
        self.rig = SensorRig(self.world, self.ego, c.camera, c.sim.sensor_timeout)
        slots = obstacle_slots(self.path, self.route_wps, self.lane_opts, n_req, rng, c.obstacle, c.avoid,
                               self.route_opts)
        self.obstacles = spawn_static_obstacles(self.world, self.path, self.route_wps, slots, rng, c.obstacle)
        self.obstacle_ids = {o["id"] for o in self.obstacles}
        self.scenario = {"seed": scen_seed, "n_requested": n_req, "source": source}

        self.tracker.reset()
        self.bumps, self.unavoidable = [], set()
        self.idx, self.step_count, self.stuck_count = 0, 0, 0
        self.prev_steer, self.prev_accel = 0.0, 0.0
        self.phase = self.prev_phase = PH_NORMAL
        self.stats = {"reward": 0.0, "speed_sum": 0.0, "dev_sum": 0.0, "dev_max": 0.0, "dist": 0.0}
        self.last_det, self.last_terms = None, {}

        for _ in range(10):                      # chờ xe rơi xuống mặt đường, cảm biến ổn định
            self.ego.apply_control(carla.VehicleControl(brake=1.0))
            self._tick()
        self.rig.collisions.clear()
        self._update_kinematics()
        self.episode_count += 1
        return self._observe(), {"scenario": dict(self.scenario), "n_obstacles": len(self.obstacles)}

    def step(self, action):
        c = self.cfg
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        steer = float(a[0])
        if c.agent.steer_smoothing > 0:
            steer = (1 - c.agent.steer_smoothing) * steer + c.agent.steer_smoothing * self.prev_steer
        accel = float(a[1])
        self.ego.apply_control(carla.VehicleControl(throttle=max(accel, 0.0), steer=steer,
                                                    brake=max(-accel, 0.0)))
        self._tick()
        self.step_count += 1

        s_prev = self.s_ego
        self._update_kinematics()
        progress = self.s_ego - s_prev
        self._perceive()
        self.phase = phase_at(self.s_ego, self.bumps)
        for o in self.obstacles:
            if not o["passed"] and self.s_ego > o["s"] + c.avoid.obstacle_length + 1.0:
                o["passed"] = True

        collision = len(self.rig.collisions) > 0
        goal = (self.s_ego >= self.path.length - c.reward.goal_radius or
                np.linalg.norm(self.ego_xy - self.path.base_xy[-1]) < c.reward.goal_radius)
        self.stuck_count = self.stuck_count + 1 if (self.step_count > 50 and
                                                    self.speed_kmh < c.reward.stuck_speed_kmh) else 0
        m = StepMetrics(speed_kmh=self.speed_kmh, lat_dev=self.lat_dev, heading_err=self.heading_err,
                        progress_m=progress, steer=steer, prev_steer=self.prev_steer, phase=self.phase,
                        prev_phase=self.prev_phase, collision=collision, goal_reached=goal,
                        stuck=self.stuck_count > c.reward.stuck_steps,
                        timeout=self.step_count >= c.reward.max_steps)
        reward, terminated, truncated, event, terms = compute_reward(m, c.reward)
        self.last_terms = terms

        st = self.stats
        st["reward"] += reward
        st["speed_sum"] += self.speed_kmh
        st["dev_sum"] += abs(self.lat_dev)
        st["dev_max"] = max(st["dev_max"], abs(self.lat_dev))
        st["dist"] += max(progress, 0.0)
        self.prev_steer, self.prev_accel, self.prev_phase = steer, accel, self.phase

        if c.sim.debug_draw and self.step_count % 4 == 0:
            self._debug_draw()

        info = {"event": event, "speed_kmh": self.speed_kmh, "lat_dev": self.lat_dev,
                "phase": PHASE_NAMES[self.phase], "reward_terms": terms,
                "obstacle_dist": None if self.last_det is None else self.last_det.dist}
        if terminated or truncated:
            if event == "collision":
                hit = self.rig.collisions[0]["other_id"]
                event = "collision_obstacle" if hit in self.obstacle_ids else "collision_other"
            info["event"] = event
            info["episode_summary"] = self._summary(event)
        return self._observe(), reward, terminated, truncated, info

    def close(self):
        try:
            self._destroy_episode()
            restore_settings(self.world, self._orig_settings)
        except RuntimeError:
            pass

    # ================================================================== tiện ích cho drive.py
    def register_actor(self, actor):
        """Gắn thêm actor (ví dụ camera quay phim) để môi trường tự xoá khi reset/đóng."""
        self.extra_actors.append(actor)

    @property
    def tags_rgb(self):
        return tags_to_rgb(self.tags)

    # ================================================================== nội bộ
    def _destroy_episode(self):
        actors = list(self.extra_actors)
        if self.rig is not None:
            actors += self.rig.actors
        actors += [o["actor"] for o in self.obstacles]
        if self.ego is not None:
            actors.append(self.ego)
        destroy_actors(self.client, actors)
        self.extra_actors, self.rig, self.obstacles, self.ego = [], None, [], None

    def _tick(self):
        frame = self.world.tick()
        sem, dep = self.rig.read(frame)
        h, w = self.cfg.camera.height, self.cfg.camera.width
        self.tags = tags_from_bgra(bgra_array(sem.raw_data, h, w))
        self.depth = depth_from_bgra(bgra_array(dep.raw_data, h, w))
        self.cam_matrix = np.array(sem.transform.get_matrix())
        if self.cfg.sim.follow_spectator:
            follow_spectator(self.world, self.ego)

    def _update_kinematics(self):
        tf = self.ego.get_transform()
        self.ego_xy = np.array([tf.location.x, tf.location.y])
        self.ego_z = tf.location.z
        self.ego_yaw = math.radians(tf.rotation.yaw)
        k, s, lat, seg_yaw = self.path.project(self.ego_xy, self.idx)
        self.idx = max(self.idx, k)
        self.s_ego, self.lat_dev = s, lat
        self.heading_err = wrap_angle(seg_yaw - self.ego_yaw)
        v = self.ego.get_velocity()
        self.speed_kmh = 3.6 * math.hypot(v.x, v.y)

    def _perceive(self):
        c = self.cfg
        det = detect_obstacle(self.tags, self.depth, self.cam_matrix, self.intr, self.path, self.idx,
                              self.s_ego, self.ego_z, c.detection)
        self.last_det = det
        if det is None:
            return
        i1 = self.path.index_at_s(self.s_ego + c.detection.lookahead + 10.0)
        lat_base = float(self.path.project_many(det.xy[None], self.idx, i1, use_base=True)[1][0])
        track, is_new = self.tracker.update(det, lat_base, self.step_count, self.dt)
        for o in self.obstacles:                     # chỉ để thống kê chất lượng nhận diện
            if o["first_det"] is None and np.linalg.norm(o["xy"] - det.xy) < 4.0:
                o["first_det"] = det.dist
        replan = track.bump is not None and self.s_ego < track.bump.a0 - c.avoid.freeze_before
        if is_new or replan or track.bump is None:
            self._plan_for(track)

    def _plan_for(self, track):
        others = [b for b in self.bumps if b.track_id != track.tid]
        base = compose_offsets(self.path.s, others)
        bump = plan_bump(self.path, self.lane_opts, self.s_ego, track.s_near, track.lat_base, base,
                         self.cfg.avoid, track.tid)
        if bump is None:
            if track.bump is None:
                self.unavoidable.add(track.tid)      # không còn làn nào để né
            return
        self.unavoidable.discard(track.tid)
        track.bump = bump
        self.bumps = others + [bump]
        self.path.set_offsets(compose_offsets(self.path.s, self.bumps))

    def _observe(self):
        c = self.cfg
        n = len(self.path)
        ids = np.clip(np.arange(self.idx + 1, self.idx + 1 + c.route.n_obs_waypoints), 0, n - 1)
        wps = np.clip(to_local(self.path.xy[ids], self.ego_xy, self.ego_yaw) / 50.0, -1, 1).ravel()

        man = np.zeros(len(ROAD_OPTIONS), np.float32)
        opt = self.route_opts[min(self.idx, n - 1)].name
        if opt in ROAD_OPTIONS:
            man[ROAD_OPTIONS.index(opt)] = 1.0

        g = to_local(self.path.base_xy[-1:], self.ego_xy, self.ego_yaw)[0]
        goal = [np.clip(g[0] / 500.0, -1, 1), np.clip(g[1] / 500.0, -1, 1),
                (self.path.length - self.s_ego) / 1000.0]

        tr = self.tracker.nearest_ahead(self.s_ego, c.avoid.obstacle_length)
        if tr is not None and tr.s_near - self.s_ego <= c.detection.detect_range:
            lat_rel = tr.lat_base - self.path.offsets[self.path.index_at_s(tr.s_near)]
            obstacle = [1.0, float(np.clip((tr.s_near - self.s_ego) / c.detection.detect_range, 0, 1)),
                        float(np.clip(lat_rel / 3.0, -1.5, 1.5)), float(tr.is_static)]
        else:
            obstacle = [0.0, 1.0, 0.0, 0.0]

        phase = np.zeros(N_PHASES, np.float32)
        phase[self.phase] = 1.0
        out = {
            "vehicle": np.array([self.prev_steer, self.prev_accel, self.speed_kmh / 50.0,
                                 self.heading_err / math.pi, self.lat_dev / c.reward.max_center_dev], np.float32),
            "waypoints": wps.astype(np.float32),
            "maneuver": man,
            "goal": np.array(goal, np.float32),
            "obstacle": np.array(obstacle, np.float32),
            "avoid_phase": phase,
            "sem_grid": semantic_grid(self.tags, 8, 16, c.detection.vehicle_tags),
        }
        if self.encoder is not None:
            out["img_latent"] = self.encoder(tags_to_rgb(self.tags))
        return out

    def _summary(self, event):
        st, n = self.stats, max(self.step_count, 1)
        dets = [o["first_det"] for o in self.obstacles if o["first_det"] is not None]
        success = event == "success"
        promoted = self.scheduler.report(self.scenario["seed"], self.scenario["n_requested"],
                                         self.scenario["source"], success)
        self.last_summary = {
            "episode": self.episode_count, **self.scenario, "event": event, "success": success,
            "n_obstacles": len(self.obstacles), "passed": sum(o["passed"] for o in self.obstacles),
            "steps": self.step_count, "sim_time_s": self.step_count * self.dt,
            "route_len_m": self.path.length, "completion": min(self.s_ego / self.path.length, 1.0),
            "mean_speed_kmh": st["speed_sum"] / n, "mean_abs_dev_m": st["dev_sum"] / n,
            "max_abs_dev_m": st["dev_max"], "reward_sum": st["reward"],
            "first_detect_dist_m": float(np.mean(dets)) if dets else float("nan"),
            "n_detected": len(dets), "n_unavoidable": len(self.unavoidable),
            "curriculum_level": self.scheduler.curriculum.n_obstacles, "promoted": promoted,
        }
        return self.last_summary

    def _debug_draw(self):
        dbg, life = self.world.debug, self.dt * 4.5
        z = self.ego_z + 0.5
        for i in range(self.idx, min(self.idx + 25, len(self.path))):
            x, y = self.path.xy[i]
            col = carla.Color(0, 220, 0) if abs(self.path.offsets[i]) < 0.1 else carla.Color(255, 200, 0)
            dbg.draw_point(carla.Location(float(x), float(y), z), 0.08, col, life)
        for tr in self.tracker.tracks:
            x, y = tr.xy
            dbg.draw_point(carla.Location(float(x), float(y), z + 1.5), 0.25, carla.Color(255, 0, 0), life)
