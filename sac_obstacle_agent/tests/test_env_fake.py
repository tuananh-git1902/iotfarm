# -*- coding: utf-8 -*-
"""Chạy thử TRỌN VÒNG môi trường trên CARLA GIẢ (tests/fakes) — không cần CARLA, gymnasium, torch.

Kiểm tra: reset/step, observation, nhận diện vật cản, lập tuyến né, các pha né, reward,
điều kiện kết thúc, curriculum. Bộ lái luật (pure pursuit) phải né được vật cản và tới đích.
Chạy:  python tests/test_env_fake.py
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "fakes"))           # carla / gymnasium / agents GIẢ
sys.path.insert(1, os.path.dirname(HERE))
for _m in list(sys.modules):
    if _m.split(".")[0] in ("carla", "gymnasium", "agents"):
        del sys.modules[_m]

import numpy as np                                       # noqa: E402

from config import Config                                # noqa: E402
from core.avoidance import PH_PASS, PH_RETURN, PH_SHIFT  # noqa: E402
from core.expert import pure_pursuit                     # noqa: E402
from envs.obstacle_env import ObstacleAvoidEnv           # noqa: E402


def make_env(n_obstacles, seed=3, **cfg_kw):
    cfg = Config()
    cfg.agent.use_resnet = False
    cfg.sim.follow_spectator = False
    for k, v in cfg_kw.items():
        sec, name = k.split("__")
        setattr(getattr(cfg, sec), name, v)
    return ObstacleAvoidEnv(cfg, seed=seed, fixed_obstacles=n_obstacles)


def run(env, policy, options=None):
    obs, _ = env.reset(options=options)
    phases, done, info, total = set(), False, {}, 0.0
    while not done:
        obs, r, term, trunc, info = env.step(policy(obs))
        total += r
        phases.add(env.phase)
        done = term or trunc
    return info["episode_summary"], phases, total


def test_observation_shapes():
    env = make_env(1)
    obs, info = env.reset()
    for k, space in env.observation_space.spaces.items():
        assert obs[k].shape == space.shape, (k, obs[k].shape, space.shape)
        assert obs[k].dtype == np.float32 and np.all(np.isfinite(obs[k])), k
        assert space.contains(obs[k]), f"{k} nằm ngoài giới hạn observation_space"
    assert info["n_obstacles"] == 1
    env.close()


def test_expert_without_obstacles():
    s, _, _ = run(make_env(0), pure_pursuit)
    assert s["event"] == "success", s
    assert s["mean_speed_kmh"] > 18 and s["mean_abs_dev_m"] < 0.5, s


def test_expert_avoids_one_obstacle():
    s, phases, _ = run(make_env(1, seed=5), pure_pursuit)
    assert s["event"] == "success" and s["passed"] == 1, s
    assert {PH_SHIFT, PH_PASS, PH_RETURN} <= phases, phases
    assert s["first_detect_dist_m"] > 20, f"phát hiện quá muộn: {s['first_detect_dist_m']:.1f} m"


def test_expert_avoids_two_obstacles():
    ok = 0
    for seed in (11, 12, 13):
        s, _, _ = run(make_env(2, seed=seed), pure_pursuit)
        ok += s["event"] == "success" and s["passed"] == s["n_obstacles"] == 2
    assert ok == 3, "bộ lái luật phải vượt được 2 vật cản ở cả 3 kịch bản"


def test_ignoring_detour_hits_obstacle():
    def straight(obs):
        speed = obs["vehicle"][2] * 50
        return np.array([0.0, np.clip(0.15 * (25 - speed), -1, 1)], np.float32)
    # Bình thường: tuyến dự kiến đã dời sang làn trái → xe đi thẳng bị "lệch tuyến" trước khi đâm
    s, _, _ = run(make_env(1, seed=5), straight)
    assert s["event"] == "off_route" and s["passed"] == 0, s
    # Nới ngưỡng lệch tuyến → xe đâm thẳng vào vật cản → cảm biến va chạm phải bắt được
    s, _, _ = run(make_env(1, seed=5, reward__max_center_dev=10.0), straight)
    assert s["event"] == "collision_obstacle", s


def test_random_policy_terminates():
    env = make_env(1, seed=7)
    rng = np.random.default_rng(0)
    s, _, _ = run(env, lambda o: rng.uniform(-1, 1, 2).astype(np.float32))
    assert s["event"] in ("off_route", "collision_obstacle", "collision_other", "stuck", "timeout"), s


def test_curriculum_promotes_on_success():
    env = make_env(None, seed=21, curriculum__window=3, curriculum__promote_success=0.6,
                   curriculum__p_failure_replay=0.0, curriculum__p_rehearsal=0.0)
    for _ in range(3):
        s, _, _ = run(env, pure_pursuit)
    assert env.scheduler.curriculum.n_obstacles == 1, (env.scheduler.curriculum.level, s)
    s, _, _ = run(env, pure_pursuit)
    assert s["n_requested"] == 1


def test_same_seed_same_scenario():
    a = make_env(2, seed=1)
    a.reset(options={"scenario": (42, 2)})
    xa = [o["xy"].round(3).tolist() for o in a.obstacles]
    a.reset(options={"scenario": (42, 2)})
    xb = [o["xy"].round(3).tolist() for o in a.obstacles]
    assert xa == xb and len(xa) == 2


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
