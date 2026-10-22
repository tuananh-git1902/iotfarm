# -*- coding: utf-8 -*-
"""KIỂM TRA MÔI TRƯỜNG trước khi huấn luyện (cần CARLA đang chạy).

  python tools/check_env.py                    # bộ lái luật, 1 vật cản, 2 episode
  python tools/check_env.py --obstacles 2 --episodes 3
  python tools/check_env.py --random           # hành động ngẫu nhiên (xe sẽ đâm sớm — bình thường)

Bạn sẽ thấy trong cửa sổ CARLA: chấm XANH LÁ = tuyến bình thường, chấm VÀNG = tuyến né,
chấm ĐỎ = vật cản xe đã nhận diện. Terminal in tốc độ, pha né, khoảng cách vật cản, reward.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config                       # noqa: E402
from core.expert import pure_pursuit            # noqa: E402
from envs.obstacle_env import ObstacleAvoidEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obstacles", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--no-resnet", action="store_true")
    ap.add_argument("--skip-check", action="store_true", help="bỏ qua check_env của SB3")
    args = ap.parse_args()

    cfg = Config()
    cfg.sim.debug_draw = True
    cfg.agent.use_resnet = not args.no_resnet
    env = ObstacleAvoidEnv(cfg, seed=1, fixed_obstacles=args.obstacles)
    try:
        if not args.skip_check:
            from stable_baselines3.common.env_checker import check_env
            print("1) Kiểm tra chuẩn Gymnasium (check_env) …")
            check_env(env, warn=True, skip_render_check=True)
            print("   → ĐẠT\n")
        for ep in range(args.episodes):
            obs, info = env.reset()
            print(f"=== Episode {ep + 1}: kịch bản {info['scenario']}, {info['n_obstacles']} vật cản, "
                  f"tuyến {env.path.length:.0f} m")
            if ep == 0:
                print("    observation:", {k: v.shape for k, v in obs.items()})
            done, t = False, 0
            while not done:
                action = env.action_space.sample() if args.random else pure_pursuit(obs)
                obs, r, term, trunc, info = env.step(action)
                done = term or trunc
                if t % 20 == 0 or done:
                    od = info["obstacle_dist"]
                    terms = {k: round(v, 2) for k, v in info["reward_terms"].items() if v}
                    print(f"  t={t * cfg.sim.fixed_dt:5.1f}s  v={info['speed_kmh']:5.1f} km/h  "
                          f"lệch={info['lat_dev']:+.2f} m  pha={info['phase']:<13} "
                          f"vật cản={'-' if od is None else f'{od:4.1f} m'}  r={r:+.2f} {terms}")
                t += 1
            s = info["episode_summary"]
            print(f"  → {s['event']}: đi {100 * s['completion']:.0f}% tuyến, vượt {s['passed']}/{s['n_obstacles']} "
                  f"vật cản, phát hiện lần đầu ở {s['first_detect_dist_m']:.1f} m, "
                  f"tổng reward {s['reward_sum']:.1f}\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
