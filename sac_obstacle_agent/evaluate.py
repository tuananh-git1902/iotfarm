# -*- coding: utf-8 -*-
"""BƯỚC ĐÁNH GIÁ: chạy agent (tất định) trên các kịch bản CHƯA dùng khi train, in bảng như đồ án.

  python evaluate.py runs/sac_v1/best_model.zip --episodes 20 --levels 0 1 2
  python evaluate.py --expert --episodes 20                 # mốc so sánh: bộ lái luật
"""
import argparse
import csv
import math
import os

import numpy as np

from config import Config
from core.expert import pure_pursuit
from envs.obstacle_env import ObstacleAvoidEnv
from run_utils import load_run_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=None)
    ap.add_argument("--expert", action="store_true", help="đánh giá bộ lái luật thay cho SAC")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seed-base", type=int, default=900_000, help="khác seed lúc train")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.expert and not args.model:
        ap.error("cần đường dẫn model hoặc --expert")

    cfg = load_run_config(args.model) if args.model else Config()
    cfg.sim.debug_draw = True
    if args.expert:
        cfg.agent.use_resnet = False      # bộ lái luật không cần encoder ảnh
    env = ObstacleAvoidEnv(cfg, seed=0)
    scen = [(args.seed_base + 1000 * lvl + i, lvl) for lvl in args.levels for i in range(args.episodes)]
    env.scheduler.set_eval(scen)
    model = None
    if not args.expert:
        from stable_baselines3 import SAC
        model = SAC.load(args.model)

    rows = []
    try:
        for k in range(len(scen)):
            obs, _ = env.reset()
            done = False
            while not done:
                a = pure_pursuit(obs) if args.expert else model.predict(obs, deterministic=True)[0]
                obs, _, term, trunc, info = env.step(a)
                done = term or trunc
            s = info["episode_summary"]
            rows.append(s)
            print(f"[{k + 1:3d}/{len(scen)}] {s['n_requested']} vật cản: {s['event']:<18} "
                  f"v̄={s['mean_speed_kmh']:4.1f} km/h  lệch̄={s['mean_abs_dev_m']:.2f} m  "
                  f"vượt {s['passed']}/{s['n_obstacles']}")
    finally:
        env.close()

    print("\n================ KẾT QUẢ ĐÁNH GIÁ ================")
    head = f"{'Vật cản':>8} | {'Thành công':>10} | {'Đâm vật cản':>11} | {'Đâm khác':>8} | " \
           f"{'Lệch làn':>8} | {'Kẹt/hết giờ':>11} | {'v̄ km/h':>6} | {'lệch̄ m':>6} | {'Phát hiện ở':>11}"
    print(head)
    print("-" * len(head))
    for lvl in args.levels:
        r = [x for x in rows if x["n_requested"] == lvl]
        if not r:
            continue
        pct = lambda ev: 100 * sum(x["event"] == ev for x in r) / len(r)
        stuck = 100 * sum(x["event"] in ("stuck", "timeout") for x in r) / len(r)
        d = [x["first_detect_dist_m"] for x in r if not math.isnan(x["first_detect_dist_m"])]
        det = f"{np.mean(d):.1f} m" if d else "-"
        print(f"{lvl:>8} | {pct('success'):>9.0f}% | {pct('collision_obstacle'):>10.0f}% | "
              f"{pct('collision_other'):>7.0f}% | {pct('off_route'):>7.0f}% | {stuck:>10.0f}% | "
              f"{np.mean([x['mean_speed_kmh'] for x in r]):>6.1f} | "
              f"{np.mean([x['mean_abs_dev_m'] for x in r]):>6.2f} | {det:>11}")
    print("Tham chiếu đồ án gốc: thành công 95% (0 vật cản), 85% (1), 70–75% (≥2); v̄ 12–15 km/h")

    out = args.out or ("eval_expert.csv" if args.expert else
                       os.path.join(os.path.dirname(os.path.abspath(args.model)), "eval.csv"))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Chi tiết từng episode: {out}")


if __name__ == "__main__":
    main()
