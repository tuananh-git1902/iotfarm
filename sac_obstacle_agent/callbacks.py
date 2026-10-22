# -*- coding: utf-8 -*-
"""Theo dõi quá trình học qua từng episode: TensorBoard + file CSV + lưu model tốt nhất."""
import csv
import math
import os
from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

EVENTS = ("success", "collision_obstacle", "collision_other", "off_route", "stuck", "timeout")


class EpisodeLogger(BaseCallback):
    def __init__(self, out_dir, window=50, min_episodes_for_best=20, verbose=1):
        super().__init__(verbose)
        self.out_dir = out_dir
        self.hist = deque(maxlen=window)
        self.min_best = min_episodes_for_best
        self.best_score = -math.inf
        self.csv_path = os.path.join(out_dir, "episodes.csv")
        self._file, self._writer = None, None

    def _on_step(self):
        for info in self.locals.get("infos", []):
            summary = info.get("episode_summary")
            if summary:
                self._record(summary)
        return True

    def _record(self, s):
        self.hist.append(s)
        if self._writer is None:
            new = not os.path.exists(self.csv_path)
            self._file = open(self.csv_path, "a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=list(s.keys()))
            if new:
                self._writer.writeheader()
        self._writer.writerow(s)
        self._file.flush()

        h = list(self.hist)
        n = len(h)
        for ev in EVENTS:
            self.logger.record(f"episode/{ev}_rate", sum(x["event"] == ev for x in h) / n)
        self.logger.record("episode/mean_speed_kmh", np.mean([x["mean_speed_kmh"] for x in h]))
        self.logger.record("episode/mean_abs_dev_m", np.mean([x["mean_abs_dev_m"] for x in h]))
        self.logger.record("episode/completion", np.mean([x["completion"] for x in h]))
        n_obs = sum(x["n_obstacles"] for x in h)
        if n_obs:
            self.logger.record("obstacle/passed_ratio", sum(x["passed"] for x in h) / n_obs)
            self.logger.record("obstacle/detected_ratio", sum(x["n_detected"] for x in h) / n_obs)
            d = [x["first_detect_dist_m"] for x in h if not math.isnan(x["first_detect_dist_m"])]
            if d:
                self.logger.record("obstacle/first_detect_dist_m", float(np.mean(d)))
        self.logger.record("curriculum/n_obstacles", s["curriculum_level"])

        if s["promoted"] and self.verbose:
            print(f"\n[CURRICULUM] Lên cấp: giờ huấn luyện với {s['curriculum_level']} vật cản "
                  f"(episode {s['episode']}, bước {self.num_timesteps})\n")

        # Model tốt nhất: ưu tiên cấp curriculum cao hơn, rồi tới tỉ lệ thành công ở cấp đó
        level = s["curriculum_level"]
        same = [x for x in h if x["n_requested"] == level and x["source"] == "new"]
        if len(same) >= self.min_best:
            score = level + sum(x["success"] for x in same) / len(same)
            if score > self.best_score:
                self.best_score = score
                self.model.save(os.path.join(self.out_dir, "best_model"))
                if self.verbose:
                    print(f"[BEST] Lưu best_model.zip: {level} vật cản, thành công "
                          f"{100 * (score - level):.0f}% ({len(same)} episode gần nhất)")

    def _on_training_end(self):
        if self._file:
            self._file.close()
