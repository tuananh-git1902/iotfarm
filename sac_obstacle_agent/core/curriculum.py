# -*- coding: utf-8 -*-
"""Học từ các episode đã chạy: curriculum tự lên cấp + chạy lại kịch bản từng thất bại.

SAC vốn đã "học từ quá khứ" qua replay buffer (kho lưu mọi bước đã chạy). Module này thêm
2 cơ chế ở cấp episode:
  * AdaptiveCurriculum: bắt đầu 0 vật cản; tỉ lệ thành công ≥ 80% trong 50 episode gần nhất
    thì tự tăng lên 1, 2, 3 vật cản. Thỉnh thoảng ôn lại cấp dễ để không quên.
  * FailureReplay: lưu "hạt giống" (seed) của kịch bản thất bại và cho chạy lại sau,
    giống người học lái tập lại đúng đoạn đường mình hay đâm.
"""
import random
from collections import deque


class AdaptiveCurriculum:
    def __init__(self, levels, start_level=0, promote_success=0.8, window=50, p_rehearsal=0.15):
        self.levels = list(levels)
        self.level = int(start_level)
        self.promote_success = promote_success
        self.window = window
        self.p_rehearsal = p_rehearsal
        self.history = deque(maxlen=window)
        self.promotions = []

    @property
    def n_obstacles(self):
        return self.levels[self.level]

    def sample(self, rng):
        """Số vật cản cho episode tới và nguồn ('new' hoặc 'rehearsal')."""
        if self.level > 0 and rng.random() < self.p_rehearsal:
            return self.levels[rng.randrange(0, self.level)], "rehearsal"
        return self.n_obstacles, "new"

    def report(self, success, n_obstacles, episode):
        if n_obstacles != self.n_obstacles:
            return False                     # chỉ tính kết quả ở cấp hiện tại
        self.history.append(1.0 if success else 0.0)
        full = len(self.history) >= self.window
        if full and self.success_rate >= self.promote_success and self.level < len(self.levels) - 1:
            self.level += 1
            self.history.clear()
            self.promotions.append((episode, self.level))
            return True
        return False

    @property
    def success_rate(self):
        return sum(self.history) / len(self.history) if self.history else 0.0


class FailureReplay:
    def __init__(self, capacity=60, p_replay=0.25, max_replays=3):
        self.items = deque(maxlen=capacity)   # [seed, n_obstacles, số lần đã chạy lại]
        self.p_replay = p_replay
        self.max_replays = max_replays

    def add(self, seed, n_obstacles):
        self.items.append([seed, n_obstacles, 0])

    def sample(self, rng):
        if not self.items or rng.random() >= self.p_replay:
            return None
        item = self.items[rng.randrange(len(self.items))]
        item[2] += 1
        if item[2] >= self.max_replays:
            self.items.remove(item)
        return item[0], item[1]


class ScenarioScheduler:
    """Quyết định kịch bản (seed + số vật cản) cho mỗi episode."""

    def __init__(self, cfg, seed=0, fixed_obstacles=None):
        self.rng = random.Random(seed)
        self.fixed = fixed_obstacles
        self.enabled = cfg.enabled and fixed_obstacles is None
        self.curriculum = AdaptiveCurriculum(cfg.levels, cfg.start_level, cfg.promote_success,
                                             cfg.window, cfg.p_rehearsal)
        self.replay = FailureReplay(cfg.replay_capacity, cfg.p_failure_replay, cfg.max_replays)
        self.eval_queue = None
        self.episode = 0

    def set_eval(self, scenarios):
        """Chế độ đánh giá: chạy lần lượt danh sách (seed, n_obstacles) cố định."""
        self.eval_queue = deque(scenarios)

    def next(self):
        self.episode += 1
        if self.eval_queue is not None:
            seed, n = self.eval_queue[0]
            self.eval_queue.rotate(-1)
            return seed, n, "eval"
        if not self.enabled:
            return self.rng.randrange(1, 10**9), int(self.fixed or 0), "fixed"
        rep = self.replay.sample(self.rng)
        if rep is not None:
            return rep[0], rep[1], "replay"
        n, source = self.curriculum.sample(self.rng)
        return self.rng.randrange(1, 10**9), n, source

    def report(self, seed, n_obstacles, source, success):
        if source in ("eval", "fixed"):
            return False
        if not success and source != "replay":
            self.replay.add(seed, n_obstacles)
        if source == "new":
            return self.curriculum.report(success, n_obstacles, self.episode)
        return False
