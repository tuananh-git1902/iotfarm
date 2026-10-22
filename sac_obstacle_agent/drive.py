# -*- coding: utf-8 -*-
"""TÍCH HỢP AGENT VÀO XE: cho agent SAC đã huấn luyện lái chiếc xe trong CARLA, có màn hình theo dõi.

  python drive.py runs/sac_v1/best_model.zip --obstacles 2 --episodes 3
  python drive.py runs/sac_v1/best_model.zip --record demo.mp4       # quay video
  python drive.py --expert --obstacles 1                              # xem bộ lái luật để so sánh
Phím tắt: ESC thoát, N bỏ qua sang episode tiếp theo.

Màn hình: trái = camera quay theo xe; phải trên = ảnh semantic agent đang "nhìn";
phải giữa = thông số; phải dưới = bản đồ thu nhỏ (xám: tuyến gốc, xanh/vàng: tuyến dự kiến/né,
đỏ: vật cản thật, cam: vật cản xe đã nhận diện).
"""
import argparse
import math

import carla
import numpy as np
import pygame

from config import Config
from core.avoidance import PHASE_NAMES
from core.expert import pure_pursuit
from core.geometry import to_local
from envs.obstacle_env import ObstacleAvoidEnv
from run_utils import load_run_config

CHASE_W, CHASE_H = 960, 540
W, H = 1280, 600
PANEL_X = CHASE_W


class ChaseCamera:
    """Camera RGB gắn sau xe chỉ để hiển thị (agent KHÔNG dùng camera này)."""

    def __init__(self, env):
        bp = env.world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(CHASE_W))
        bp.set_attribute("image_size_y", str(CHASE_H))
        bp.set_attribute("fov", "90")
        tf = carla.Transform(carla.Location(x=-6.5, z=2.8), carla.Rotation(pitch=-12))
        self.actor = env.world.spawn_actor(bp, tf, attach_to=env.ego)
        self.image = None
        self.actor.listen(self._on_image)
        env.register_actor(self.actor)

    def _on_image(self, img):
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
        self.image = arr[:, :, 2::-1].copy()           # BGRA → RGB


def draw_minimap(surf, env, rect, scale=2.6):
    x0, y0, w, h = rect
    pygame.draw.rect(surf, (20, 26, 40), rect)
    surf.set_clip(pygame.Rect(rect))
    cx, cy = x0 + w // 2, y0 + int(h * 0.72)

    def to_px(pts):
        loc = to_local(np.atleast_2d(pts), env.ego_xy, env.ego_yaw)
        return [(int(cx + p[1] * scale), int(cy - p[0] * scale)) for p in loc]

    i0, i1 = max(env.idx - 15, 0), min(env.idx + 60, len(env.path))
    base = to_px(env.path.base_xy[i0:i1])
    plan = to_px(env.path.xy[i0:i1])
    if len(base) > 1:
        pygame.draw.lines(surf, (90, 95, 110), False, base, 2)
        col = (255, 200, 0) if np.abs(env.path.offsets[i0:i1]).max() > 0.1 else (80, 220, 140)
        pygame.draw.lines(surf, col, False, plan, 3)
    for o in env.obstacles:
        px, py = to_px(o["xy"])[0]
        pygame.draw.rect(surf, (230, 60, 60), (px - 5, py - 9, 10, 18))
    for tr in env.tracker.tracks:
        px, py = to_px(tr.xy)[0]
        pygame.draw.circle(surf, (255, 150, 40), (px, py), 6, 2)
    pygame.draw.polygon(surf, (64, 196, 255), [(cx, cy - 10), (cx - 6, cy + 7), (cx + 6, cy + 7)])
    surf.set_clip(None)


def bar(x_mid, value, y=CHASE_H + 20, length=200, thick=16):
    w = int(np.clip(value, -1, 1) * length)
    return pygame.Rect(x_mid + min(w, 0), y, max(abs(w), 1), thick)


def draw(screen, fonts, cam, env, info, action, r, total, ep_label):
    small, big = fonts
    screen.fill((14, 20, 34))
    if cam.image is not None:
        screen.blit(pygame.surfarray.make_surface(np.ascontiguousarray(cam.image.swapaxes(0, 1))), (0, 0))
    sem = pygame.surfarray.make_surface(np.ascontiguousarray(env.tags_rgb.swapaxes(0, 1)))
    screen.blit(pygame.transform.scale(sem, (320, 160)), (PANEL_X, 0))

    tr = env.tracker.nearest_ahead(env.s_ego, env.cfg.avoid.obstacle_length)
    obst = "không" if tr is None else f"{max(tr.s_near - env.s_ego, 0):.1f} m ({'tĩnh' if tr.is_static else 'đang chạy?'})"
    phase = PHASE_NAMES[env.phase]
    lines = [
        (f"{info['speed_kmh']:5.1f} km/h", big, (232, 238, 247)),
        (f"Pha né: {phase}", small, (255, 204, 77) if env.phase else (150, 165, 190)),
        (f"Vật cản phía trước: {obst}", small, (255, 120, 120) if tr else (150, 165, 190)),
        (f"Lệch tâm tuyến: {info['lat_dev']:+.2f} m", small, (232, 238, 247)),
        (f"Lái {action[0]:+.2f}   Ga/Phanh {action[1]:+.2f}", small, (232, 238, 247)),
        (f"Reward bước {r:+.2f}   Tổng {total:7.1f}", small, (232, 238, 247)),
        (ep_label, small, (64, 196, 255)),
    ]
    y = 168
    for text, font, col in lines:
        screen.blit(font.render(text, True, col), (PANEL_X + 10, y))
        y += font.get_height() + 4
    draw_minimap(screen, env, (PANEL_X, 380, 320, 220))
    # thanh lái / ga dưới camera
    pygame.draw.rect(screen, (24, 33, 52), (0, CHASE_H, CHASE_W, H - CHASE_H))
    mid = CHASE_W // 4
    pygame.draw.rect(screen, (64, 196, 255), bar(mid, action[0]))
    pygame.draw.line(screen, (232, 238, 247), (mid, CHASE_H + 14), (mid, CHASE_H + 42), 2)
    col = (80, 220, 140) if action[1] >= 0 else (255, 99, 99)
    mid2 = 3 * CHASE_W // 4
    pygame.draw.rect(screen, col, bar(mid2, action[1]))
    pygame.draw.line(screen, (232, 238, 247), (mid2, CHASE_H + 14), (mid2, CHASE_H + 42), 2)
    screen.blit(small.render("lái", True, (150, 165, 190)), (20, CHASE_H + 18))
    screen.blit(small.render("ga / phanh", True, (150, 165, 190)), (CHASE_W // 2 + 20, CHASE_H + 18))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default=None)
    ap.add_argument("--expert", action="store_true")
    ap.add_argument("--obstacles", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--vehicle", default=None, help="đổi mẫu xe, ví dụ vehicle.lincoln.mkz_2020")
    ap.add_argument("--town", default=None)
    ap.add_argument("--record", default=None, help="đường dẫn file .mp4 để quay")
    args = ap.parse_args()
    if not args.expert and not args.model:
        ap.error("cần đường dẫn model hoặc --expert")

    cfg = load_run_config(args.model) if args.model else Config()
    cfg.sim.debug_draw = True
    if args.expert:
        cfg.agent.use_resnet = False
    if args.vehicle:
        print(f"[CẢNH BÁO] Agent được huấn luyện trên {cfg.sim.vehicle}; xe khác có động lực học khác "
              "→ nên huấn luyện tiếp (fine-tune) bằng train.py --resume.")
        cfg.sim.vehicle = args.vehicle
    if args.town:
        cfg.sim.town = args.town
    env = ObstacleAvoidEnv(cfg, seed=args.seed)
    model = None
    if not args.expert:
        from stable_baselines3 import SAC
        model = SAC.load(args.model)

    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Agent SAC né vật cản tĩnh — CARLA")
    fonts = (pygame.font.SysFont("arial", 18), pygame.font.SysFont("arial", 30, bold=True))
    writer = None
    if args.record:
        import cv2
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), int(1 / cfg.sim.fixed_dt), (W, H))

    quit_all = False
    try:
        for ep in range(args.episodes):
            obs, info = env.reset(options={"scenario": (args.seed + ep, args.obstacles)})
            cam = ChaseCamera(env)
            total, done = 0.0, False
            label = f"Episode {ep + 1}/{args.episodes} • {'bộ lái luật' if args.expert else 'agent SAC'}"
            while not done:
                for e in pygame.event.get():
                    if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                        quit_all = True
                    if e.type == pygame.KEYDOWN and e.key == pygame.K_n:
                        done = True
                if quit_all:
                    break
                action = pure_pursuit(obs) if args.expert else model.predict(obs, deterministic=True)[0]
                obs, r, term, trunc, info = env.step(action)
                total += r
                done = done or term or trunc
                draw(screen, fonts, cam, env, info, action, r, total, label)
                pygame.display.flip()
                if writer is not None:
                    frame = pygame.surfarray.array3d(screen).swapaxes(0, 1)
                    writer.write(frame[:, :, ::-1].copy())
            if quit_all:
                break
            s = info.get("episode_summary")
            if s:
                print(f"Episode {ep + 1}: {s['event']} | vượt {s['passed']}/{s['n_obstacles']} vật cản | "
                      f"v̄ {s['mean_speed_kmh']:.1f} km/h | lệch̄ {s['mean_abs_dev_m']:.2f} m | "
                      f"phát hiện ở {s['first_detect_dist_m']:.1f} m")
    finally:
        if writer is not None:
            writer.release()
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
