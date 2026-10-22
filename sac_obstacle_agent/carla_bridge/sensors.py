# -*- coding: utf-8 -*-
"""Bộ cảm biến gắn lên xe: camera semantic + camera depth (cùng vị trí) + cảm biến va chạm."""
import queue

import carla

from .world import ROLE_SENSOR


class SensorRig:
    def __init__(self, world, ego, cam, timeout=10.0):
        self.timeout = timeout
        bp_lib = world.get_blueprint_library()
        tf = carla.Transform(carla.Location(x=cam.x, z=cam.z), carla.Rotation(pitch=cam.pitch))
        self.q_sem, self.q_depth = queue.Queue(), queue.Queue()
        self.actors = []
        for name, q in (("sensor.camera.semantic_segmentation", self.q_sem), ("sensor.camera.depth", self.q_depth)):
            bp = bp_lib.find(name)
            bp.set_attribute("image_size_x", str(cam.width))
            bp.set_attribute("image_size_y", str(cam.height))
            bp.set_attribute("fov", str(cam.fov))
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", ROLE_SENSOR)
            sensor = world.spawn_actor(bp, tf, attach_to=ego)
            sensor.listen(q.put)
            self.actors.append(sensor)
        col_bp = bp_lib.find("sensor.other.collision")
        if col_bp.has_attribute("role_name"):
            col_bp.set_attribute("role_name", ROLE_SENSOR)
        col = world.spawn_actor(col_bp, carla.Transform(), attach_to=ego)
        col.listen(self._on_collision)
        self.actors.append(col)
        self.collisions = []

    def _on_collision(self, event):
        other = event.other_actor
        self.collisions.append({"frame": event.frame, "other_id": other.id if other else -1,
                                "other_type": other.type_id if other else "?",
                                "impulse": event.normal_impulse.length()})

    def _get(self, q, frame):
        while True:
            try:
                item = q.get(timeout=self.timeout)
            except queue.Empty:
                raise RuntimeError(
                    f"Không nhận được ảnh cảm biến cho frame {frame}. Kiểm tra CARLA còn chạy, "
                    "đang ở chế độ đồng bộ và không có client khác cũng gọi tick().")
            if item.frame >= frame:
                return item

    def read(self, frame):
        """Lấy ảnh semantic + depth đúng khung hình vừa tick."""
        return self._get(self.q_sem, frame), self._get(self.q_depth, frame)
