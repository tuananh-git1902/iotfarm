# -*- coding: utf-8 -*-
"""Toàn bộ tham số của agent SAC né vật cản tĩnh, gom về một chỗ.

Đổi tham số ở đây (hoặc qua dòng lệnh của train.py) thay vì sửa rải rác trong code.
Các giá trị mặc định bám theo đồ án gốc (HCMUT 2025), kèm ghi chú ở chỗ đã chỉnh.
"""
from dataclasses import asdict, dataclass, field
import json


@dataclass
class SimConfig:
    host: str = "localhost"
    port: int = 2000
    town: str = "Town10HD_Opt"
    fixed_dt: float = 0.05                # 20 Hz, mỗi step = 0,05 s mô phỏng
    vehicle: str = "vehicle.tesla.model3"
    # Bỏ bớt lớp bản đồ cho nhẹ GPU (chỉ có tác dụng với bản đồ *_Opt).
    # Xe đỗ ven đường bị bỏ để không lẫn với vật cản do ta sinh ra.
    unload_layers: tuple = ("ParkedVehicles", "Particles")
    follow_spectator: bool = True         # camera khán giả bám theo xe
    debug_draw: bool = False              # vẽ tuyến dự kiến / vật cản lên cửa sổ CARLA
    sensor_timeout: float = 10.0


@dataclass
class CameraConfig:
    # Camera semantic và depth đặt CÙNG vị trí, CÙNG kích thước → pixel trùng nhau.
    width: int = 160                      # W = 160
    height: int = 80                      # H = 80
    fov: float = 90.0
    x: float = 1.5                        # m, tính từ tâm xe về phía trước
    z: float = 1.7                        # m, chiều cao
    pitch: float = -8.0                   # độ, chúc xuống


@dataclass
class RouteConfig:
    sampling: float = 2.0                 # m giữa 2 waypoint của GlobalRoutePlanner
    min_len: float = 150.0
    max_len: float = 450.0
    n_obs_waypoints: int = 15             # số waypoint đưa vào observation (như đồ án)


@dataclass
class ObstacleConfig:
    min_gap: float = 35.0                 # m giữa 2 vật cản trên tuyến
    start_clearance: float = 45.0         # vật cản đầu tiên cách điểm xuất phát tối thiểu
    goal_clearance: float = 30.0
    lateral_jitter: float = 0.4           # m, lệch ngang ngẫu nhiên khỏi tim làn
    yaw_jitter_deg: float = 10.0
    blueprint_filter: str = "vehicle.*"


@dataclass
class DetectionConfig:
    # Nhãn semantic của CARLA >= 0.9.14: 14 Car, 15 Truck, 16 Bus, 18 Motorcycle, 19 Bicycle
    vehicle_tags: tuple = (14, 15, 16, 18, 19)
    detect_range: float = 30.0            # m, xa hơn thì bỏ qua
    min_pixels: int = 10                  # số pixel tối thiểu trong hành lang để báo vật cản
    lookahead: float = 40.0               # m, chiều dài hành lang xét phía trước
    corridor_margin: float = 0.3          # m, nới thêm mỗi bên ngoài nửa bề rộng làn
    max_height: float = 3.0               # m, điểm cao hơn mặt đường bấy nhiêu thì bỏ (cây, biển)
    assoc_radius: float = 3.5             # m, gộp các lần phát hiện về cùng 1 vật cản
    static_speed_tol: float = 1.5         # m/s, chậm hơn ngưỡng này coi là vật cản tĩnh


@dataclass
class AvoidConfig:
    allow_opposite_lane: bool = True      # cho mượn làn ngược chiều (đồ án có, ưu tiên thấp nhất)
    obstacle_length: float = 5.0          # m, giả định chiều dài vật cản
    lead: float = 6.0                     # m, đã sang hẳn làn bên cạnh trước vật cản bấy nhiêu
    ramp: float = 12.0                    # m, quãng chuyển làn (đi và về)
    clearance_after: float = 6.0          # m, đi qua vật cản thêm bấy nhiêu rồi mới về làn
    min_ramp: float = 5.0
    min_lat_clear: float = 2.6            # m, tâm làn mới phải cách vật cản tối thiểu
    freeze_before: float = 2.0            # m, không chỉnh lại tuyến né khi sắp vào đoạn chuyển


@dataclass
class RewardConfig:
    target_speed: float = 30.0            # km/h, thưởng tốc độ đạt đỉnh ở đây (đồ án: vùng 20–40)
    max_speed: float = 40.0
    avoid_target_speed: float = 15.0      # km/h, tốc độ mong muốn khi đang né (khớp ngưỡng phạt 15 km/h của đồ án)
    center_norm: float = 2.0              # m, thưởng giữ tâm = 0 khi lệch 2 m
    max_angle_deg: float = 20.0
    w_progress: float = 0.5               # thưởng mỗi mét tiến về đích (sửa lỗi xe chạy chậm)
    progress_clip: float = 1.0
    w_steer_rate: float = 0.1             # phạt đánh lái giật
    phase_bonus: float = 0.3              # như đồ án: +0.3 mỗi khi xong 1 giai đoạn né
    w_avoid_overspeed: float = 0.05       # như đồ án: phạt khi đang né mà chạy quá nhanh
    avoid_overspeed_kmh: float = 15.0
    r_collision: float = -10.0
    r_off_route: float = -10.0
    r_stuck: float = -5.0
    r_goal: float = 20.0
    max_center_dev: float = 3.0           # m, lệch khỏi tuyến dự kiến quá mức này → kết thúc
    goal_radius: float = 5.0
    stuck_speed_kmh: float = 1.0
    stuck_steps: int = 200                # 10 s đứng yên → kẹt
    max_steps: int = 2500                 # 125 s mô phỏng → hết giờ


@dataclass
class CurriculumConfig:
    enabled: bool = True
    levels: tuple = (0, 1, 2, 3)          # số vật cản ở mỗi cấp độ
    start_level: int = 0
    promote_success: float = 0.8          # đạt 80% thành công trong cửa sổ → lên cấp
    window: int = 50
    p_rehearsal: float = 0.15             # thỉnh thoảng ôn lại cấp dễ hơn để không quên
    p_failure_replay: float = 0.25        # chạy lại kịch bản từng thất bại
    replay_capacity: int = 60
    max_replays: int = 3


@dataclass
class AgentConfig:
    use_resnet: bool = True               # encoder ResNet18 3 tầng đầu như đồ án
    net_arch: tuple = (500, 300)          # như đồ án
    log_std_init: float = -3.0            # như đồ án
    lr_start: float = 1e-4                # như đồ án (đồ án giảm tới 5e-7, ở đây dừng ở 1e-5)
    lr_end: float = 1e-5
    buffer_size: int = 200_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    learning_starts: int = 5_000
    train_freq: int = 16                  # như đồ án: 16 bước môi trường …
    gradient_steps: int = 8               # … cập nhật mạng 8 lần
    steer_smoothing: float = 0.0          # 0 = không lọc lệnh lái


@dataclass
class Config:
    sim: SimConfig = field(default_factory=SimConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    route: RouteConfig = field(default_factory=RouteConfig)
    obstacle: ObstacleConfig = field(default_factory=ObstacleConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    avoid: AvoidConfig = field(default_factory=AvoidConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    def to_json(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @staticmethod
    def from_json(path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        cfg = Config()
        for section, values in d.items():
            sub = getattr(cfg, section)
            for k, v in values.items():
                cur = getattr(sub, k)
                setattr(sub, k, tuple(v) if isinstance(cur, tuple) else v)
        return cfg
