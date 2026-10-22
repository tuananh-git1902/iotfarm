# Agent SAC né vật cản tĩnh trong CARLA

Gói code hoàn chỉnh để huấn luyện một **agent SAC** (Soft Actor-Critic) lái **Tesla Model 3** trong CARLA:
bám theo tuyến đường, **nhận diện vật cản tĩnh** bằng camera, **đổi làn để né rồi quay về làn gốc**,
và **học dần từ các episode đã chạy**. Thiết kế bám theo đồ án *"Huấn luyện xe tự hành với giải thuật
học tăng cường SAC trên nền tảng CARLA"* (HCMUT, 2025), đồng thời sửa các điểm yếu mà đồ án tự nêu ra.

---

## 1. So với đồ án gốc, gói này giữ gì và sửa gì

| Hạng mục | Đồ án gốc | Gói này |
|---|---|---|
| Thuật toán | SAC, Stable-Baselines3, `MultiInputPolicy`, mạng [500, 300], `log_std_init = -3` | **Giữ nguyên** |
| Encoder ảnh | 3 tầng đầu ResNet18 ("ResNetEarlySkip") | Giữ, nhưng ghép trung bình của cả 3 tầng (448 số), bỏ lớp `conv_fuse` chưa huấn luyện |
| Thông tin vị trí trong ảnh | Mất do average pooling | **Thêm** lưới semantic thô 2 × 8 × 16 (vị trí đường và xe trong ảnh) |
| Nhận diện vật cản | Đếm > 8% pixel "xe" trên **cả ảnh**, chia đôi trái/phải | **Chiếu từng pixel ra 3D bằng camera depth**, chỉ tính điểm nằm **trong hành lang của tuyến** |
| Khoảng cách phát hiện | Rất gần (xem ghi chú bên dưới) | Khoảng **30 m**, kể cả vật cản sau khúc cua |
| Tuyến né | Đổi sang làn bên cạnh | Dời ngang **mượt** (đường cong smoothstep); né được 2 vật cản sát nhau |
| Phần thưởng tốc độ | Thưởng như nhau trong cả vùng [min, target] → xe chạy 12–15 km/h | Thưởng tăng dần tới 30 km/h + **thưởng theo số mét tiến được** |
| Hành động | Lái + ga | Lái + **ga/phanh** (1 trục từ −1 tới 1) để dừng kịp trước vật cản |
| Học từ episode cũ | Replay buffer | Replay buffer + **curriculum tự lên cấp** + **chạy lại kịch bản thất bại** |
| Kiểm thử | — | 19 bài kiểm thử lõi + 8 bài chạy trọn vòng trên CARLA giả lập |

**Ghi chú về quy tắc 8%.** Mình dựng lại camera với W = 160, H = 80, FOV 90°, cao 1,7 m, chúc 8°.
Đồ án không ghi độ cao và góc chúc, nên hai thông số này là giả định. Với cấu hình đó, một chiếc ô tô
chỉ chiếm 8% ảnh khi cách tâm xe khoảng **5–6 m**. Đây là lý do hợp lý giải thích vì sao đồ án hay
đâm vật cản ở khúc cua: phát hiện quá muộn. Cách mới phát hiện từ khoảng 30 m.

---

## 2. Kiến trúc và luồng dữ liệu mỗi bước (0,05 s)

```
            ┌──────────────────────── CARLA (Town10HD_Opt, chế độ đồng bộ 20 Hz) ───────────────────────┐
            │  Tesla Model 3  ──►  camera SEMANTIC  W=160 H=80 ┐                                          │
            │  (sensor đặt x=1.5 m, z=1.7 m, chúc 8°)            ├──► ảnh nhãn + ảnh độ sâu (cùng pixel)  │
            │                 ──►  camera DEPTH     W=160 H=80 ┘                                          │
            │                 ──►  cảm biến VA CHẠM                                                       │
            └──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                                           ▼
  core/perception.py   NHẬN DIỆN: pixel "xe" → điểm 3D → nằm trong hành lang tuyến? → vật cản gần nhất (m)
                       THEO DÕI: gộp nhiều khung hình → nhớ vị trí, biết vật cản đứng yên (tĩnh)
                                                           ▼
  core/avoidance.py    LẬP TUYẾN NÉ: chọn làn (cùng chiều trước, ngược chiều sau) → dời tuyến mượt
                       PHA NÉ:  0 bình thường → 0.5 tiếp cận → 1 chuyển làn → 2 vượt qua → 3 về làn
                                                           ▼
  envs/obstacle_env.py OBSERVATION (757 số) ──► AGENT SAC: Actor [500, 300] ──► ACTION [lái, ga/phanh]
                                                           │          Critic Q1, Q2 [500, 300]
  core/reward.py       REWARD + kết thúc ◄─────────────────┘
  core/curriculum.py   hết episode → báo kết quả → lên cấp / lưu kịch bản thất bại để chạy lại
```

### Observation: agent "nhìn thấy" gì

| Khóa | Số chiều | Nội dung |
|---|---|---|
| `vehicle` | 5 | góc lái, ga/phanh vừa áp, tốc độ / 50, góc lệch hướng / π, lệch tâm tuyến / 3 m |
| `waypoints` | 30 | 15 điểm tiếp theo của **tuyến dự kiến** (đã gồm đoạn né), hệ tọa độ xe, / 50 m |
| `maneuver` | 6 | thao tác sắp tới (LEFT, RIGHT, STRAIGHT, LANEFOLLOW, đổi làn trái/phải) |
| `goal` | 3 | vị trí đích trong hệ tọa độ xe, quãng đường còn lại |
| `obstacle` | 4 | có vật cản?, khoảng cách / 30 m, lệch ngang so với làn đang đi, vật cản tĩnh? |
| `avoid_phase` | 5 | pha né hiện tại (one-hot) |
| `sem_grid` | 256 | tỉ lệ pixel "đường" và "xe" trong 8 × 16 ô của ảnh |
| `img_latent` | 448 | đặc trưng ảnh từ ResNet18 (tắt bằng `--no-resnet`) |

### Action: agent làm gì

`[lái, ga_phanh]`, mỗi giá trị trong [−1, 1].
- Lái dương là rẽ phải.
- `ga_phanh` dương là đạp ga, âm là đạp phanh.

### Reward: phần thưởng mỗi bước

| Thành phần | Công thức / giá trị | Nguồn |
|---|---|---|
| Liên tục | `r_tốc_độ × r_giữ_tâm × r_góc` (mỗi phần từ 0 tới 1) | như đồ án |
| r_tốc_độ | tăng tuyến tính tới 1 ở **30 km/h** (khi đang né: **15 km/h**), về 0 ở 40 km/h | sửa |
| Tiến độ | `+0,5 × số mét tiến về đích` | mới |
| Lái giật | `−0,1 × |Δ góc lái|` | mới |
| Xong pha né | `+0,3` khi pha 1 → 2 và pha 3 → 0 | như đồ án |
| Né mà chạy nhanh | `−0,05` khi đang né và > 15 km/h | như đồ án |
| Va chạm / lệch tuyến > 3 m | `−10`, kết thúc | mạnh hơn đồ án (−0,5) |
| Đứng yên 10 s | `−5`, kết thúc | |
| Tới đích (< 5 m) | `+20`, kết thúc | |
| Hết 125 s | cắt ngang (không phạt) | |

---

## 3. Cấu trúc thư mục

```
sac_obstacle_agent/
├── config.py               ← MỌI tham số (camera, nhận diện, né, reward, SAC, curriculum)
├── core/                   ← logic thuần NumPy, kiểm thử được không cần CARLA
│   ├── geometry.py         tọa độ, tuyến đường, chiếu điểm lên tuyến
│   ├── perception.py       nhận diện + theo dõi vật cản
│   ├── avoidance.py        lập tuyến né, các pha né
│   ├── reward.py           phần thưởng + điều kiện kết thúc
│   ├── curriculum.py       curriculum + chạy lại kịch bản thất bại
│   └── expert.py           bộ lái luật (pure pursuit) để kiểm tra và làm mốc so sánh
├── carla_bridge/           ← giao tiếp CARLA: kết nối, tuyến A*, sinh vật cản, cảm biến
├── models/encoder.py       ← ResNet18 3 tầng đầu (đóng băng)
├── envs/obstacle_env.py    ← môi trường Gymnasium ghép tất cả lại
├── train.py                ← huấn luyện
├── evaluate.py             ← đánh giá, in bảng như đồ án
├── drive.py                ← TÍCH HỢP: agent lái xe, có màn hình theo dõi, quay video
├── tools/check_env.py      ← kiểm tra môi trường với CARLA thật trước khi train
└── tests/                  ← kiểm thử (có CARLA giả lập trong tests/fakes)
```

---

## 4. Cài đặt

Làm theo hướng dẫn cài CARLA 0.9.16, Miniconda, PyTorch CUDA đã có. Sau đó, trong Anaconda Prompt:

```
conda activate carla
cd /d D:\DATN\code\sac_obstacle_agent
pip install -r requirements.txt
```

`PYTHONPATH` phải trỏ tới `CARLA_0.9.16\PythonAPI\carla` để import được `agents` (GlobalRoutePlanner).

---

## 5. Quy trình chạy từng bước

### Bước 1: Kiểm thử không cần CARLA (2–3 phút)
```
python tests/test_core.py        # 19 bài: hình học, nhận diện, né, reward, curriculum
python tests/test_env_fake.py    # 8 bài: chạy trọn vòng môi trường trên CARLA giả lập
```
Phải ra `19/19` và `8/8`. Mỗi lần sửa code trong `core/`, chạy lại bước này trước.

### Bước 2: Kiểm tra với CARLA thật (bộ lái luật, chưa có AI)
Mở CARLA (`CarlaUE4.exe -quality-level=Low -ResX=800 -ResY=600`), rồi chạy:
```
python tools/check_env.py --obstacles 1
python tools/check_env.py --obstacles 2 --episodes 3
```
Trong cửa sổ CARLA, bạn sẽ thấy:
- chấm **xanh lá** là tuyến bình thường,
- chấm **vàng** là tuyến né,
- chấm **đỏ** là vật cản xe đã nhận diện.

Terminal in tốc độ, pha né, khoảng cách vật cản, reward. **Bộ lái luật phải né được vật cản và tới đích.**
Nếu chưa được, sửa môi trường trước. Đừng train khi môi trường còn sai.

### Bước 3: Huấn luyện thử (khoảng 30 phút)
```
python train.py --name thu_nhanh --steps 20000 --no-resnet
```
Mục tiêu: chạy hết không lỗi, có thư mục `runs/thu_nhanh/` gồm `final.zip`, `episodes.csv`, `tb/`.

### Bước 4: Huấn luyện đầy đủ (vài giờ tới một đêm)
```
python train.py --name sac_v1 --steps 1000000
```
- Curriculum tự bắt đầu với **0 vật cản**. Khi tỉ lệ thành công ≥ 80% trong 50 episode thì tự lên **1 → 2 → 3** vật cản.
- Dừng giữa chừng bằng `Ctrl + C`, model vẫn được lưu.
- Huấn luyện tiếp:
  ```
  python train.py --name sac_v1b --steps 500000 --resume runs/sac_v1/final.zip
  ```

Theo dõi trong cửa sổ khác: `tensorboard --logdir runs`, rồi mở http://localhost:6006

| Biểu đồ | Ý nghĩa | Dấu hiệu tốt |
|---|---|---|
| `episode/success_rate` | tỉ lệ tới đích (50 episode gần nhất) | tăng, đạt ≥ 0,8 rồi tụt khi lên cấp mới (bình thường) |
| `curriculum/n_obstacles` | đang tập với bao nhiêu vật cản | bậc thang 0 → 1 → 2 → 3 |
| `episode/collision_obstacle_rate` | tỉ lệ đâm vật cản | giảm dần |
| `obstacle/first_detect_dist_m` | phát hiện vật cản từ bao xa | khoảng 25–30 m |
| `episode/mean_speed_kmh` | tốc độ trung bình | 20–28 km/h (đồ án: 12–15) |
| `rollout/ep_rew_mean` | tổng reward trung bình | tăng rồi đi ngang |
| `train/critic_loss` | sai số của Critic | giảm nhanh rồi ổn định |
| `train/ent_coef` | hệ số khám phá α | giảm dần |

`runs/sac_v1/episodes.csv` ghi từng episode: kết quả, tốc độ, độ lệch, khoảng cách phát hiện, kịch bản (seed).
Mở bằng Excel để xem xe hay hỏng ở đâu.

### Bước 5: Đánh giá (như bảng kết quả của đồ án)
```
python evaluate.py runs/sac_v1/best_model.zip --episodes 20 --levels 0 1 2
python evaluate.py --expert --episodes 20        # mốc so sánh: bộ lái luật
```
Kịch bản đánh giá dùng seed khác lúc train, tức là tuyến và vật cản **chưa từng thấy**.

### Bước 6: Tích hợp agent vào xe và xem nó lái
```
python drive.py runs/sac_v1/best_model.zip --obstacles 2 --episodes 3
python drive.py runs/sac_v1/best_model.zip --obstacles 2 --record demo.mp4
```
Phím: `ESC` thoát, `N` sang episode tiếp theo.

---

## 6. Tích hợp agent SAC vào một mô hình xe trong CARLA

"Tích hợp" nghĩa là nối 4 thứ lại với nhau, và **phải giống hệt lúc huấn luyện**:

1. **Xe:** blueprint `vehicle.tesla.model3`, spawn bằng `world.try_spawn_actor(...)` (`carla_bridge/world.py → spawn_ego`).
2. **Cảm biến:** camera semantic + depth W = 160, H = 80, FOV 90°, đặt `x = 1.5, z = 1.7, pitch = −8`, cùng cảm biến va chạm (`carla_bridge/sensors.py`). **Đổi vị trí camera thì agent "nhìn" khác → phải train lại.**
3. **Đường ống observation:** ảnh → nhận diện → tuyến né → 757 số, theo đúng thứ tự trong `envs/obstacle_env.py → _observe()`. Cấu hình lúc train được lưu ở `runs/<tên>/config.json`. `drive.py` và `evaluate.py` tự đọc file này.
4. **Vòng điều khiển 20 Hz đồng bộ:** observation → `model.predict(obs, deterministic=True)` → `carla.VehicleControl(throttle, steer, brake)` → `world.tick()`.

Đoạn code tối giản nhất để agent lái xe (chính là lõi của `drive.py`):

```python
from stable_baselines3 import SAC
from envs.obstacle_env import ObstacleAvoidEnv
from run_utils import load_run_config

cfg = load_run_config("runs/sac_v1/best_model.zip")      # đúng cấu hình lúc train
env = ObstacleAvoidEnv(cfg, fixed_obstacles=2)             # spawn Tesla + cảm biến + vật cản
model = SAC.load("runs/sac_v1/best_model.zip")             # bộ não đã học

obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)     # tất định: không thêm nhiễu khám phá
    obs, reward, terminated, truncated, info = env.step(action)   # áp lệnh lên xe, tick CARLA
    done = terminated or truncated
print(info["episode_summary"])
env.close()
```

**Đổi sang mẫu xe hoặc bản đồ khác:**
```
python drive.py runs/sac_v1/best_model.zip --vehicle vehicle.lincoln.mkz_2020
python drive.py runs/sac_v1/best_model.zip --town Town03
```
Xe khác có khối lượng, chiều dài cơ sở và góc lái khác. Agent vẫn chạy được, nhưng thường kém hơn.
Cách chuẩn là **huấn luyện tiếp (fine-tune)** vài trăm nghìn bước trên xe mới:
```
python train.py --name sac_lincoln --steps 300000 --resume runs/sac_v1/best_model.zip --vehicle vehicle.lincoln.mkz_2020
```

---

## 7. Agent "học từ các episode đã chạy" bằng cách nào

1. **Replay buffer (bộ nhớ trải nghiệm):** SAC lưu mọi bước `(observation, action, reward, observation kế)` của **mọi episode cũ**, rồi rút ngẫu nhiên 256 mẫu để học. Encoder ảnh bị đóng băng, nên mỗi mẫu chỉ khoảng 6 KB: 200.000 mẫu ≈ 1,2 GB RAM.
2. **Curriculum tự lên cấp:** học dễ trước (0 vật cản) rồi khó dần. Thỉnh thoảng (15%) ôn lại cấp dễ để không quên.
3. **Chạy lại kịch bản thất bại:** mỗi episode thất bại được lưu "hạt giống" (seed). 25% episode sau đó là chạy lại đúng tuyến và vật cản đã làm xe hỏng, tối đa 3 lần mỗi kịch bản.
4. **Model tốt nhất:** `best_model.zip` được lưu mỗi khi đạt cấp cao hơn, hoặc thành công nhiều hơn ở cùng cấp.

---

## 8. Tham số hay chỉnh (trong `config.py`)

| Muốn | Chỉnh |
|---|---|
| Xe chạy nhanh hơn / chậm hơn | `RewardConfig.target_speed`, `w_progress` |
| Né sớm hơn / chuyển làn êm hơn | `AvoidConfig.lead`, `ramp` |
| Phát hiện xa hơn | `DetectionConfig.detect_range` (camera nhỏ thì xa quá sẽ ít pixel) |
| Không cho mượn làn ngược chiều | `AvoidConfig.allow_opposite_lane = False` |
| Máy yếu / thiếu RAM | `--no-resnet`, `--buffer 100000`, mở CARLA với `-RenderOffScreen` |
| Vật cản dày hơn | `ObstacleConfig.min_gap` (mặc định 35 m) |

---

## 9. Lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `time-out while waiting for the simulator` | Mở CARLA trước; cho phép CARLA qua Firewall |
| `Không nhận được ảnh cảm biến cho frame …` | Có script khác cũng đang `tick()` CARLA → tắt nó; hoặc khởi động lại CARLA |
| `No module named 'agents'` | Chưa đặt `PYTHONPATH` tới `PythonAPI\carla` |
| Lỗi kích thước observation khi `--resume` | Đã đổi `use_resnet` hoặc camera so với lúc train → giữ nguyên `config.json` |
| CARLA đứng hình sau khi tắt script đột ngột | Server còn ở chế độ đồng bộ → khởi động lại CARLA |
| Train mãi không lên cấp 1 | Xem `episodes.csv`: nếu `off_route` nhiều, giảm `target_speed`; nếu `stuck` nhiều, tăng `w_progress` |
| Hết VRAM (6 GB) | `-quality-level=Low -RenderOffScreen`; không mở `drive.py` lúc đang train |

Gói này đã được kiểm thử với CARLA giả lập. Lần chạy đầu trên CARLA thật, hãy đi đúng Bước 2
(`tools/check_env.py`) và gửi lại log nếu có lỗi.
