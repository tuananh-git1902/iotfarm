# -*- coding: utf-8 -*-
"""Bộ trích đặc trưng ảnh "ResNetEarlySkip" như đồ án: 3 tầng đầu của ResNet18 (đã học trên ImageNet).

Đồ án: ảnh (3, 80, 160) → stem → layer1 (64, 20, 40) → layer2 (128, 10, 20) → layer3 (256, 5, 10).
Ở đây lấy trung bình toàn ảnh (global average pooling) của CẢ 3 tầng rồi ghép lại
→ vector 64 + 128 + 256 = 448 chiều. Không dùng lớp conv_fuse chưa huấn luyện như đồ án,
vì khi encoder bị đóng băng thì lớp đó chỉ là phép chiếu ngẫu nhiên.

Encoder được ĐÓNG BĂNG (không học thêm) và chạy trong môi trường, nên replay buffer chỉ phải
lưu vector 448 số thay vì cả ảnh → tiết kiệm RAM rất nhiều cho laptop 16 GB.
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class ResNetEarlySkip(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=weights)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)   # stride 4
        self.layer1, self.layer2, self.layer3 = r.layer1, r.layer2, r.layer3
        self.out_dim = 64 + 128 + 256

    def forward(self, x):
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return torch.cat([f1.mean((2, 3)), f2.mean((2, 3)), f3.mean((2, 3))], 1)


class FrozenImageEncoder:
    """Gọi như hàm: encoder(ảnh RGB uint8 H×W×3) → np.ndarray (448,) float32."""

    def __init__(self, device="auto", pretrained=True):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.net = ResNetEarlySkip(pretrained).to(self.device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.mean, self.std = _MEAN.to(self.device), _STD.to(self.device)
        self.out_dim = self.net.out_dim

    @torch.no_grad()
    def __call__(self, rgb):
        x = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device)
        x = x.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        x = (x - self.mean) / self.std
        return self.net(x)[0].cpu().numpy().astype(np.float32)
