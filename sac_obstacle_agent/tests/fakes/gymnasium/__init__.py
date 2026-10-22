"""Gymnasium giả tối thiểu cho kiểm thử (máy thật dùng gymnasium thật)."""
from . import spaces  # noqa: F401


class Env:
    def reset(self, *, seed=None, options=None):
        return None

    def close(self):
        pass
