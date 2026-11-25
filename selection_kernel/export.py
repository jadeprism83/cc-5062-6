"""导出：掩码（16-bit PNG 保留细边）、绘画 RGBA、棋盘底合成图。

内部覆盖率始终是 float32；只有在最终导出时才量化。掩码默认 16-bit
PNG（I;16），把抗锯齿细边的量化误差压到 1/65535，避免 8-bit 阈值化
丢失细边。导出按画布全分辨率进行，绝不使用显示分辨率蒙版。
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .painting import RGBA, composite_over_checker


def coverage_to_u16(coverage: np.ndarray) -> np.ndarray:
    c = np.clip(coverage.astype(np.float64), 0.0, 1.0)
    return np.round(c * 65535.0).astype(np.uint16)


def save_mask_png(coverage: np.ndarray, path: str) -> None:
    arr = coverage_to_u16(coverage)
    img = Image.new("I;16", (arr.shape[1], arr.shape[0]))
    img.putdata(arr.flatten())
    img.save(path)


def save_mask_u8_png(coverage: np.ndarray, path: str) -> None:
    c = np.clip(coverage.astype(np.float64), 0.0, 1.0)
    Image.fromarray(np.round(c * 255.0).astype(np.uint8), mode="L").save(path)


def save_rgba(rgba: RGBA, path: str) -> None:
    a = np.clip(rgba, 0.0, 1.0)
    u8 = np.round(a * 255.0).astype(np.uint8)
    Image.fromarray(u8, mode="RGBA").save(path)


def save_on_checker(rgba: RGBA, path: str, block: int = 16) -> None:
    rgb = composite_over_checker(rgba, block)
    u8 = np.round(rgb * 255.0).astype(np.uint8)
    Image.fromarray(u8, mode="RGB").save(path)


def save_rgb(rgb: np.ndarray, path: str) -> None:
    u8 = np.round(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(u8, mode="RGB").save(path)


def load_image_rgba(path: str) -> RGBA:
    img = Image.open(path).convert("RGBA")
    return (np.asarray(img, dtype=np.float64) / 255.0)
