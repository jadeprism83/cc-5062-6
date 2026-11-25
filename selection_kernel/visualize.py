"""覆盖率可视化编码：灰度掩码、亚像素边缘色、PNG/base64。

GUI 与样册导出共用同一套编码，保证界面所见颜色与导出图、证据报告中的
边缘颜色一致；这里不产生任何固定示例数据，只做数组编码。
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image


def png_bytes(arr_uint8: np.ndarray, mode: str) -> bytes:
    img = Image.fromarray(arr_uint8, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def png_b64(arr_uint8: np.ndarray, mode: str) -> str:
    return base64.b64encode(png_bytes(arr_uint8, mode)).decode("ascii")


def coverage_gray_rgba(coverage: np.ndarray) -> np.ndarray:
    """float 覆盖率 -> 不透明灰度 RGBA（保留软边层次，不做阈值化）。"""
    u8 = np.round(np.clip(coverage, 0, 1) * 255).astype(np.uint8)
    rgba = np.zeros((*u8.shape, 4), dtype=np.uint8)
    rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = u8
    rgba[..., 3] = 255
    return rgba


def edge_color_rgba(coverage: np.ndarray) -> np.ndarray:
    """边缘视图：只显示 0<c<1 的亚像素软边，红色通道值=round(c*255)。

    返回的 RGBA 像素颜色因此与覆盖率有逐像素确定的换算关系，证据脚本
    可从导出的 PNG 读回边缘颜色并与手工约定的覆盖率核对。
    """
    soft = ((coverage > 0.0) & (coverage < 1.0))
    vals = np.round(np.clip(coverage, 0, 1) * 255).astype(np.uint8)
    rgba = np.zeros((*coverage.shape, 4), dtype=np.uint8)
    rgba[..., 0] = np.where(soft, vals, 0)
    rgba[..., 3] = np.where(soft, 255, 0)
    return rgba


def edge_expected_rgba(c: float) -> tuple:
    """edge_color_rgba 在覆盖率 c 处应写出的像素值。"""
    if 0.0 < c < 1.0:
        v = int(round(min(1.0, max(0.0, c)) * 255.0))
        return (v, 0, 0, 255)
    return (0, 0, 0, 0)
