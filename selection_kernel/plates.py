"""双色套版工程：分色印版、实际油墨覆盖回读与套印校样。

印版文件是 8-bit 灰度 PNG，像素值 0..255 对应油墨覆盖 0..1（255=实地
油墨）。核验细丝/空孔时不使用内存数组，而是把印版写盘后再用
load_plate_ink 从实际导出文件读回——“换版后从实际油墨覆盖读出细丝”。

套印校样用减法混色近似：每层油墨把纸白通道按 1-c(1-ink_rgb) 压暗，
两版重叠处得到叠印深色。另出黑油墨菲林阳片（黑=油墨）与带规线的贴版图。
"""
from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .samplebook import roi_stats

INK_COLORS = {
    "vermilion": (0.86, 0.24, 0.10),   # 朱红
    "indigo": (0.08, 0.27, 0.72),      # 靛蓝
    "amber": (0.95, 0.62, 0.08),
    "black": (0.05, 0.05, 0.06),
}


def save_plate_ink_png(coverage: np.ndarray, path: str) -> None:
    """保存印版灰度 PNG：数值即油墨覆盖（255=实地）。"""
    u8 = np.round(np.clip(coverage, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(u8, mode="L").save(path)


def load_plate_ink(path: str) -> np.ndarray:
    """从磁盘上的实际印版 PNG 读回油墨覆盖 c=gray/255。"""
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def save_film_positive(coverage: np.ndarray, path: str) -> None:
    """菲林阳片：油墨处为黑、无墨处为白，可直接用于冲印/晒版参考。"""
    u8 = np.round((1.0 - np.clip(coverage, 0, 1)) * 255).astype(np.uint8)
    Image.fromarray(u8, mode="L").save(path)


def overprint_proof(plates: Sequence[Tuple[np.ndarray, Tuple[float, float, float]]],
                    paper: Tuple[float, float, float] = (1.0, 1.0, 0.99)
                    ) -> np.ndarray:
    """多色版减法叠印校样，返回 float RGB HxWx3。"""
    rgb = np.broadcast_to(np.array(paper, dtype=np.float64),
                          plates[0][0].shape + (3,)).copy()
    for coverage, ink in plates:
        ink_arr = np.array(ink, dtype=np.float64)
        rgb = rgb * (1.0 - np.clip(coverage, 0, 1)[..., None] * (1.0 - ink_arr))
    return np.clip(rgb, 0.0, 1.0)


def save_rgb(rgb: np.ndarray, path: str) -> None:
    Image.fromarray(np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8),
                    mode="RGB").save(path)


def plate_sheet(coverage: np.ndarray, title: str,
                ink_rgb: Tuple[float, float, float],
                margin: int = 26) -> np.ndarray:
    """加页边、角规线与色标的贴版预览（RGB uint8），两版同尺寸可对齐套印。"""
    h, w = coverage.shape
    sheet = np.ones((h + 2 * margin, w + 2 * margin, 3), dtype=np.uint8) * 255
    plate_u8 = np.round(np.clip(coverage, 0, 1) * 255).astype(np.uint8)
    tint = (1.0 - plate_u8.astype(np.float32) / 255.0)[..., None]
    color = 1.0 - tint * (1.0 - np.array(ink_rgb, dtype=np.float32))
    sheet[margin:margin + h, margin:margin + w] = np.round(color * 255)
    img = Image.fromarray(sheet, "RGB")
    d = ImageDraw.Draw(img)
    L = 14
    for cx, cy in [(margin // 2, margin // 2),
                   (w + margin + margin // 2, margin // 2),
                   (margin // 2, h + margin + margin // 2),
                   (w + margin + margin // 2, h + margin + margin // 2)]:
        d.line([cx - L, cy, cx + L, cy], fill=(20, 20, 20), width=1)
        d.line([cx, cy - L, cx, cy + L], fill=(20, 20, 20), width=1)
    for (x0, y0), (x1, y1) in [
        ((margin, margin), (margin + 12, margin)),
        ((margin, margin), (margin, margin + 12)),
        ((w + margin, margin), (w + margin - 12, margin)),
        ((w + margin, margin), (w + margin, margin + 12)),
        ((margin, h + margin), (margin + 12, h + margin)),
        ((margin, h + margin), (margin, h + margin - 12)),
        ((w + margin, h + margin), (w + margin - 12, h + margin)),
        ((w + margin, h + margin), (w + margin, h + margin - 12)),
    ]:
        d.line([x0, y0, x1, y1], fill=(20, 20, 20), width=1)
    d.rectangle([margin, 4, margin + 12, 16],
                fill=tuple(int(round(v * 255)) for v in ink_rgb),
                outline=(20, 20, 20))
    d.text((margin + 18, 6), title, fill=(20, 20, 20))
    return np.asarray(img)


def fill_enclosed_holes(coverage: np.ndarray, threshold: float = 0.5
                        ) -> np.ndarray:
    """故意做错的坏版对照：二值化后把所有封闭孔洞补满。

    用边界泛洪找出与画布边缘连通的背景；其余 0 值封闭区一律填 1。
    核验程序必须能在这种对照上报出“空孔被错误补满”，证明检查有牙齿。
    """
    binary = coverage >= threshold
    h, w = binary.shape
    filled = binary.copy()
    from collections import deque
    seen = np.zeros_like(binary, dtype=bool)
    q: deque = deque()
    for x in range(w):
        for y in (0, h - 1):
            if not binary[y, x] and not seen[y, x]:
                seen[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if not binary[y, x] and not seen[y, x]:
                seen[y, x] = True
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not binary[ny, nx] \
                    and not seen[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    holes = (~binary) & (~seen)
    filled[holes] = True
    return filled.astype(np.float32)


def verify_filament_and_hole(ink: np.ndarray,
                             filament_roi: Sequence[int],
                             hole_roi: Sequence[int],
                             scale: float = 1.0,
                             filament_peak_min: float = 0.10,
                             hole_max_tol: float = 0.02
                             ) -> Dict[str, object]:
    """从实际油墨覆盖读数核验细丝存在、空孔未被补满。"""
    fil = roi_stats(ink, filament_roi, scale)
    hole = roi_stats(ink, hole_roi, scale)
    filament_present = fil["max"] >= filament_peak_min
    hole_empty = hole["max"] <= hole_max_tol
    return {
        "filament": fil,
        "hole": hole,
        "filament_peak_min": filament_peak_min,
        "hole_max_tol": hole_max_tol,
        "filament_present": bool(filament_present),
        "hole_empty": bool(hole_empty),
        "pass": bool(filament_present and hole_empty),
    }
