"""带符号距离场与可控羽化。

距离以画布坐标（图像像素）定义：羽化半径 r 意味着边界内外各 r 个
图像像素的线性过渡，界面缩放不改变其物理宽度。

为保证瓦片边缘与整幅参考计算完全一致，距离场直接从矢量轮廓重算
（不依赖上一次光栅化结果），瓦片只改变采样原点，数学公式不变。
硬边覆盖率本身是解析面积覆盖率，由它反推的距离在离边一个像素内
做亚像素校正，避免“先阈值化再羽化”丢失细边。
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from .rasterize import rasterize_paths

_EPS = 1e-9


def polygon_signed_distance(polys: Sequence[np.ndarray], width: int, height: int,
                            origin: Tuple[float, float] = (0.0, 0.0),
                            hard_cover: Optional[np.ndarray] = None) -> np.ndarray:
    """每像素中心到轮廓边界的带符号距离（负=内部），float32 HxW。

    若提供 hard_cover（解析硬边覆盖率），用它在距边界约 1 像素内做
    亚像素符号/距离校正：覆盖 0.5 对应距离 0，覆盖 c 对应距离
    d ≈ 0.5 - c（局部直线边界下精确）。
    """
    ox, oy = origin
    yy, xx = np.mgrid[0:height, 0:width]
    px = xx.astype(np.float64) + 0.5 + ox
    py = yy.astype(np.float64) + 0.5 + oy

    dmin = np.full((height, width), np.inf, dtype=np.float64)
    for poly in polys:
        n = len(poly)
        if n < 3:
            continue
        for i in range(n):
            ax, ay = poly[i]
            bx, by = poly[(i + 1) % n]
            ex, ey = bx - ax, by - ay
            length2 = ex * ex + ey * ey
            if length2 < _EPS:
                continue
            t = ((px - ax) * ex + (py - ay) * ey) / length2
            t = np.clip(t, 0.0, 1.0)
            qx = ax + t * ex
            qy = ay + t * ey
            d = np.hypot(px - qx, py - qy)
            np.minimum(dmin, d, out=dmin)
    if not np.isfinite(dmin).any():
        dmin = np.full((height, width), 1e6, dtype=np.float64)

    if hard_cover is None:
        hard_cover = rasterize_paths(list(polys), width, height, origin)
    inside = hard_cover > 0.5
    sdf = np.where(inside, -dmin, dmin)

    # 亚像素校正：边界像素用覆盖率反推距离，保留抗锯齿细边
    near = (hard_cover > 0.0) & (hard_cover < 1.0)
    sdf = np.where(near, 0.5 - hard_cover.astype(np.float64), sdf)
    return sdf.astype(np.float32)


def feather_from_distance(sdf: np.ndarray, radius: float,
                          hard_cover: np.ndarray | None = None) -> np.ndarray:
    """带符号距离 -> 线性羽化覆盖率。

    radius 为画布坐标下的羽化半径（单边宽度）。radius=0 时返回解析
    硬边覆盖率（若提供 hard_cover），即面积覆盖本身——它可能包含
    0<c<1 的抗锯齿细边，而不是二值蒙版；未提供硬边时退化为距离符号。
    过渡函数为分段线性，语义明确可复现。
    """
    if radius <= 0.0:
        if hard_cover is not None:
            return np.clip(hard_cover, 0.0, 1.0).astype(np.float32)
        return (sdf <= 0.0).astype(np.float32)
    c = 0.5 - sdf / (2.0 * radius)
    return np.clip(c, 0.0, 1.0).astype(np.float32)


def rasterize_feathered(polys: Sequence[np.ndarray], width: int, height: int,
                        radius: float,
                        origin: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """矢量轮廓 -> 硬边解析覆盖 -> 距离场 -> 羽化覆盖，全链路无阈值化。"""
    hard = rasterize_paths(list(polys), width, height, origin)
    sdf = polygon_signed_distance(polys, width, height, origin, hard_cover=hard)
    return feather_from_distance(sdf, radius, hard_cover=hard)
