"""解析法多边形光栅化：输出每像素真实面积覆盖率（0..1）。

不使用超采样/阈值化。对每个像素行条带 y∈[j,j+1]，取多边形与条带
相交得到的梯形（上下边水平）。条带内若有顶点，则在顶点高度处切开，
使每个子区间内交点配对不变。子区间内多边形横向覆盖宽度 W(y) 为
分段线性函数，面积按梯形公式精确积分：
    area = (W_bottom + W_top) / 2 * (y_top - y_bottom)
每个梯形再按像素列裁剪分摊。

重叠轮廓使用非零环绕规则（non-zero winding）：正向多边形贡献正覆盖、
反向内圈贡献负覆盖，因此带洞轮廓与自交轮廓均可正确处理。
全部计算在画布坐标中进行，与显示分辨率无关。
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

_EPS = 1e-10


def _edge_data(poly: np.ndarray) -> np.ndarray:
    """每条非水平边：(y0, y1, x_at_y0, dx/dy, winding)，保证 y0<y1。

    winding 取整个多边形的环绕符号（有符号面积；y 轴向下时顺时针为
    正填充方向），因此反向内圈整体贡献 -1，配合非零环绕形成洞。
    """
    n = len(poly)
    signed_area = 0.0
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        signed_area += x0 * y1 - x1 * y0
    orientation = 1.0 if signed_area > 0.0 else -1.0
    edges = []
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if y0 == y1:
            continue
        if y0 < y1:
            # 向上的边穿扫描线时环绕 +1，向下的边 -1；整体方向由
            # orientation 翻转，使反向内圈贡献相反符号形成洞
            edges.append((y0, y1, x0, (x1 - x0) / (y1 - y0), orientation))
        else:
            edges.append((y1, y0, x1, (x0 - x1) / (y0 - y1), -orientation))
    return np.asarray(edges, dtype=np.float64).reshape(-1, 5)


def _intersections(edges: np.ndarray, y: float) -> Tuple[np.ndarray, np.ndarray]:
    """水平线 y 与所有边的交点 x 及环绕方向，按 x 稳定排序。"""
    active = edges[(edges[:, 0] <= y) & (edges[:, 1] > y)]
    if len(active) == 0:
        return np.empty(0), np.empty(0)
    xs = active[:, 2] + (y - active[:, 0]) * active[:, 3]
    wind = active[:, 4]
    order = np.argsort(xs, kind="stable")
    return xs[order], wind[order]


def _winding_spans(xs: np.ndarray, wind: np.ndarray) -> List[Tuple[float, float, int]]:
    """标准非零环绕：相邻交点之间，环绕累加值非零即为覆盖区间。

    交点按 x 排序，区间 (xs[i], xs[i+1]) 内的环绕值是前 i+1 个交点
    符号的累加和；非零即覆盖。返回的第三项是该区间的环绕累加值，
    用于同一多边形自交时的带符号面积积分。
    """
    out: List[Tuple[float, float, int]] = []
    w = 0
    for i in range(len(xs) - 1):
        w += int(wind[i])
        if w != 0 and xs[i + 1] > xs[i]:
            out.append((xs[i], xs[i + 1], w))
    return out


def rasterize_polygon(poly: np.ndarray, width: int, height: int,
                      origin: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """单个多边形的带符号面积覆盖率 HxW（float32）。

    填充方向（屏幕顺时针，有符号面积 > 0）输出正覆盖；反向轮廓输出
    负覆盖，供 rasterize_paths 累加形成洞。轮廓内部用非零环绕判定。
    origin 给出掩码 [0,0] 像素对应的画布坐标，支持瓦片与整幅共用同一套计算。
    """
    ox, oy = origin
    cover = np.zeros((height, width), dtype=np.float64)
    if len(poly) < 3:
        return cover.astype(np.float32)
    signed_area = float(
        np.sum(poly[:, 0] * np.roll(poly[:, 1], -1)
               - np.roll(poly[:, 0], -1) * poly[:, 1]))
    sign = 1.0 if signed_area > 0.0 else -1.0
    edges = _edge_data(poly)
    if len(edges) == 0:
        return cover.astype(np.float32)

    j0 = max(0, int(np.floor(edges[:, 0].min() - oy)))
    j1 = min(height, int(np.ceil(edges[:, 1].max() - oy)))

    for j in range(j0, j1):
        yb = oy + j
        yt = yb + 1.0
        active = edges[(edges[:, 0] < yt) & (edges[:, 1] > yb)]
        if len(active) == 0:
            continue
        # 条带内的顶点高度：在这些位置切开，保证子区间内配对关系不变
        cuts = {yb, yt}
        for ey0, ey1, *_ in active:
            if yb < ey0 < yt:
                cuts.add(ey0)
            if yb < ey1 < yt:
                cuts.add(ey1)
        cuts = sorted(cuts)
        for ya, yz in zip(cuts[:-1], cuts[1:]):
            h = yz - ya
            if h <= 0.0:
                continue
            ya_q = yb + _EPS if ya == yb else ya
            yz_q = yt - _EPS if yz == yt else yz
            spans_a = _winding_spans(*_intersections(edges, ya_q))
            spans_z = _winding_spans(*_intersections(edges, yz_q))
            if len(spans_a) != len(spans_z):
                # 数值兜底：在子区间中点求值，按常数宽度处理
                ym = 0.5 * (ya + yz)
                spans_a = spans_z = _winding_spans(*_intersections(edges, ym))
            # 区间内正覆盖积分，再按轮廓整体环绕方向带符号；反向内圈
            # 在 rasterize_paths 层累加时自然扣除洞面积
            for (a0, a1, _), (z0, z1, _) in zip(spans_a, spans_z):
                _accumulate(cover[j], a0 - ox, a1 - ox,
                            z0 - ox, z1 - ox, h, width, sign)
    return cover.astype(np.float32)


def _accumulate(row: np.ndarray, a0: float, a1: float, z0: float, z1: float,
                h: float, width: int, weight: float = 1.0) -> None:
    """把高 h 的梯形（底宽区间 [a0,a1]、顶宽区间 [z0,z1]）精确分摊到像素列。

    weight 为该区间的非零环绕值（洞为 -1），带符号累加后由调用方裁剪。
    """
    lo = min(a0, z0)
    hi = max(a1, z1)
    i0 = max(0, int(np.floor(lo)))
    i1 = min(width - 1, int(np.ceil(hi)) - 1)
    for i in range(i0, i1 + 1):
        xl, xr = float(i), float(i + 1)
        base = max(0.0, min(xr, a1) - max(xl, a0))
        top = max(0.0, min(xr, z1) - max(xl, z0))
        if base == 0.0 and top == 0.0:
            continue
        row[i] += 0.5 * (base + top) * h * weight


def rasterize_paths(paths: List[np.ndarray], width: int, height: int,
                    origin: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """多轮廓非零环绕光栅化，输出 0..1 面积覆盖率 float32 HxW。"""
    cover = np.zeros((height, width), dtype=np.float64)
    for poly in paths:
        cover += rasterize_polygon(poly, width, height, origin)
    return np.clip(cover, 0.0, 1.0).astype(np.float32)
