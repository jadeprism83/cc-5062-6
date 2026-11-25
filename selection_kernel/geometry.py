"""曲线轮廓几何：贝塞尔路径、自适应展平、仿射变换。

轮廓用 Path 表示，Path 由若干 SubPath 组成；每个 SubPath 是一串
Bezier 段（MOVETO/LINETO/CURVETO/CLOSEPOLY）。坐标一律使用画布坐标
（浮点，单位为高分辨率图像像素），与界面显示分辨率无关。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

Point = Tuple[float, float]

_MOVE = 0
_LINE = 1
_CURVE = 2
_CLOSE = 3


@dataclass
class Path:
    """一条或多条子路径组成的轮廓。"""

    subpaths: List["SubPath"] = field(default_factory=list)

    def flattened(self, tolerance: float = 0.01) -> List[np.ndarray]:
        """把所有子路径展平为闭合多边形（画布坐标浮点数组 Nx2）。"""
        return [sp.flatten(tolerance) for sp in self.subpaths]

    def transformed(self, matrix: np.ndarray) -> "Path":
        """施加 3x3 仿射（或相似/刚性）变换，返回新 Path。"""
        out = Path()
        for sp in self.subpaths:
            nsp = SubPath()
            for cmd, pts in sp.segments:
                tpts = [_apply(matrix, p) for p in pts]
                nsp.segments.append((cmd, tpts))
            out.subpaths.append(nsp)
        return out

    def bounds(self) -> Tuple[float, float, float, float]:
        xs, ys = [], []
        for poly in self.flattened():
            xs.append(poly[:, 0].min())
            xs.append(poly[:, 0].max())
            ys.append(poly[:, 1].min())
            ys.append(poly[:, 1].max())
        return min(xs), min(ys), max(xs), max(ys)


@dataclass
class SubPath:
    # 每段：(命令, 点序列)。MOVE/LINE 一个点，CURVE 三个点（含两个控制点和终点）
    segments: List[Tuple[int, List[Point]]] = field(default_factory=list)

    def move_to(self, p: Point) -> None:
        self.segments.append((_MOVE, [p]))

    def line_to(self, p: Point) -> None:
        self.segments.append((_LINE, [p]))

    def curve_to(self, c1: Point, c2: Point, p: Point) -> None:
        self.segments.append((_CURVE, [c1, c2, p]))

    def close(self) -> None:
        self.segments.append((_CLOSE, []))

    def flatten(self, tolerance: float = 0.01) -> np.ndarray:
        """自适应展平为闭合多边形顶点数组。"""
        pts: List[Point] = []
        start: Point | None = None
        cur: Point | None = None
        for cmd, data in self.segments:
            if cmd == _MOVE:
                start = data[0]
                cur = start
                pts.append(cur)
            elif cmd == _LINE:
                cur = data[0]
                pts.append(cur)
            elif cmd == _CURVE:
                assert cur is not None
                c1, c2, end = data
                depth = _flatten_depth(cur, c1, c2, end, tolerance)
                _recursive_flatten(cur, c1, c2, end, depth, pts)
                cur = end
            elif cmd == _CLOSE:
                if cur is not None and start is not None and cur != start:
                    pts.append(start)
                cur = start
        return np.asarray(pts, dtype=np.float64)


def _apply(matrix: np.ndarray, p: Point) -> Point:
    x, y = p
    v = matrix @ np.array([x, y, 1.0])
    return float(v[0]), float(v[1])


def rotation_matrix(angle_rad: float, center: Point) -> np.ndarray:
    """绕 center 旋转 angle_rad 的 3x3 矩阵。"""
    cx, cy = center
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    t1 = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]])
    r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    t2 = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]])
    return t2 @ r @ t1


def _flatten_depth(p0: Point, p1: Point, p2: Point, p3: Point,
                   tolerance: float) -> int:
    # 用控制点到弦的偏差估计展平深度（与常用矢量光栅化器一致）
    d1 = _point_line_distance(p1, p0, p3)
    d2 = _point_line_distance(p2, p0, p3)
    err = max(d1, d2)
    if err <= tolerance:
        return 0
    depth = int(math.ceil(math.log(err / tolerance) / math.log(4.0)))
    return max(1, min(depth, 18))


def _recursive_flatten(p0, p1, p2, p3, depth: int, out: List[Point]) -> None:
    if depth <= 0:
        out.append(p3)
        return
    m01 = _mid(p0, p1)
    m12 = _mid(p1, p2)
    m23 = _mid(p2, p3)
    a = _mid(m01, m12)
    b = _mid(m12, m23)
    c = _mid(a, b)
    _recursive_flatten(p0, m01, a, c, depth - 1, out)
    _recursive_flatten(c, b, m23, p3, depth - 1, out)


def _mid(a: Point, b: Point) -> Point:
    return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)


def _point_line_distance(p: Point, a: Point, b: Point) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom == 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / denom
    t = max(0.0, min(1.0, t))
    qx, qy = ax + t * dx, ay + t * dy
    return math.hypot(px - qx, py - qy)


def circle_path(cx: float, cy: float, r: float, segments: int = 96) -> Path:
    """用四段三次贝塞尔逼近圆（kappa 常数），构造曲线轮廓。"""
    k = 0.5522847498307936
    sp = SubPath()
    sp.move_to((cx + r, cy))
    sp.curve_to((cx + r, cy + k * r), (cx + k * r, cy + r), (cx, cy + r))
    sp.curve_to((cx - k * r, cy + r), (cx - r, cy + k * r), (cx - r, cy))
    sp.curve_to((cx - r, cy - k * r), (cx - k * r, cy - r), (cx, cy - r))
    sp.curve_to((cx + k * r, cy - r), (cx + r, cy - k * r), (cx + r, cy))
    sp.close()
    return Path([sp])


def rectangle_path(x: float, y: float, w: float, h: float) -> Path:
    sp = SubPath()
    sp.move_to((x, y))
    sp.line_to((x + w, y))
    sp.line_to((x + w, y + h))
    sp.line_to((x, y + h))
    sp.close()
    return Path([sp])


def slanted_path(x: float, y: float, w: float, h: float, shear: float) -> Path:
    """斜边（平行四边形）轮廓，用于检查斜边上的亚像素覆盖。"""
    sp = SubPath()
    sp.move_to((x + shear, y))
    sp.line_to((x + w + shear, y))
    sp.line_to((x + w, y + h))
    sp.line_to((x, y + h))
    sp.close()
    return Path([sp])


def thin_slit_path(cx: float, cy: float, width: float, height: float) -> Path:
    """极细缝：宽度小于一个像素的矩形轮廓。"""
    return rectangle_path(cx - width / 2, cy - height / 2, width, height)
