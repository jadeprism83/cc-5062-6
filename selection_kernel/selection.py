"""精细选区模型：保留坐标域、矢量轮廓与操作次序。

设计要点：
- 画布尺寸（高分辨率图像坐标）是固定坐标域，界面缩放只改变显示。
- 每个选区记录矢量 Path（可重新精确光栅化）、当前 float 覆盖率、
  以及操作日志（操作次序是结果的一部分，不可重排）。
- “先旋转再羽化”与“先羽化再旋转”是两条不同的操作序列：
  * rotate 作用于矢量轮廓并以解析覆盖率重新光栅化（硬边精度不退化）；
  * feather 作用于当前覆盖率对应的距离场；
  顺序不同，边缘结果不同，测试中分别核查。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .distance import feather_from_distance, polygon_signed_distance
from .geometry import Path, rotation_matrix
from .mask import CoverageMask, CombineOp, combine
from .rasterize import rasterize_paths

FLATTEN_TOL = 0.01  # 画布像素，贝塞尔展平容差


@dataclass
class Selection:
    """一个命名选区：矢量轮廓 + 当前覆盖率 + 操作日志。"""

    name: str
    canvas_size: Tuple[int, int]
    paths: List[Path]
    origin: Tuple[int, int] = (0, 0)
    feather_radius: float = 0.0
    history: List[str] = field(default_factory=list)
    _coverage: Optional[CoverageMask] = None

    @property
    def height(self) -> int:
        return self.canvas_size[0]

    def render_hard(self) -> CoverageMask:
        """从矢量轮廓重新解析光栅化硬边覆盖率（无阈值化）。"""
        h, w = self.canvas_size
        polys = [poly for path in self.paths for poly in path.flattened(FLATTEN_TOL)]
        cov = rasterize_paths(polys, w, h, self.origin)
        return CoverageMask(cov, self.origin)

    def render(self) -> CoverageMask:
        """硬边 + 当前羽化半径的最终覆盖率。"""
        hard = self.render_hard()
        if self.feather_radius <= 0.0:
            return hard
        polys = [poly for path in self.paths for poly in path.flattened(FLATTEN_TOL)]
        sdf = polygon_signed_distance(polys, hard.width, hard.height,
                                      self.origin, hard_cover=hard.coverage)
        return CoverageMask(feather_from_distance(sdf, self.feather_radius,
                                                  hard_cover=hard.coverage),
                            self.origin)

    @property
    def coverage(self) -> CoverageMask:
        if self._coverage is None:
            self._coverage = self.render()
        return self._coverage

    def invalidate(self) -> None:
        self._coverage = None

    def rotate(self, angle_rad: float, center: Optional[Tuple[float, float]] = None) -> "Selection":
        """旋转矢量轮廓（保留曲线），硬边重新解析光栅化，羽化半径归零待重设。"""
        if center is None:
            x0, y0, x1, y1 = self._bounds()
            center = (0.5 * (x0 + x1), 0.5 * (y0 + y1))
        m = rotation_matrix(angle_rad, center)
        new_paths = [p.transformed(m) for p in self.paths]
        hist = self.history + [f"rotate({math.degrees(angle_rad):.3f}deg)"]
        return Selection(self.name, self.canvas_size, new_paths, self.origin,
                         feather_radius=0.0, history=hist)

    def set_feather(self, radius: float) -> "Selection":
        """对（当前硬边）轮廓施加画布坐标羽化半径。"""
        if radius < 0.0:
            raise ValueError("feather radius must be >= 0")
        hist = self.history + [f"feather({radius:.4f}px)"]
        return Selection(self.name, self.canvas_size, self.paths, self.origin,
                         feather_radius=radius, history=hist)

    def _bounds(self) -> Tuple[float, float, float, float]:
        boxes = [p.bounds() for p in self.paths]
        return (min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes))


@dataclass
class SelectionDocument:
    """整张画布的选区集合与组合结果，固定坐标域。"""

    canvas_size: Tuple[int, int]  # (height, width)，高分辨率图像像素
    selections: List[Selection] = field(default_factory=list)
    history: List[str] = field(default_factory=list)

    def add(self, sel: Selection) -> None:
        if sel.canvas_size != self.canvas_size:
            raise ValueError("selection canvas_size must match document domain")
        self.selections.append(sel)
        self.history.append(f"add:{sel.name}")

    def combined(self, ops: Optional[List[CombineOp]] = None) -> CoverageMask:
        """按给定操作次序组合所有选区；默认全部 UNION。"""
        if not self.selections:
            h, w = self.canvas_size
            return CoverageMask(np.zeros((h, w), np.float32))
        if ops is None:
            ops = [CombineOp.UNION] * (len(self.selections) - 1)
        if len(ops) != len(self.selections) - 1:
            raise ValueError("need one op between each pair of selections")
        acc = self.selections[0].coverage.copy()
        for sel, op in zip(self.selections[1:], ops):
            acc = combine(acc, sel.coverage, op)
        return acc

    def apply_operation(self, name: str, *args) -> None:
        """记录文档级操作（组合、剪切等），保持操作次序可追溯。"""
        self.history.append(name + "(" + ", ".join(str(a) for a in args) + ")")
