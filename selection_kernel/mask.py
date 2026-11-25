"""覆盖率掩码与选区组合运算。

覆盖率 c∈[0,1] 表示像素面积中属于选区的比例，不是布尔真假值。
组合运算在连续覆盖率上定义，全程不做阈值化：

- 合并 UNION：假设两个选区的覆盖区域统计独立，
  c = 1 - (1-a)(1-b)；同一选区重复合并不改变自身（幂等由模型保证）。
- 相减 SUBTRACT：c = a * (1-b)，即 A 的覆盖中未被 B 占有的部分。
- 相交 INTERSECT：c = a * b，两处覆盖共同占有的面积比例。
- 异或 XOR：c = a + b - 2ab。

所有运算逐像素独立，因此任意瓦片上的结果与整幅参考计算逐元素一致。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Tuple

import numpy as np


class CombineOp(str, Enum):
    UNION = "union"
    SUBTRACT = "subtract"
    INTERSECT = "intersect"
    XOR = "xor"


@dataclass
class CoverageMask:
    """画布坐标域上的 float32 覆盖率瓦片或整幅掩码。"""

    coverage: np.ndarray
    origin: Tuple[int, int] = (0, 0)  # [0,0] 像素对应的画布坐标

    def __post_init__(self) -> None:
        if self.coverage.dtype != np.float32:
            self.coverage = self.coverage.astype(np.float32)
        if np.min(self.coverage) < -1e-6 or np.max(self.coverage) > 1.0 + 1e-6:
            raise ValueError("coverage must lie within [0, 1]")
        np.clip(self.coverage, 0.0, 1.0, out=self.coverage)

    @property
    def height(self) -> int:
        return self.coverage.shape[0]

    @property
    def width(self) -> int:
        return self.coverage.shape[1]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.coverage.shape

    def copy(self) -> "CoverageMask":
        return CoverageMask(self.coverage.copy(), self.origin)

    def tiles(self, tile_h: int, tile_w: int) -> Iterator["CoverageMask"]:
        """按规则网格切瓦片；瓦片保留各自画布原点。"""
        h, w = self.coverage.shape
        for y in range(0, h, tile_h):
            for x in range(0, w, tile_w):
                block = self.coverage[y:y + tile_h, x:x + tile_w]
                yield CoverageMask(block, (self.origin[0] + x, self.origin[1] + y))

    def to_alpha_u8(self) -> np.ndarray:
        """导出为 8-bit alpha；仅在最终导出时量化，内部始终保留 float。"""
        return np.round(self.coverage * 255.0).astype(np.uint8)


def _aligned(a: CoverageMask, b: CoverageMask) -> Tuple[np.ndarray, np.ndarray]:
    if a.origin != b.origin or a.shape != b.shape:
        raise ValueError(
            f"mask domains differ: {a.origin}/{a.shape} vs {b.origin}/{b.shape}")
    return a.coverage, b.coverage


def combine(a: CoverageMask, b: CoverageMask, op: CombineOp) -> CoverageMask:
    """连续覆盖率组合运算，返回同坐标域的新掩码。"""
    ca, cb = _aligned(a, b)
    if op == CombineOp.UNION:
        out = ca + cb - ca * cb
    elif op == CombineOp.SUBTRACT:
        out = ca * (1.0 - cb)
    elif op == CombineOp.INTERSECT:
        out = ca * cb
    elif op == CombineOp.XOR:
        out = ca + cb - 2.0 * ca * cb
    else:
        raise ValueError(f"unknown op {op}")
    return CoverageMask(np.clip(out, 0.0, 1.0).astype(np.float32), a.origin)


def merge_masks(masks, op: CombineOp) -> CoverageMask:
    """对同一坐标域的掩码序列依次组合。"""
    masks = list(masks)
    if not masks:
        raise ValueError("need at least one mask")
    acc = masks[0].copy()
    for m in masks[1:]:
        acc = combine(acc, m, op)
    return acc


def reassemble_tiles(tiles, full_shape, full_origin=(0, 0)) -> CoverageMask:
    """把带原点的瓦片拼回整幅掩码（用于校验瓦片/整幅一致性）。"""
    out = np.zeros(full_shape, dtype=np.float32)
    ox, oy = full_origin
    for t in tiles:
        th, tw = t.shape
        x0 = t.origin[0] - ox
        y0 = t.origin[1] - oy
        out[y0:y0 + th, x0:x0 + tw] = t.coverage
    return CoverageMask(out, full_origin)
