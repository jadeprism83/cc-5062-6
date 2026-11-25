"""瓦片处理：大画布分块计算并与整幅参考逐像素核对。

光栅化、距离场与组合运算都以 origin 参数表达瓦片在画布坐标中的位置，
因此瓦片结果必须与整幅参考计算逐元素相同（允许 float32 累加顺序的
微小误差）。本模块提供分块渲染/羽化/组合与一致性断言。
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .distance import feather_from_distance, polygon_signed_distance
from .mask import CoverageMask, CombineOp, combine, reassemble_tiles
from .rasterize import rasterize_paths

TILE_TOL = 1e-6


def tiled_render(polys: Sequence[np.ndarray], height: int, width: int,
                 tile_h: int, tile_w: int, feather: float = 0.0
                 ) -> CoverageMask:
    """逐瓦片渲染（含羽化），拼回整幅。"""
    tiles: List[CoverageMask] = []
    for y in range(0, height, tile_h):
        for x in range(0, width, tile_w):
            th = min(tile_h, height - y)
            tw = min(tile_w, width - x)
            hard = rasterize_paths(list(polys), tw, th, origin=(x, y))
            if feather > 0.0:
                sdf = polygon_signed_distance(polys, tw, th, (x, y),
                                              hard_cover=hard)
                cov = feather_from_distance(sdf, feather, hard_cover=hard)
            else:
                cov = hard
            tiles.append(CoverageMask(cov, (x, y)))
    return reassemble_tiles(tiles, (height, width))


def tiled_combine(masks_a: List[CoverageMask], masks_b: List[CoverageMask],
                  full_shape: Tuple[int, int], op: CombineOp) -> CoverageMask:
    """逐瓦片执行组合运算后拼回。"""
    out_tiles = [combine(a, b, op) for a, b in zip(masks_a, masks_b)]
    return reassemble_tiles(out_tiles, full_shape)


def assert_tile_consistency(polys: Sequence[np.ndarray], height: int, width: int,
                            tile_h: int, tile_w: int,
                            feather: float = 0.0) -> float:
    """核对瓦片拼接结果与整幅参考计算，返回最大绝对误差。"""
    full_hard = rasterize_paths(list(polys), width, height)
    if feather > 0.0:
        sdf = polygon_signed_distance(polys, width, height, (0, 0),
                                      hard_cover=full_hard)
        full = feather_from_distance(sdf, feather, hard_cover=full_hard)
    else:
        full = full_hard
    tiled = tiled_render(polys, height, width, tile_h, tile_w, feather).coverage
    err = float(np.abs(full.astype(np.float64) - tiled.astype(np.float64)).max())
    if err > TILE_TOL:
        raise AssertionError(f"tile mismatch: max abs diff {err}")
    return err
