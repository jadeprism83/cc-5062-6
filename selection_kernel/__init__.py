"""精细绘画选区内核（subpixel selection kernel）。

模块组成：
- geometry: 贝塞尔曲线轮廓与仿射变换（画布坐标）
- rasterize: 解析法多边形面积覆盖率光栅化（无超采样、无阈值化）
- distance: 带符号距离场与画布坐标羽化
- mask: 连续覆盖率组合运算（合并/相减/相交/异或）与瓦片
- selection: 选区模型，保留坐标域与操作次序
- painting: 选区限制的笔刷、图层、跨图层剪切与棋盘底
- tiling: 瓦片处理与整幅参考一致性
- export: 16-bit 掩码/绘画 RGBA/棋盘底导出
"""
from .geometry import Path, SubPath, circle_path, rectangle_path, slanted_path, thin_slit_path, rotation_matrix
from .rasterize import rasterize_paths, rasterize_polygon
from .distance import polygon_signed_distance, feather_from_distance, rasterize_feathered
from .mask import CoverageMask, CombineOp, combine, merge_masks, reassemble_tiles
from .selection import Selection, SelectionDocument, FLATTEN_TOL
from .painting import Canvas, Layer, brush_stamp_coverage, paint_brush, cut_with_selection, checkerboard, composite_over_checker, ownership_difference
from .tiling import tiled_render, assert_tile_consistency
from . import export
from . import pathio
from . import visualize
from . import samplebook
from . import plates

__all__ = [
    "Path", "SubPath", "circle_path", "rectangle_path", "slanted_path",
    "thin_slit_path", "rotation_matrix",
    "rasterize_paths", "rasterize_polygon",
    "polygon_signed_distance", "feather_from_distance", "rasterize_feathered",
    "CoverageMask", "CombineOp", "combine", "merge_masks", "reassemble_tiles",
    "Selection", "SelectionDocument", "FLATTEN_TOL",
    "Canvas", "Layer", "brush_stamp_coverage", "paint_brush",
    "cut_with_selection", "checkerboard", "composite_over_checker",
    "ownership_difference",
    "tiled_render", "assert_tile_consistency",
    "export", "pathio", "visualize", "samplebook", "plates",
]
