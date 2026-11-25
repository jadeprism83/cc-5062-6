"""绘画引擎、图层与跨图层剪切。

笔刷的每一笔先在画布坐标生成自身的覆盖率（soft brush 同样是亚像素
面积覆盖），再与当前选区覆盖率相乘：实际写入 = 笔刷覆盖 × 选区覆盖。
因此选区外像素保持不变，软边按比例写入，绝不把选区阈值化成硬蒙版。

图层为预乘 RGBA float64。跨图层剪切支持两种所有权语义（覆盖率相同
的软边，叠加后归属可能不同）：

- merge_first：先把各源层的预乘颜色按各自 alpha 合成，再整体乘以
  选区覆盖 c（先合并选区/内容，再提取）；
- cut_each：每层先各自乘 c 提取，再按正常 alpha 合成（分别提取再合成）。
  非零 c 与非线性 alpha 合成下两者结果不同，测试中直接对比。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

from .geometry import circle_path
from .rasterize import rasterize_paths
from .selection import FLATTEN_TOL

RGBA = np.ndarray  # HxWx4 float64，预乘 alpha


@dataclass
class Layer:
    name: str
    rgba: RGBA
    visible: bool = True

    @classmethod
    def blank(cls, name: str, height: int, width: int) -> "Layer":
        return cls(name, np.zeros((height, width, 4), dtype=np.float64))


@dataclass
class Canvas:
    height: int
    width: int
    layers: List[Layer] = field(default_factory=list)

    def new_layer(self, name: str) -> Layer:
        layer = Layer.blank(name, self.height, self.width)
        self.layers.append(layer)
        return layer

    def composite(self) -> RGBA:
        """把可见层按顺序合成到透明背景（Porter-Duff over，预乘域）。"""
        acc = np.zeros((self.height, self.width, 4), dtype=np.float64)
        for layer in self.layers:
            if not layer.visible:
                continue
            acc = over(acc, layer.rgba)
        return acc


def over(dst: RGBA, src: RGBA) -> RGBA:
    """预乘 alpha 的 Porter-Duff over 合成。"""
    sa = src[..., 3:4]
    da = dst[..., 3:4]
    out_rgb = src[..., :3] + dst[..., :3] * (1.0 - sa)
    out_a = sa + da * (1.0 - sa)
    return np.concatenate([out_rgb, out_a], axis=-1)


def brush_stamp_coverage(canvas_h: int, canvas_w: int,
                         center: Tuple[float, float], radius: float,
                         hardness: float = 0.5) -> np.ndarray:
    """生成一次笔刷触点的亚像素覆盖率 HxW。

    覆盖率 = 解析圆面积光栅化 × 径向衰减包络：
    - 硬核心（dist <= radius*hardness）内包络为 1，保留精确面积覆盖；
    - 核心到外沿线性衰减到 0；
    - 解析圆保证圆外严格为 0，衰减带宽度在画布坐标定义。
    """
    cx, cy = center
    path = circle_path(cx, cy, radius)
    hard = rasterize_paths(path.flattened(FLATTEN_TOL), canvas_w, canvas_h)
    if hardness >= 1.0:
        return hard
    inner = max(radius * hardness, 0.0)
    yy, xx = np.mgrid[0:canvas_h, 0:canvas_w]
    dist = np.hypot(xx + 0.5 - cx, yy + 0.5 - cy)
    if radius - inner <= 1e-9:
        envelope = (dist <= radius).astype(np.float32)
    else:
        envelope = np.clip((radius - dist) / (radius - inner), 0.0,
                           1.0).astype(np.float32)
    return hard * envelope


def paint_brush(canvas: Canvas, layer: Layer,
                points: Sequence[Tuple[float, float]], radius: float,
                color_rgb: Tuple[float, float, float],
                selection: np.ndarray, spacing: float = 0.25,
                hardness: float = 0.5, opacity: float = 1.0) -> None:
    """沿点序列绘制笔刷，实际写入 = 笔刷覆盖 × 选区覆盖 × opacity。

    selection 为与画布同尺寸的 float 覆盖率（可含羽化）。颜色写入
    预乘 RGBA 层，并与层上已有内容做 over 合成。opacity<1 产生半透明
    笔迹（水彩罩染），不改变选区裁决：选区外 alpha 仍严格为 0。
    """
    h, w = canvas.height, canvas.width
    if selection.shape != (h, w):
        raise ValueError("selection coverage must match canvas size")
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must lie in [0, 1]")
    pts = list(points)
    if not pts:
        return
    stamps = [pts[0]]
    for p in pts[1:]:
        prev = stamps[-1]
        d = math_hypot(p, prev)
        steps = max(1, int(np.ceil(d / max(radius * spacing, 1e-6))))
        for k in range(1, steps + 1):
            t = k / steps
            stamps.append((prev[0] + (p[0] - prev[0]) * t,
                           prev[1] + (p[1] - prev[1]) * t))
    r, g, b = color_rgb
    # 整条笔画在覆盖率域累加（并集：1-∏(1-c)），保证触点重叠处不超过 1；
    # 之后统一乘以选区覆盖与不透明度，使最终写入严格受选区裁决，软边按比例写入。
    stroke_cov = np.zeros((h, w), dtype=np.float64)
    for cx, cy in stamps:
        cov = brush_stamp_coverage(h, w, (cx, cy), radius, hardness).astype(np.float64)
        stroke_cov = stroke_cov + cov - stroke_cov * cov
    write = (stroke_cov * selection * opacity).astype(np.float64)
    src = np.zeros((h, w, 4), dtype=np.float64)
    src[..., 0] = write * r
    src[..., 1] = write * g
    src[..., 2] = write * b
    src[..., 3] = write
    layer.rgba = over(layer.rgba, src)


def math_hypot(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def cut_with_selection(canvas: Canvas, layer_names: Sequence[str],
                       coverage: np.ndarray, mode: str = "merge_first"
                       ) -> Tuple[Layer, List[Layer]]:
    """跨图层剪切。

    返回 (新图层（剪下图块）, 修改后的源层列表（原层留下洞）)。
    mode:
      - "merge_first"：源层先 over 合成，再整体乘覆盖率 c；
      - "cut_each"：每层先乘 c 提取，再把提取结果 over 合成。
    洞的透明度同样由 c 裁决：源层 alpha 乘以 (1-c)。
    """
    h, w = canvas.height, canvas.width
    if coverage.shape != (h, w):
        raise ValueError("coverage must match canvas size")
    c = coverage.astype(np.float64)
    src_layers = [ly for ly in canvas.layers if ly.name in layer_names]

    if mode == "merge_first":
        merged = np.zeros((h, w, 4), dtype=np.float64)
        for ly in src_layers:
            merged = over(merged, ly.rgba)
        cut = merged * c[..., None]
    elif mode == "cut_each":
        cut = np.zeros((h, w, 4), dtype=np.float64)
        for ly in src_layers:
            piece = ly.rgba * c[..., None]
            cut = over(cut, piece)
    else:
        raise ValueError("mode must be merge_first or cut_each")

    # 原层留洞：保留的部分 = 原内容 × (1-c)（预乘域直接缩放）
    for ly in src_layers:
        ly.rgba = ly.rgba * (1.0 - c)[..., None]

    new_layer = Layer("cut(" + mode + ")", cut)
    return new_layer, src_layers


def checkerboard(h: int, w: int, block: int = 16,
                 light=(0.93, 0.93, 0.93), dark=(0.78, 0.78, 0.78)) -> RGBA:
    """棋盘透明底，用于由实际透明度裁决地呈现剪下图块。"""
    yy, xx = np.mgrid[0:h, 0:w]
    odd = ((xx // block) + (yy // block)) % 2
    rgb = np.where(odd[..., None] == 0, np.array(light), np.array(dark))
    out = np.zeros((h, w, 4), dtype=np.float64)
    out[..., :3] = rgb
    out[..., 3] = 1.0
    return out


def composite_over_checker(rgba: RGBA, block: int = 16) -> np.ndarray:
    bg = checkerboard(rgba.shape[0], rgba.shape[1], block)
    comp = over(bg, rgba)
    return np.clip(comp[..., :3], 0.0, 1.0)


def ownership_difference(canvas: Canvas, layer_names: Sequence[str],
                         coverage: np.ndarray) -> Tuple[np.ndarray, float]:
    """两种剪切语义的差异图与最大差值，用于所有权对比演示。"""
    c1 = canvas_copy(canvas)
    c2 = canvas_copy(canvas)
    cut1, _ = cut_with_selection(c1, layer_names, coverage, "merge_first")
    cut2, _ = cut_with_selection(c2, layer_names, coverage, "cut_each")
    diff = np.abs(cut1.rgba - cut2.rgba)
    return diff, float(diff.max())


def canvas_copy(canvas: Canvas) -> Canvas:
    c = Canvas(canvas.height, canvas.width)
    for ly in canvas.layers:
        c.layers.append(Layer(ly.name, ly.rgba.copy(), ly.visible))
    return c
