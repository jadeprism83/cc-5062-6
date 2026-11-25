"""命令行演示：用精细选区内核完成完整工作流并导出真实产物。

输出（默认到 output/）：
- mask_hard.png / mask_feather.png：16-bit 覆盖率掩码
- paint_rgba.png：受选区限制的真实笔刷绘画（RGBA）
- paint_on_checker.png：棋盘底上的绘画（实际透明度裁决）
- cut_merge_first.png / cut_each.png：两种跨图层剪切所有权语义
- ownership_diff.png：两种语义的差异放大图
- tiles_check.txt：瓦片与整幅参考一致性数值

用法：
  python3 examples/demo_pipeline.py --width 220 --height 200 --feather 4.0
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from selection_kernel import (  # noqa: E402
    Canvas, CombineOp, Selection, assert_tile_consistency,
    circle_path, cut_with_selection, export, geometry,
    ownership_difference, paint_brush, rectangle_path,
)
from selection_kernel.distance import (  # noqa: E402
    feather_from_distance, polygon_signed_distance,
)
from selection_kernel.rasterize import rasterize_paths  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=220)
    ap.add_argument("--height", type=int, default=200)
    ap.add_argument("--feather", type=float, default=4.0)
    ap.add_argument("--outdir", default="output")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    h, w = args.height, args.width

    # 1) 曲线轮廓选区（圆 + 相减矩形），全画布坐标、亚像素解析覆盖
    circle = circle_path(w * 0.45, h * 0.5, min(h, w) * 0.32)
    notch = rectangle_path(w * 0.62, h * 0.2, w * 0.25, h * 0.28)
    sel_circle = Selection("circle", (h, w), [circle])
    sel_notch = Selection("notch", (h, w), [notch])
    from selection_kernel import SelectionDocument
    document = SelectionDocument((h, w))
    document.add(sel_circle)
    document.add(sel_notch)
    hard = document.combined([CombineOp.SUBTRACT])

    # 2) 羽化（画布坐标半径），全程不阈值化
    polys = circle.flattened() + notch.flattened()
    sdf = polygon_signed_distance(polys, w, h, hard_cover=hard.coverage)
    feathered = feather_from_distance(sdf, args.feather, hard_cover=hard.coverage)

    export.save_mask_png(hard.coverage, os.path.join(args.outdir, "mask_hard.png"))
    export.save_mask_png(feathered, os.path.join(args.outdir, "mask_feather.png"))
    export.save_mask_u8_png(feathered, os.path.join(args.outdir, "mask_feather_u8.png"))

    # 3) 真实绘画：笔刷写入 = 笔刷覆盖 × 羽化选区覆盖
    canvas = Canvas(h, w)
    layer = canvas.new_layer("watercolor")
    pts = [(w * 0.18 + t * w * 0.6, h * 0.5 + 18.0 * np.sin(t * 6.283))
           for t in np.linspace(0.0, 1.0, 40)]
    paint_brush(canvas, layer, pts, radius=min(h, w) * 0.09,
                color_rgb=(0.85, 0.25, 0.15), selection=feathered,
                hardness=0.6)
    export.save_rgba(layer.rgba, os.path.join(args.outdir, "paint_rgba.png"))
    export.save_on_checker(layer.rgba, os.path.join(args.outdir, "paint_on_checker.png"))

    # 4) 跨图层剪切：两种所有权语义 + 差异
    l2 = canvas.new_layer("wash")
    sel2 = rasterize_paths(circle_path(w * 0.55, h * 0.55,
                                       min(h, w) * 0.22).flattened(), w, h)
    paint_brush(canvas, l2, [(w * 0.55, h * 0.55)], radius=min(h, w) * 0.22,
                color_rgb=(0.1, 0.35, 0.85), selection=sel2, hardness=0.4)

    cut_sel = rasterize_paths(
        rectangle_path(w * 0.3, h * 0.3, w * 0.4, h * 0.4).flattened(), w, h)
    cut_sdf = polygon_signed_distance(
        rectangle_path(w * 0.3, h * 0.3, w * 0.4, h * 0.4).flattened(),
        w, h, hard_cover=cut_sel)
    cut_cov = feather_from_distance(cut_sdf, 3.0, hard_cover=cut_sel)

    c1 = _clone(canvas)
    cut1, _ = cut_with_selection(c1, ["watercolor", "wash"], cut_cov, "merge_first")
    c2 = _clone(canvas)
    cut2, _ = cut_with_selection(c2, ["watercolor", "wash"], cut_cov, "cut_each")
    export.save_on_checker(cut1.rgba, os.path.join(args.outdir, "cut_merge_first.png"))
    export.save_on_checker(cut2.rgba, os.path.join(args.outdir, "cut_each.png"))
    diff_img, max_diff = ownership_difference(canvas, ["watercolor", "wash"], cut_cov)
    diff_vis = np.clip(diff_img[..., :3] * 8.0, 0.0, 1.0)
    export.save_rgb(diff_vis, os.path.join(args.outdir, "ownership_diff.png"))

    # 5) 瓦片一致性（非整除瓦片尺寸 + 羽化）
    err_hard = assert_tile_consistency(polys, h, w, 37, 41, feather=0.0)
    err_feather = assert_tile_consistency(polys, h, w, 37, 41, feather=args.feather)
    with open(os.path.join(args.outdir, "tiles_check.txt"), "w") as f:
        f.write(f"tile max abs diff (hard):    {err_hard:.3e}\n")
        f.write(f"tile max abs diff (feather): {err_feather:.3e}\n")
        f.write(f"ownership max abs diff:      {max_diff:.6f}\n")

    print("exports written to", os.path.abspath(args.outdir))
    for name in sorted(os.listdir(args.outdir)):
        print(" -", name)
    return 0


def _clone(canvas: Canvas) -> Canvas:
    from selection_kernel.painting import Layer
    c = Canvas(canvas.height, canvas.width)
    for ly in canvas.layers:
        c.layers.append(Layer(ly.name, ly.rgba.copy(), ly.visible))
    return c


if __name__ == "__main__":
    raise SystemExit(main())
