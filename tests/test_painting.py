"""笔刷受选区限制、跨图层剪切所有权、棋盘底、真实绘画导出。"""
import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from selection_kernel.export import (
    load_image_rgba, save_mask_png, save_on_checker, save_rgba,
)
from selection_kernel.geometry import circle_path, rectangle_path
from selection_kernel.painting import (
    Canvas, composite_over_checker, cut_with_selection, ownership_difference,
    paint_brush,
)
from selection_kernel.rasterize import rasterize_paths
from selection_kernel.selection import Selection


class PaintingTests(unittest.TestCase):
    def _selection(self, h, w, feather=0.0):
        path = circle_path(w / 2, h / 2, min(h, w) / 3)
        sel = Selection("sel", (h, w), [path]).set_feather(feather)
        return sel.coverage.coverage

    def test_brush_confined_to_selection(self):
        h, w = 40, 40
        canvas = Canvas(h, w)
        layer = canvas.new_layer("paint")
        sel = self._selection(h, w)
        paint_brush(canvas, layer, [(6, 6), (34, 34)], radius=4.0,
                    color_rgb=(1.0, 0.2, 0.1), selection=sel, hardness=0.8)
        alpha = layer.rgba[..., 3]
        # 选区外严格无写入
        outside = sel == 0.0
        self.assertEqual(float(alpha[outside].max()), 0.0)
        # 选区内软边：alpha 不超过选区覆盖
        self.assertTrue(np.all(alpha <= sel + 1e-6))
        # 笔触确实发生
        self.assertGreater(float(alpha.max()), 0.5)

    def test_feathered_selection_gives_partial_paint(self):
        h, w = 36, 36
        canvas = Canvas(h, w)
        layer = canvas.new_layer("p")
        sel = self._selection(h, w, feather=4.0)
        paint_brush(canvas, layer, [(18, 18)], radius=10.0,
                    color_rgb=(0.0, 0.5, 1.0), selection=sel, hardness=1.0)
        alpha = layer.rgba[..., 3]
        band = (sel > 0.02) & (sel < 0.98)
        self.assertGreater(np.count_nonzero((alpha > 0.0) & (alpha < 1.0)), 0)
        self.assertTrue(np.all(alpha[band] <= sel[band] + 1e-6))

    def test_cut_leaves_hole_and_two_ownership_modes_differ(self):
        h, w = 30, 30
        canvas = Canvas(h, w)
        l1 = canvas.new_layer("l1")
        l2 = canvas.new_layer("l2")
        # 两层画不同颜色的重叠软圆
        sel1 = rasterize_paths(circle_path(13, 15, 9).flattened(), w, h)
        sel2 = rasterize_paths(circle_path(17, 15, 9).flattened(), w, h)
        paint_brush(canvas, l1, [(13, 15)], radius=9.0,
                    color_rgb=(1.0, 0.0, 0.0), selection=sel1, hardness=0.5)
        paint_brush(canvas, l2, [(17, 15)], radius=9.0,
                    color_rgb=(0.0, 0.0, 1.0), selection=sel2, hardness=0.5)

        cut_cov = rasterize_paths(rectangle_path(8, 8, 14, 14).flattened(), w, h)
        # 软化为部分覆盖，制造所有权分歧
        from selection_kernel.distance import polygon_signed_distance, feather_from_distance
        polys = rectangle_path(8, 8, 14, 14).flattened()
        sdf = polygon_signed_distance(polys, w, h, hard_cover=cut_cov)
        cut_cov = feather_from_distance(sdf, 2.5)

        before_l1 = l1.rgba.copy()
        before_l2 = l2.rgba.copy()
        new_layer, src = cut_with_selection(canvas, ["l1", "l2"], cut_cov,
                                            "merge_first")
        # 洞：原层残留 alpha = 原 alpha × (1-c)
        for ly, before in zip(src, (before_l1, before_l2)):
            np.testing.assert_allclose(ly.rgba[..., 3],
                                       before[..., 3] * (1.0 - cut_cov), atol=1e-6)
        # 剪下图块 alpha 受 c 裁决
        self.assertTrue(np.all(new_layer.rgba[..., 3] <= cut_cov + 1e-6))

        # 两种语义确实不同（覆盖率相同的软边，归属不同）
        diff_img, max_diff = ownership_difference(
            Canvas(h, w, [type(l1)("l1", before_l1), type(l2)("l2", before_l2)]),
            ["l1", "l2"], cut_cov)
        self.assertGreater(max_diff, 1e-3)

        # 棋盘底上能看到半透明边缘，且完全透明处显示底色
        preview = composite_over_checker(new_layer.rgba)
        self.assertEqual(preview.shape, (h, w, 3))
        empty = new_layer.rgba[..., 3] == 0
        self.assertTrue(np.all(preview[empty] >= 0.0))

    def test_round_trip_exports_full_resolution(self):
        h, w = 33, 41
        cov = rasterize_paths(circle_path(20.5, 16.5, 10.25).flattened(), w, h)
        with tempfile.TemporaryDirectory() as d:
            mp = os.path.join(d, "mask.png")
            save_mask_png(cov, mp)
            reloaded = np.asarray(Image.open(mp), dtype=np.float64) / 65535.0
            self.assertEqual(reloaded.shape, (h, w))
            np.testing.assert_allclose(reloaded, cov, atol=1.0 / 65535.0 + 1e-6)

            canvas = Canvas(h, w)
            layer = canvas.new_layer("p")
            paint_brush(canvas, layer, [(20.5, 16.5)], radius=10.25,
                        color_rgb=(0.2, 0.8, 0.3), selection=cov, hardness=0.7)
            rp = os.path.join(d, "paint.png")
            save_rgba(layer.rgba, rp)
            back = load_image_rgba(rp)
            self.assertEqual(back.shape, (h, w, 4))
            cp = os.path.join(d, "checker.png")
            save_on_checker(layer.rgba, cp)
            with Image.open(cp) as im:
                self.assertEqual(im.size, (w, h))


if __name__ == "__main__":
    unittest.main()
