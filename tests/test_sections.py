"""截面、对称性、选区外保持与显示分辨率无关性核查。"""
import math
import unittest

import numpy as np

from selection_kernel.geometry import (
    circle_path, rectangle_path, slanted_path, thin_slit_path,
)
from selection_kernel.rasterize import rasterize_paths


def render(path, w, h, origin=(0.0, 0.0)):
    return rasterize_paths(path.flattened(), w, h, origin)


class CrossSectionTests(unittest.TestCase):
    def test_circle_horizontal_cross_section(self):
        r = 8.0
        cx = cy = 16.0
        cov = render(circle_path(cx, cy, r), 32, 32)
        row = cov[16]  # 过圆心的水平截面
        for i, v in enumerate(row):
            x = i + 0.5 - cx
            if abs(x) < r - 1.0:
                self.assertAlmostEqual(float(v), 1.0, places=5)
            elif abs(x) > r + 1.0:
                self.assertEqual(float(v), 0.0)
        # 截面上覆盖率之和 ≈ 直径 2r（两端部分覆盖像素带来小误差）
        self.assertAlmostEqual(float(row.sum()), 2 * r, delta=0.1)

    def test_slit_cross_section_width(self):
        # 0.25px 细缝：竖直截面的总覆盖 ≈ 缝宽
        path = thin_slit_path(cx=12.5, cy=12.0, width=0.25, height=10.0)
        cov = render(path, 24, 24)
        mid_row = cov[12].sum()
        self.assertAlmostEqual(float(mid_row), 0.25, places=2)
        # 每条中间行都应相同（缝上下端除外）
        for j in range(8, 16):
            self.assertAlmostEqual(float(cov[j].sum()), 0.25, places=2)
        # 细缝不能扩散到相邻整列：最大单像素覆盖 <= 缝宽（直线边界）
        self.assertLessEqual(float(cov.max()), 0.25 + 1e-6)

    def test_slanted_cross_section_moves_with_y(self):
        path = slanted_path(4.0, 2.0, 10.0, 18.0, shear=4.0)
        cov = render(path, 24, 24)
        # 用每行“覆盖重心 x”度量斜边平移，避免部分覆盖像素的离散影响
        def centroid(j):
            row = cov[j]
            xs = np.arange(len(row)) + 0.5
            return float((row * xs).sum() / row.sum())
        c6, c10, c14 = centroid(6), centroid(10), centroid(14)
        # 左边界随 y 下移而左移（shear 方向）
        self.assertLess(c10, c6)
        self.assertLess(c14, c10)
        # 平移速率 ≈ shear/height = 4/18 每像素行
        self.assertAlmostEqual((c6 - c14) / 8.0, 4.0 / 18.0, places=2)

    def test_symmetry_under_reflection(self):
        cov = render(circle_path(15.5, 15.5, 7.3), 31, 31)
        np.testing.assert_allclose(cov, cov[::-1, :], atol=1e-6)
        np.testing.assert_allclose(cov, cov[:, ::-1], atol=1e-6)
        # 180° 旋转对称
        np.testing.assert_allclose(cov, cov[::-1, ::-1], atol=1e-6)

    def test_outside_pixels_preserved_for_all_shapes(self):
        shapes = [
            circle_path(12.0, 12.0, 6.0),
            slanted_path(5.0, 5.0, 9.0, 9.0, 3.0),
            thin_slit_path(12.0, 12.0, 0.4, 8.0),
        ]
        h = w = 28
        for path in shapes:
            cov = render(path, w, h)
            x0, y0, x1, y1 = path.bounds()
            outside = np.ones((h, w), dtype=bool)
            outside[int(math.floor(y0)) - 1:int(math.ceil(y1)) + 1,
                    int(math.floor(x0)) - 1:int(math.ceil(x1)) + 1] = False
            outside &= np.indices((h, w))[0] >= 0
            self.assertEqual(float(cov[outside].max()), 0.0)

    def test_display_zoom_does_not_change_canvas_mask(self):
        # 内核不接收显示缩放参数：同一几何在“高分辨率画布”上的输出只由
        # 画布坐标决定。这里用不同 origin 模拟视图平移，结果应严格平移一致。
        path = rectangle_path(6.25, 7.75, 9.5, 8.25)
        full = render(path, 32, 32, origin=(0.0, 0.0))
        shifted = render(path, 32, 32, origin=(4.0, 3.0))
        # origin=(4,3) 时，画布 (y,x) 对应 shifted 的 (y-3,x-4)
        crop_full = full[3:, 4:]
        crop_shift = shifted[:-3, :-4]
        np.testing.assert_allclose(crop_full, crop_shift, atol=1e-6)

    def test_high_resolution_export_not_limited_by_display_grid(self):
        # 在 200x200 全分辨率画布渲染细弧；若误用粗显示网格（20x20 即
        # 10x 降采样）会丢失软边，对比证明全分辨率保留了更多覆盖率层级
        path = circle_path(100.0, 100.0, 40.0)
        hi = render(path, 200, 200)
        down = render(path, 20, 20)
        self.assertEqual(hi.shape, (200, 200))
        self.assertGreater(np.count_nonzero((hi > 0) & (hi < 1)),
                           np.count_nonzero((down > 0) & (down < 1)) * 4)
        # 高分辨率总面积更接近 πr²
        self.assertAlmostEqual(float(hi.sum()), math.pi * 40.0 ** 2,
                               delta=math.pi * 40.0 ** 2 * 1e-3)


if __name__ == "__main__":
    unittest.main()
