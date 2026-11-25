"""解析光栅化的正确性：面积守恒、亚像素、斜边、极细缝、圆。"""
import math
import unittest

import numpy as np

from selection_kernel.geometry import (
    Path, SubPath, circle_path, rectangle_path, slanted_path, thin_slit_path,
)
from selection_kernel.rasterize import rasterize_paths


def render(path, w, h, origin=(0.0, 0.0)):
    return rasterize_paths(path.flattened(), w, h, origin)


def reversed_rect(x, y, w, h):
    # 与 rectangle_path 顶点顺序相反，形成反向环绕内圈（洞）
    sp = SubPath()
    sp.move_to((x, y))
    sp.line_to((x, y + h))
    sp.line_to((x + w, y + h))
    sp.line_to((x + w, y))
    sp.close()
    return sp


class RasterizeTests(unittest.TestCase):
    def test_axis_rectangle_total_area(self):
        # 整数对齐矩形：覆盖率之和必须精确等于几何面积
        path = rectangle_path(2.0, 3.0, 10.0, 7.0)
        cov = render(path, 20, 15)
        self.assertAlmostEqual(float(cov.sum()), 70.0, places=5)
        self.assertEqual(float(cov[3:10, 2:12].min()), 1.0)
        self.assertEqual(float(cov[0:3, :].max()), 0.0)
        self.assertEqual(float(cov[10:, :].max()), 0.0)

    def test_subpixel_offset_preserves_area(self):
        # 亚像素偏移：总面积守恒；非整数对齐时边缘像素出现 0<c<1
        areas = []
        for dx in (0.0, 0.137, 0.5, 0.873):
            path = rectangle_path(2.0 + dx, 4.0, 8.0, 6.0)
            cov = render(path, 24, 16)
            areas.append(float(cov.sum()))
        for a in areas:
            self.assertAlmostEqual(a, 48.0, places=4)
        soft_path = rectangle_path(2.137, 4.0, 8.0, 6.0)
        soft_cov = render(soft_path, 24, 16)
        soft = np.count_nonzero((soft_cov > 0.0) & (soft_cov < 1.0))
        self.assertGreater(soft, 0)

    def test_slanted_edge_profile(self):
        # 斜边：每行软边像素覆盖率应单调，且截面沿边平移一致
        path = slanted_path(4.0, 2.0, 12.0, 20.0, shear=3.0)
        cov = render(path, 24, 26)
        self.assertAlmostEqual(float(cov.sum()), 12.0 * 20.0, places=3)
        row_profiles = []
        for j in (5, 10, 15):
            row = cov[j]
            xs = np.where(row > 0)[0]
            self.assertGreater(len(xs), 0)
            soft = row[(row > 0) & (row < 1.0)]
            # 左右两条斜边各自构成一个单调软边段（上升段、下降段）
            self.assertGreaterEqual(len(soft), 2)
            peak = int(np.argmax(soft))
            self.assertTrue(np.all(np.diff(soft[:peak + 1]) >= -1e-6))
            self.assertTrue(np.all(np.diff(soft[peak:]) <= 1e-6))
            row_profiles.append(soft.sum())
        # 平行四边形各行总覆盖宽度相同（12px），容差考虑起止行
        for j in (6, 10, 14, 18):
            self.assertAlmostEqual(float(cov[j].sum()), 12.0, places=4)

    def test_thin_slit_subpixel(self):
        # 0.35 像素宽的缝：不能消失，也不能被阈值化成整列 1
        path = thin_slit_path(cx=10.5, cy=10.0, width=0.35, height=12.0)
        cov = render(path, 22, 22)
        total = float(cov.sum())
        self.assertGreater(total, 0.0)
        self.assertLess(total, 12.0)  # 宽 <1 像素，总面积 < 高度
        self.assertAlmostEqual(total, 0.35 * 12.0, places=2)
        self.assertLess(float(cov.max()), 1.0)  # 没有任何像素被完全占满

    def test_circle_area_and_symmetry(self):
        r = 9.0
        path = circle_path(20.0, 20.0, r)
        cov = render(path, 40, 40)
        # 贝塞尔圆面积与 πr² 误差 < 0.1%
        self.assertAlmostEqual(float(cov.sum()), math.pi * r * r,
                               delta=math.pi * r * r * 1e-3)
        # 中心对称与轴对称
        np.testing.assert_allclose(cov, cov[::-1, :], atol=1e-6)
        np.testing.assert_allclose(cov, cov[:, ::-1], atol=1e-6)

    def test_hole_via_winding(self):
        # 外框 + 反向内圈（洞）：净面积 = 外面积 - 洞面积
        outer = rectangle_path(1.0, 1.0, 10.0, 10.0).subpaths[0]
        inner = reversed_rect(4.0, 4.0, 4.0, 4.0)
        path = Path([outer, inner])
        cov = render(path, 14, 14)
        self.assertAlmostEqual(float(cov.sum()), 100.0 - 16.0, places=3)
        self.assertEqual(float(cov[5, 5]), 0.0)

    def test_pixels_outside_untouched(self):
        path = rectangle_path(5.0, 5.0, 6.0, 6.0)
        cov = render(path, 30, 30)
        outside = np.ones_like(cov, dtype=bool)
        outside[5:11, 5:11] = False
        self.assertEqual(float(cov[outside].max()), 0.0)


if __name__ == "__main__":
    unittest.main()
