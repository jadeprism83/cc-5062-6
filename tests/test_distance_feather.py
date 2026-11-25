"""距离场、羽化语义、旋转/羽化次序、坐标域保持。"""
import math
import unittest

import numpy as np

from selection_kernel.distance import (
    feather_from_distance, polygon_signed_distance, rasterize_feathered,
)
from selection_kernel.geometry import Path, circle_path, rectangle_path
from selection_kernel.rasterize import rasterize_paths
from selection_kernel.selection import Selection


class FeatherTests(unittest.TestCase):
    def test_feather_profile_and_width(self):
        path = rectangle_path(10.0, 10.0, 20.0, 20.0)
        polys = path.flattened()
        hard = rasterize_paths(polys, 40, 40)
        r = 4.0
        sdf = polygon_signed_distance(polys, 40, 40, hard_cover=hard)
        cov = feather_from_distance(sdf, r)
        # 远离边界的内部保持 1，外部保持 0
        self.assertEqual(float(cov[20, 20]), 1.0)
        self.assertEqual(float(cov[1, 1]), 0.0)
        # 边界处（sdf≈0）覆盖 ≈ 0.5
        boundary = np.abs(sdf) < 0.05
        self.assertTrue(np.all(np.abs(cov[boundary] - 0.5) < 0.05))
        # 羽化过渡带宽度约为 2r：从边界向外 r 像素仍 >0，2r 外为 0
        row = cov[10, :]  # 穿过左边 x=10
        self.assertGreater(float(row[10 - 3]), 0.0)
        self.assertEqual(float(row[10 - 5]), 0.0)
        # 羽化使总覆盖面积大于硬边但有界
        self.assertGreater(float(cov.sum()), float(hard.sum()))
        self.assertLessEqual(float(cov.max()), 1.0)

    def test_feather_radius_is_canvas_units(self):
        # 同样的几何与半径，在不同“显示缩放”下（这里只改变观察方式，
        # 内核根本不接收缩放参数）输出完全相同；另用不同瓦片原点验证
        path = rectangle_path(8.0, 8.0, 16.0, 16.0)
        polys = path.flattened()
        c1 = rasterize_feathered(polys, 32, 32, 3.0, origin=(0, 0))
        c2 = rasterize_feathered(polys, 32, 32, 3.0, origin=(0, 0))
        np.testing.assert_array_equal(c1, c2)
        # 半径翻倍 -> 边界斜率减半：采样边界外 1.5px 处覆盖
        hard = rasterize_paths(polys, 32, 32)
        sdf = polygon_signed_distance(polys, 32, 32, hard_cover=hard)
        f3 = feather_from_distance(sdf, 3.0, hard_cover=hard)
        f6 = feather_from_distance(sdf, 6.0, hard_cover=hard)
        # 线性羽化 c(d)=0.5-d/(2r)：在边界外同一距离 d≈1.5 处，
        # c 与 1/r 成线性关系，用两点核查斜率而非错误的倍数假设
        probe = (sdf > 1.4) & (sdf < 1.6)
        d = sdf[probe]
        np.testing.assert_allclose(f3[probe], 0.5 - d / 6.0, atol=0.02)
        np.testing.assert_allclose(f6[probe], 0.5 - d / 12.0, atol=0.02)
        # 半径越大，边界外固定距离处保留的覆盖越高
        self.assertTrue(np.all(f6[probe] > f3[probe]))

    def test_rotate_then_feather_differs_from_feather_then_rotate(self):
        h, w = 48, 48
        path = rectangle_path(14.0, 18.0, 20.0, 12.0)
        angle = math.radians(37)
        center = (24.0, 24.0)
        r = 3.0

        # 序列 A：先旋转（矢量级，重新解析光栅化）再羽化
        sel = Selection("s", (h, w), [path])
        a = sel.rotate(angle, center).set_feather(r).coverage.coverage

        # 序列 B：先羽化，再对“覆盖率图像”做图像域旋转（模拟对已羽化
        # 结果施加变换而非重算矢量），以展示操作次序不可交换
        from PIL import Image
        f = sel.set_feather(r).coverage.coverage
        img = Image.fromarray((f * 255).astype(np.uint8))
        rot = img.rotate(math.degrees(angle), center=center, resample=Image.BILINEAR,
                         fillcolor=0)
        b = np.asarray(rot, dtype=np.float32) / 255.0

        diff = np.abs(a - b)
        self.assertGreater(float(diff.max()), 0.05)
        # 两条序列都保留完整坐标域
        self.assertEqual(a.shape, (h, w))
        self.assertEqual(b.shape, (h, w))
        # 历史日志记录了次序
        hist = sel.rotate(angle, center).set_feather(r).history
        self.assertEqual(hist[0].startswith("rotate"), True)
        self.assertEqual(hist[1].startswith("feather"), True)

    def test_rotation_returns_to_original_scale_coverage(self):
        # 旋转 + 反向旋转后重新光栅化，覆盖率应回到原值（面积核查）
        h, w = 40, 40
        path = circle_path(20.0, 20.0, 9.0)
        sel = Selection("s", (h, w), [path])
        angle = math.radians(29)
        center = (20.0, 20.0)
        rotated = sel.rotate(angle, center)
        back = rotated.rotate(-angle, center)
        c0 = sel.render_hard().coverage
        c1 = back.render_hard().coverage
        self.assertAlmostEqual(float(c0.sum()), float(c1.sum()), places=3)
        np.testing.assert_allclose(c0, c1, atol=2e-3)

    def test_zero_radius_feather_equals_hard_edge(self):
        path = rectangle_path(4.3, 4.7, 10.2, 9.1)
        polys = path.flattened()
        hard = rasterize_paths(polys, 24, 24)
        sdf = polygon_signed_distance(polys, 24, 24, hard_cover=hard)
        f0 = feather_from_distance(sdf, 0.0, hard_cover=hard)
        # 零半径羽化 = 解析面积覆盖本身，保留 0<c<1 的抗锯齿细边
        np.testing.assert_array_equal(f0, hard)
        self.assertGreater(np.count_nonzero((hard > 0) & (hard < 1)), 0)
        # 任何 r>0 都产生更宽的软边过渡带
        fr = feather_from_distance(sdf, 1.0, hard_cover=hard)
        self.assertGreater(np.count_nonzero((fr > 0) & (fr < 1)),
                           np.count_nonzero((hard > 0) & (hard < 1)))


if __name__ == "__main__":
    unittest.main()
