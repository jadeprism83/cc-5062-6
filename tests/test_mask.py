"""组合运算语义、软边不被阈值化、相互接近轮廓。"""
import unittest

import numpy as np

from selection_kernel.geometry import circle_path, rectangle_path
from selection_kernel.mask import CombineOp, CoverageMask, combine, merge_masks
from selection_kernel.rasterize import rasterize_paths


def cov_of(path, w, h):
    return CoverageMask(rasterize_paths(path.flattened(), w, h))


class MaskSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.a = cov_of(rectangle_path(2, 2, 8, 8), 16, 16)
        self.b = cov_of(rectangle_path(6, 6, 8, 8), 16, 16)

    def test_union_subtract_intersect_identities(self):
        u = combine(self.a, self.b, CombineOp.UNION)
        i = combine(self.a, self.b, CombineOp.INTERSECT)
        s = combine(self.a, self.b, CombineOp.SUBTRACT)
        # 全集：union = A + B - intersection（连续覆盖恒等式）
        np.testing.assert_allclose(
            u.coverage,
            self.a.coverage + self.b.coverage - i.coverage, atol=1e-6)
        # A 减 B 后与 B 相交应为 0
        self.assertLess(float(combine(s, self.b, CombineOp.INTERSECT)
                              .coverage.max()), 1e-6)
        # 相交不超过任一操作数
        self.assertTrue(np.all(i.coverage <= self.a.coverage + 1e-6))
        self.assertTrue(np.all(i.coverage <= self.b.coverage + 1e-6))

    def test_union_idempotent_and_outside_zero(self):
        u1 = combine(self.a, self.a, CombineOp.UNION)
        np.testing.assert_allclose(u1.coverage, self.a.coverage, atol=1e-6)
        # 选区外严格为 0
        self.assertEqual(float(u1.coverage[0, 0]), 0.0)
        self.assertEqual(float(u1.coverage[15, 15]), 0.0)

    def test_soft_edges_survive_combination(self):
        # 亚像素偏移矩形产生软边；组合后软边仍然是连续值，不被二值化
        soft = cov_of(rectangle_path(2.3, 2.7, 8.4, 8.1), 16, 16)
        u = combine(soft, self.b, CombineOp.UNION)
        s = combine(soft, self.b, CombineOp.SUBTRACT)
        for m in (u, s):
            vals = m.coverage[(m.coverage > 0.0) & (m.coverage < 1.0)]
            self.assertGreater(len(vals), 0)

    def test_close_contours_remain_distinct(self):
        # 两条相距 0.4 像素的接近轮廓：细缝必须保留，不能糊成一块
        left = cov_of(rectangle_path(2.0, 2.0, 5.0, 10.0), 16, 16)
        right = cov_of(rectangle_path(7.4, 2.0, 5.0, 10.0), 16, 16)
        gap = combine(left, right, CombineOp.UNION)
        # x=7.0..7.4 之间存在覆盖 <1 的缝隙列
        col = gap.coverage[:, 7]
        self.assertTrue(np.any(col < 1.0))
        # 两条轮廓各自面积守恒（并集之和 = 各自之和，因为不重叠）
        self.assertAlmostEqual(float(left.coverage.sum() + right.coverage.sum()),
                               float(gap.coverage.sum()), places=5)

    def test_merge_sequence_order_independent_for_union(self):
        c = cov_of(circle_path(8, 8, 5), 16, 16)
        u1 = merge_masks([self.a, self.b, c], CombineOp.UNION).coverage
        u2 = merge_masks([c, self.a, self.b], CombineOp.UNION).coverage
        np.testing.assert_allclose(u1, u2, atol=1e-6)

    def test_subtract_order_dependent(self):
        s1 = combine(self.a, self.b, CombineOp.SUBTRACT).coverage
        s2 = combine(self.b, self.a, CombineOp.SUBTRACT).coverage
        self.assertGreater(np.abs(s1 - s2).sum(), 1.0)


if __name__ == "__main__":
    unittest.main()
