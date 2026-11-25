"""大画布处理：瓦片路径与整幅参考在大尺寸上的一致性与性能。"""
import time
import unittest

import numpy as np

from selection_kernel.geometry import circle_path, slanted_path
from selection_kernel.tiling import assert_tile_consistency, tiled_render
from selection_kernel.rasterize import rasterize_paths


class LargeCanvasTests(unittest.TestCase):
    def test_large_canvas_tile_consistency(self):
        h, w = 600, 800
        polys = (circle_path(400.3, 300.7, 220.4).flattened()
                 + slanted_path(120.5, 90.25, 300.0, 420.0, 40.0).flattened())
        t0 = time.perf_counter()
        err = assert_tile_consistency(polys, h, w, 64, 64, feather=5.0)
        elapsed = time.perf_counter() - t0
        self.assertLess(err, 1e-6)
        # 纯 Python 解析光栅化对 48 万像素应在可接受时间内完成（留宽松上限）
        self.assertLess(elapsed, 60.0)

    def test_tiled_sum_matches_full_area(self):
        h, w = 300, 300
        polys = circle_path(150.0, 150.0, 100.0).flattened()
        tiled = tiled_render(polys, h, w, 50, 50).coverage
        full = rasterize_paths(polys, w, h)
        np.testing.assert_allclose(tiled, full, atol=1e-6)
        import math
        self.assertAlmostEqual(float(tiled.sum()), math.pi * 100.0 ** 2,
                               delta=math.pi * 1e4 * 1e-3)


if __name__ == "__main__":
    unittest.main()
