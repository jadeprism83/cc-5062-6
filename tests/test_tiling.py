"""瓦片边缘与整幅参考一致性，含非整除尺寸与羽化。"""
import unittest

import numpy as np

from selection_kernel.geometry import circle_path, slanted_path
from selection_kernel.mask import CoverageMask, CombineOp, combine
from selection_kernel.tiling import (
    assert_tile_consistency, tiled_combine, tiled_render,
)


class TilingTests(unittest.TestCase):
    def test_circle_tiles_match_full(self):
        polys = circle_path(17.3, 13.7, 9.6).flattened()
        err = assert_tile_consistency(polys, 33, 31, 8, 8)
        self.assertLess(err, 1e-6)

    def test_feathered_tiles_match_full(self):
        polys = slanted_path(4.2, 3.1, 20.0, 24.0, 5.0).flattened()
        err = assert_tile_consistency(polys, 34, 37, 9, 7, feather=3.5)
        self.assertLess(err, 1e-6)

    def test_non_divisible_tile_sizes(self):
        polys = circle_path(10.0, 10.0, 8.0).flattened()
        err = assert_tile_consistency(polys, 23, 19, 6, 5, feather=2.0)
        self.assertLess(err, 1e-6)

    def test_tiled_combine_ops(self):
        h, w = 20, 20
        pa = circle_path(8.0, 10.0, 7.0).flattened()
        pb = circle_path(12.0, 10.0, 7.0).flattened()
        for op in CombineOp:
            tiles_a = list(CoverageMask(tiled_render(pa, h, w, 6, 6).coverage)
                           .tiles(6, 6))
            tiles_b = list(CoverageMask(tiled_render(pb, h, w, 6, 6).coverage)
                           .tiles(6, 6))
            tiled = tiled_combine(tiles_a, tiles_b, (h, w), op).coverage
            from selection_kernel.rasterize import rasterize_paths
            full = combine(CoverageMask(rasterize_paths(pa, w, h)),
                           CoverageMask(rasterize_paths(pb, w, h)), op).coverage
            np.testing.assert_allclose(tiled, full, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
