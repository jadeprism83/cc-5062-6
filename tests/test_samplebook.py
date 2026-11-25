"""选区样册工程测试。

覆盖：
- 入库素材/脚本/旧页 PNG 真实存在且目录一致；
- Path JSON 编解码往返；
- 每个样例从原路径重建的覆盖量/边缘颜色符合 JSON 中写明的约定；
- 分辨率实验（预览消失、4x 矢量重 raster 回归、质量按 s^2 放大）；
- 双色套版从实际写盘 PNG 回读油墨，坏版对照必须被查出补孔；
- 旧页不可恢复标记；
- build_samplebook 全流程到临时目录必须 all_pass；
- GUI HTTP 接口确实调用内核（用一次性几何数值核对，拒绝固定数据）；
- 旧 examples/demo_pipeline.py 端到端运行到底。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from selection_kernel.pathio import path_from_dict, path_to_dict  # noqa: E402
from selection_kernel.samplebook import (  # noqa: E402
    catalog, catalog_entries, compose_spec, load_entry, load_legacy_mask,
    load_sample, mark_unrecoverable, roi_stats, sample_dir,
)
from selection_kernel.visualize import edge_color_rgba  # noqa: E402


class AssetsExistTests(unittest.TestCase):
    def test_catalog_files_really_exist(self):
        cat = catalog()
        ids = set()
        for e in cat["samples"]:
            ids.add(e["id"])
            p = os.path.join(sample_dir(), e["file"])
            self.assertTrue(os.path.isfile(p), f"missing material {p}")
            self.assertGreater(os.path.getsize(p), 20)
            with open(p, encoding="utf-8") as f:
                json.load(f)
        # 任务要求的关键素材类别都在
        for required in ("hole_filament", "subpixel_lines",
                         "repeat_spokes", "intersect_narrow",
                         "hole_still_life", "old_page_hole7"):
            self.assertIn(required, ids)

    def test_legacy_png_is_real_image(self):
        record, rp = load_entry("old_page_hole7")
        p = os.path.join(os.path.dirname(rp), record["mask_png"])
        self.assertTrue(os.path.isfile(p))
        with open(p, "rb") as f:
            self.assertEqual(f.read(8), b"\x89PNG\r\n\x1a\n")
        mask = load_legacy_mask(record, rp)
        self.assertEqual(mask.shape, (50, 60))
        self.assertEqual(set(np.unique(mask)).issubset({0.0, 1.0}), True)

    def test_scripts_exist(self):
        for rel in ("examples/build_samplebook.py",
                    "examples/create_sample_assets.py",
                    "examples/demo_pipeline.py"):
            self.assertTrue(os.path.isfile(os.path.join(ROOT, rel)))


class PathCodecTests(unittest.TestCase):
    def test_roundtrip_preserves_commands_and_geometry(self):
        spec = load_sample("hole_filament")
        for it in spec["items"]:
            p1 = path_from_dict(it["path"])
            d2 = path_to_dict(p1)
            p2 = path_from_dict(d2)
            a = [x for sp in p1.flattened() for x in np.asarray(sp).ravel()]
            b = [x for sp in p2.flattened() for x in np.asarray(sp).ravel()]
            np.testing.assert_allclose(a, b, atol=1e-12)
            self.assertEqual(
                [c[0] for sp in d2["commands"] for c in [sp]],
                [c[0] for c in it["path"]["commands"]])


class SampleConventionTests(unittest.TestCase):
    def test_all_selection_samples_meet_references(self):
        for entry in catalog_entries("selection"):
            sid = entry["id"]
            spec = load_sample(sid)
            res = compose_spec(spec)
            ref = spec["reference"]
            rm = ref["hard_mass"]
            mass = float(res.hard.coverage.sum())
            self.assertLessEqual(
                abs(mass - rm["expected"]), rm["tol_abs"],
                f"{sid} mass {mass} vs {rm['expected']}")

            for ep in ref.get("edge_pixels", []):
                i, j = ep["pixel"]
                c = float(res.hard.coverage[j, i])
                self.assertLessEqual(abs(c - ep["expected_hard"]),
                                     ep["tol"], f"{sid} edge {ep['pixel']}")
                rgba = edge_color_rgba(res.hard.coverage)[j, i]
                v = int(round(ep["expected_hard"] * 255))
                self.assertLessEqual(abs(int(rgba[0]) - v), 1)

            for name, rv in ref.get("rois", {}).items():
                st = roi_stats(res.hard.coverage, rv["xyxy"])
                if "max_at_most" in rv:
                    self.assertLessEqual(st["max"], rv["max_at_most"] + 1e-6,
                                         f"{sid} roi {name}")
                if "sum_expected" in rv:
                    self.assertLessEqual(
                        abs(st["sum"] - rv["sum_expected"]),
                        rv.get("sum_tol", 0.2), f"{sid} roi {name}")

    def test_repeat_transform_count_and_area(self):
        spec = load_sample("repeat_spokes")
        raw = len(spec["items"])
        expanded = compose_spec  # import guard
        from selection_kernel.samplebook import expanded_items
        items = expanded_items(spec)
        self.assertEqual(raw, 1)
        self.assertEqual(len(items), 12)
        res = compose_spec(spec)
        self.assertAlmostEqual(float(res.hard.coverage.sum()),
                               12 * 6 * 54, delta=3.0)

    def test_intersect_feather_has_no_ghost_edges(self):
        spec = load_sample("intersect_narrow")
        res = compose_spec(spec)
        fc = spec["reference"]["feather_checks"]
        ghost = roi_stats(res.coverage, fc["ghost_edge_roi"])
        self.assertLessEqual(ghost["max"],
                             fc["ghost_edge_max_at_most"] + 1e-6)
        i, j = fc["center_pixel"]
        self.assertGreaterEqual(float(res.coverage[j, i]), fc["center_min"])
        self.assertGreater(np.count_nonzero(
            (res.coverage > 0) & (res.coverage < 1)), 100)


class ResolutionAndPlatesTests(unittest.TestCase):
    def test_preview_disappears_but_vector_4x_recovers(self):
        from selection_kernel.samplebook import box_downsample
        spec = load_sample("hole_filament")
        hard = compose_spec(spec).hard.coverage
        lab = spec["resolution_lab"]
        prev = box_downsample(hard, lab["preview_factor"])
        pf = roi_stats(prev, lab["preview_filament_roi"])
        self.assertLess(pf["max"], lab["preview_filament_max_below"])

        s = lab["output_scale"]
        hard4 = compose_spec(spec, scale=s).hard.coverage
        f4 = roi_stats(hard4, spec["plates"]["filament_roi"], scale=s)
        h4 = roi_stats(hard4, spec["plates"]["hole_roi"], scale=s)
        self.assertGreaterEqual(f4["max"], lab["output_filament_peak_min"])
        self.assertLessEqual(h4["max"], 0.02)
        self.assertLess(abs(hard4.sum() / hard.sum() - s * s),
                        s * s * lab["mass_scale_tol_rel"])

    def test_plate_readback_and_bad_control(self):
        from selection_kernel.plates import (
            fill_enclosed_holes, load_plate_ink, save_plate_ink_png,
            verify_filament_and_hole,
        )
        with tempfile.TemporaryDirectory() as td:
            spec = load_sample("hole_filament")
            plate = compose_spec(spec).hard.coverage
            p = os.path.join(td, "plate.png")
            save_plate_ink_png(plate, p)
            ink = load_plate_ink(p)  # 从实际文件读
            ok = verify_filament_and_hole(
                ink, spec["plates"]["filament_roi"],
                spec["plates"]["hole_roi"])
            self.assertTrue(ok["pass"])

            bad_path = os.path.join(td, "bad.png")
            save_plate_ink_png(fill_enclosed_holes(plate), bad_path)
            bad = verify_filament_and_hole(
                load_plate_ink(bad_path),
                spec["plates"]["filament_roi"],
                spec["plates"]["hole_roi"])
            self.assertFalse(bad["hole_empty"])
            self.assertFalse(bad["pass"])

    def test_legacy_marking_covers_lost_roi(self):
        record, rp = load_entry("old_page_hole7")
        mask = load_legacy_mask(record, rp)
        roi = record["unrecoverable_rois"][0]
        self.assertLessEqual(roi_stats(mask, roi)["max"], 1e-6)
        marked = mark_unrecoverable(mask, [roi], upscale=4)
        self.assertEqual(marked.shape, (200, 240, 3))
        ys, xs = slice(roi[1] * 4, roi[3] * 4), slice(roi[0] * 4, roi[2] * 4)
        redish = ((marked[ys, xs, 0] > 150)
                  & (marked[ys, xs, 1] < 90)).mean()
        self.assertGreater(float(redish), 0.2)


class BuildAllEvidenceTests(unittest.TestCase):
    def test_build_all_to_temp_passes(self):
        sys.path.insert(0, os.path.join(ROOT, "examples"))
        import build_samplebook as bs
        with tempfile.TemporaryDirectory() as td:
            report = bs.build_all(td)
            self.assertTrue(report["all_pass"],
                            json.dumps({k: v.get("pass") if isinstance(v, dict)
                                        else v for k, v in report.items()},
                                       ensure_ascii=False))
            self.assertTrue(os.path.isfile(os.path.join(td, "evidence.json")))
            self.assertTrue(os.path.isfile(os.path.join(td, "evidence.md")))
            # 关键导出图真实落盘、是可解码 PNG 且非空
            from PIL import Image
            for rel in ("plates/plate_A_vermilion_ink.png",
                        "plates/proof_overprint.png",
                        "paintings/hole_still_life_checker.png",
                        "legacy/old_page_hole7_marked.png"):
                p = os.path.join(td, rel)
                self.assertTrue(os.path.isfile(p), rel)
                self.assertGreater(os.path.getsize(p), 100)
                with Image.open(p) as im:
                    self.assertGreater(im.width * im.height, 1000)


class GuiHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import gui.browser_app as ba
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), ba.Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever,
                                     daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def _get(self, path):
        with urllib.request.urlopen(
                f"http://127.0.0.1:{self.port}{path}", timeout=30) as r:
            return json.loads(r.read())

    def _post(self, route, body):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{route}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            return json.loads(e.read())

    def test_samples_catalog_and_load(self):
        d = self._get("/api/samples")
        self.assertTrue(d["ok"])
        self.assertGreaterEqual(len(d["catalog"]["samples"]), 6)
        s = self._get("/api/sample?id=hole_filament")
        self.assertTrue(s["ok"])
        self.assertEqual(s["shapes"][0]["shape"], "path")

    def test_project_and_legacy_sample_load_branches(self):
        p = self._get("/api/sample?id=hole_still_life")
        self.assertTrue(p["ok"], p)
        self.assertEqual(p["load_kind"], "project")
        self.assertTrue(p["shapes"])  # 伴随选区已展开为可载入 path
        body = {"width": 240, "height": 200,
                "feather": p["spec"]["feather"], "shapes": p["shapes"]}
        r = self._post("/api/render", body)
        self.assertTrue(r["ok"], r)
        self.assertGreater(r["stats"]["soft_pixels"], 50)

        l = self._get("/api/sample?id=old_page_hole7")
        self.assertTrue(l["ok"], l)
        self.assertEqual(l["load_kind"], "legacy")
        self.assertEqual(l["shapes"], [])  # 不伪造矢量 shapes

    def test_render_uses_kernel_not_fixed_data(self):
        # 一次性分数坐标矩形：面积严格 200（10 宽 *20 高），任何固定假数据
        # 都不可能恰好跟随这个随手选取的几何值。
        body = {"width": 60, "height": 60, "feather": 0.0, "shapes": [
            {"shape": "rectangle", "op": "union",
             "x": 13.37, "y": 7.91, "w": 10.0, "h": 20.0}]}
        d = self._post("/api/render", body)
        self.assertTrue(d["ok"], d)
        self.assertAlmostEqual(d["stats"]["hard_sum"], 200.0, delta=0.05)
        self.assertIn("edge_png", d)

    def test_loaded_sample_render_matches_kernel_mass(self):
        s = self._get("/api/sample?id=subpixel_lines")
        body = {"width": 240, "height": 200, "feather": 0.0,
                "shapes": s["shapes"]}
        d = self._post("/api/render", body)
        self.assertTrue(d["ok"])
        self.assertAlmostEqual(d["stats"]["hard_sum"], 326.75, delta=0.5)

    def test_plates_and_legacy_routes(self):
        p = self._post("/api/sample_plates", {"id": "hole_filament"})
        self.assertTrue(p["ok"], p)
        self.assertTrue(p["readback"]["pass"])
        self.assertTrue(p["readback_4x"]["pass"])
        self.assertGreater(p["readback_4x"]["filament"]["max"], 0.5)
        l = self._post("/api/legacy_mark", {"id": "old_page_hole7"})
        self.assertTrue(l["ok"], l)
        self.assertEqual(l["roi_ink"]["max"], 0.0)
        self.assertIn("marked_png", l)


class OldExampleRunsTests(unittest.TestCase):
    def test_demo_pipeline_runs_to_completion(self):
        with tempfile.TemporaryDirectory() as td:
            proc = subprocess.run(
                [sys.executable, os.path.join(ROOT, "examples",
                                              "demo_pipeline.py"),
                 "--outdir", td],
                cwd=ROOT, capture_output=True, text=True, timeout=180)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(os.path.isfile(os.path.join(td, "evidence.json"))
                            or os.path.isfile(os.path.join(td, "mask_hard.png")))
            self.assertTrue(os.path.isfile(os.path.join(td, "mask_hard.png")))


if __name__ == "__main__":
    unittest.main()
