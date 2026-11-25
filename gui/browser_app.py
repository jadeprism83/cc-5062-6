"""精细选区绘画窗口（零外部依赖，标准库 http.server 驱动）。

启动：
    python3 gui/browser_app.py
然后打开终端提示的 http://127.0.0.1:8765/ 。

架构：浏览器只负责采集鼠标与显示；所有覆盖率计算都在内核中以固定的
高分辨率画布坐标完成，前端缩放仅改变显示，不参与掩码计算，因此导出
分辨率与预览一致，不会出现显示分辨率蒙版导致的锯齿/漏画。
失败（如羽化半径非法、瓦片校验不过）由内核返回 JSON 错误并在界面
红色反馈条显示，反馈回精细选区控件。
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from selection_kernel import (  # noqa: E402
    Canvas, CombineOp, Selection, SelectionDocument,
    assert_tile_consistency, circle_path, cut_with_selection, export,
    paint_brush, rectangle_path, slanted_path, thin_slit_path,
)
from selection_kernel.distance import (  # noqa: E402
    feather_from_distance, polygon_signed_distance,
)
from selection_kernel.rasterize import rasterize_paths  # noqa: E402
from selection_kernel import samplebook  # noqa: E402
from selection_kernel import visualize  # noqa: E402
from selection_kernel.plates import (  # noqa: E402
    INK_COLORS, load_plate_ink, overprint_proof, save_plate_ink_png,
    verify_filament_and_hole,
)

HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUI_LAST_DIR = os.path.join(REPO_ROOT, "output", "samplebook", "gui_last")

SHAPE_BUILDERS = samplebook.PRIMITIVE_BUILDERS


def _png_b64(arr_uint8: np.ndarray, mode: str) -> str:
    return visualize.png_b64(arr_uint8, mode)


def _coverage_rgba(coverage: np.ndarray) -> np.ndarray:
    """把 float 覆盖率编码为灰度 RGBA（保留软边层次，不做阈值化）。"""
    return visualize.coverage_gray_rgba(coverage)


def _edge_rgba(coverage: np.ndarray) -> np.ndarray:
    """边缘视图：只显示 0<c<1 的亚像素软边，红色越亮覆盖越高。"""
    return visualize.edge_color_rgba(coverage)


def build_selection(payload: dict):
    """根据前端请求构造组合后的硬边与羽化覆盖率。

    拖选基元与样册矢量 path 项都在这里进入 samplebook.compose_shapes
    这一真实内核入口；界面不存在任何固定覆盖率数据。
    """
    h = int(payload["height"])
    w = int(payload["width"])
    feather = float(payload.get("feather", 0.0))
    shapes = payload["shapes"]
    if feather < 0 or feather > min(h, w):
        raise ValueError(f"feather radius out of range: {feather}")
    if not shapes:
        zero = np.zeros((h, w), np.float32)
        return zero, zero, []
    edge_aware = bool(payload.get("edge_aware", True))
    result = samplebook.compose_shapes(shapes, h, w, feather, edge_aware)
    return result.hard.coverage, result.coverage, result.polys


class Handler(BaseHTTPRequestHandler):
    server_version = "SelectionKernelGUI/1.0"

    def log_message(self, fmt, *args):  # 安静日志
        sys.stderr.write("[gui] " + fmt % args + "\n")

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        route = parsed.path
        if route in ("/", "/index.html"):
            with open(HTML_PATH, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif route == "/api/samples":
            self._send_json({"ok": True, "catalog": samplebook.catalog()})
        elif route == "/api/sample":
            from urllib.parse import parse_qs
            q = parse_qs(parsed.query)
            sid = q.get("id", [""])[0]
            spec = samplebook.load_sample(sid)
            kind = spec.get("kind", "selection")
            if "items" in spec:
                shapes = self._sample_gui_shapes(spec)
                load_kind = kind
            elif spec.get("selection_sample"):
                # 绘画工程：画布上的选区来自伴随的矢量样册，采用工程羽化。
                companion = samplebook.load_sample(spec["selection_sample"])
                shapes = self._sample_gui_shapes(companion)
                load_kind = "project"
            else:
                # legacy 只有低清蒙版，没有可载入的矢量 shapes。
                shapes = []
                load_kind = "legacy"
            self._send_json({"ok": True, "spec": spec, "shapes": shapes,
                             "load_kind": load_kind})
        else:
            self._send_json({"ok": False, "error": "not found"}, 404)

    @staticmethod
    def _sample_gui_shapes(spec: dict):
        """把 repeat 展开后的样册项转成前端 shape 对象（保留矢量 path）。"""
        shapes = []
        for it in samplebook.expanded_items(spec):
            shapes.append({
                "shape": "path",
                "op": it.get("op", "union"),
                "path": it["path"],
                "label": it.get("id", "path"),
            })
        return shapes

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            route = self.path
            if route == "/api/render":
                self._handle_render(payload)
            elif route == "/api/paint":
                self._handle_paint(payload)
            elif route == "/api/cut":
                self._handle_cut(payload)
            elif route == "/api/tiles":
                self._handle_tiles(payload)
            elif route == "/api/sample_plates":
                self._handle_sample_plates(payload)
            elif route == "/api/legacy_mark":
                self._handle_legacy_mark(payload)
            else:
                self._send_json({"ok": False, "error": "unknown route"}, 404)
        except Exception as exc:  # 失败反馈回界面
            traceback.print_exc()
            self._send_json({"ok": False, "error": str(exc)}, 400)

    def _handle_render(self, payload):
        hard, feathered, _ = build_selection(payload)
        self._send_json({
            "ok": True,
            "hard_png": _png_b64(_coverage_rgba(hard), "RGBA"),
            "feather_png": _png_b64(_coverage_rgba(feathered), "RGBA"),
            "edge_png": _png_b64(_edge_rgba(feathered), "RGBA"),
            "stats": {
                "hard_sum": float(hard.sum()),
                "feather_sum": float(feathered.sum()),
                "soft_pixels": int(np.count_nonzero(
                    (feathered > 0) & (feathered < 1))),
                "fully_inside": int(np.count_nonzero(feathered == 1.0)),
            },
        })

    def _handle_paint(self, payload):
        h, w = int(payload["height"]), int(payload["width"])
        _, feathered, _ = build_selection(payload)
        canvas = Canvas(h, w)
        layer = canvas.new_layer("paint")
        pts = [(float(p[0]), float(p[1])) for p in payload.get("stroke", [])]
        if pts:
            color = payload.get("color", [0.9, 0.3, 0.1])
            radius = float(payload.get("brush", max(2.0, min(h, w) * 0.05)))
            paint_brush(canvas, layer, pts, radius=radius,
                        color_rgb=tuple(color), selection=feathered,
                        hardness=float(payload.get("hardness", 0.6)))
        rgba = np.round(layer.rgba * 255).astype(np.uint8)
        checker = np.round(
            __import__("selection_kernel.painting", fromlist=["composite_over_checker"])
            .composite_over_checker(layer.rgba) * 255).astype(np.uint8)
        self._send_json({
            "ok": True,
            "paint_png": _png_b64(rgba, "RGBA"),
            "checker_png": _png_b64(checker, "RGB"),
            "outside_max": float(layer.rgba[feathered == 0, 3].max())
            if np.any(feathered == 0) else 0.0,
        })

    def _handle_cut(self, payload):
        h, w = int(payload["height"]), int(payload["width"])
        hard, feathered, _ = build_selection(payload)
        canvas = Canvas(h, w)
        l1 = canvas.new_layer("l1")
        l2 = canvas.new_layer("l2")
        s1 = rasterize_paths(circle_path(w * 0.4, h * 0.5,
                                         min(h, w) * 0.28).flattened(), w, h)
        s2 = rasterize_paths(circle_path(w * 0.6, h * 0.5,
                                         min(h, w) * 0.28).flattened(), w, h)
        paint_brush(canvas, l1, [(w * 0.4, h * 0.5)], min(h, w) * 0.28,
                    (0.9, 0.2, 0.15), s1, hardness=0.5)
        paint_brush(canvas, l2, [(w * 0.6, h * 0.5)], min(h, w) * 0.28,
                    (0.1, 0.35, 0.9), s2, hardness=0.5)

        from selection_kernel.painting import Layer, composite_over_checker
        def clone(c):
            cc = Canvas(c.height, c.width)
            for ly in c.layers:
                cc.layers.append(Layer(ly.name, ly.rgba.copy(), ly.visible))
            return cc

        c1 = clone(canvas)
        cut1, src1 = cut_with_selection(c1, ["l1", "l2"], feathered, "merge_first")
        c2 = clone(canvas)
        cut2, _ = cut_with_selection(c2, ["l1", "l2"], feathered, "cut_each")
        diff, max_diff = __import__(
            "selection_kernel.painting", fromlist=["ownership_difference"]
        ).ownership_difference(canvas, ["l1", "l2"], feathered)
        diff_vis = np.round(np.clip(diff[..., :3] * 8.0, 0, 1) * 255).astype(np.uint8)
        self._send_json({
            "ok": True,
            "merge_first_png": _png_b64(
                np.round(composite_over_checker(cut1.rgba) * 255).astype(np.uint8), "RGB"),
            "cut_each_png": _png_b64(
                np.round(composite_over_checker(cut2.rgba) * 255).astype(np.uint8), "RGB"),
            "diff_png": _png_b64(diff_vis, "RGB"),
            "max_diff": max_diff,
            "hole_alpha_sum": float(sum(ly.rgba[..., 3].sum() for ly in src1)),
        })

    def _handle_tiles(self, payload):
        h, w = int(payload["height"]), int(payload["width"])
        _, feathered, polys = build_selection(payload)
        th = int(payload.get("tile_h", 37))
        tw = int(payload.get("tile_w", 41))
        err_hard = assert_tile_consistency(polys, h, w, th, tw, feather=0.0)
        err_feather = assert_tile_consistency(
            polys, h, w, th, tw, feather=float(payload.get("feather", 0.0)))
        self._send_json({"ok": True,
                         "tile_hard_maxdiff": err_hard,
                         "tile_feather_maxdiff": err_feather})

    def _handle_sample_plates(self, payload):
        """样册 -> 双色套版：真实写印版 PNG，再从文件读回油墨核验。"""
        sid = str(payload.get("id", "hole_filament"))
        spec = samplebook.load_sample(sid)
        recipe = spec.get("plates")
        if not recipe:
            raise ValueError(f"sample {sid!r} has no two-color plate recipe")
        os.makedirs(GUI_LAST_DIR, exist_ok=True)
        h = int(spec["canvas"]["height"])
        w = int(spec["canvas"]["width"])

        plate_a = samplebook.compose_spec(spec).hard.coverage
        plate_b = samplebook.compose_recipe(
            recipe["plate_b"]["items"], h, w).hard.coverage
        a_path = os.path.join(GUI_LAST_DIR, f"{sid}_plate_A.png")
        b_path = os.path.join(GUI_LAST_DIR, f"{sid}_plate_B.png")
        save_plate_ink_png(plate_a, a_path)
        save_plate_ink_png(plate_b, b_path)
        proof = overprint_proof([
            (plate_a, INK_COLORS[recipe["plate_a"]["ink"]]),
            (plate_b, INK_COLORS[recipe["plate_b"]["ink"]]),
        ])
        proof_u8 = np.round(proof * 255).astype(np.uint8)

        ink_a = load_plate_ink(a_path)  # 从实际文件读回
        readback = verify_filament_and_hole(
            ink_a, recipe["filament_roi"], recipe["hole_roi"])

        s = 4  # 换版：4x 矢量母形重新光栅化
        a4 = samplebook.compose_spec(spec, scale=s).hard.coverage
        a4_path = os.path.join(GUI_LAST_DIR, f"{sid}_plate_A_4x.png")
        save_plate_ink_png(a4, a4_path)
        readback4 = verify_filament_and_hole(
            load_plate_ink(a4_path), recipe["filament_roi"],
            recipe["hole_roi"], scale=s, filament_peak_min=0.5)

        self._send_json({
            "ok": True, "id": sid,
            "plate_a_png": _png_b64(
                np.round(plate_a * 255).astype(np.uint8), "L"),
            "plate_b_png": _png_b64(
                np.round(plate_b * 255).astype(np.uint8), "L"),
            "proof_png": _png_b64(proof_u8, "RGB"),
            "readback": readback,
            "readback_4x": readback4,
            "a_path": os.path.relpath(a_path, REPO_ROOT),
        })

    def _handle_legacy_mark(self, payload):
        """旧页：从入库的真实低清蒙版 PNG 读入并圈出不可恢复 ROI。"""
        sid = str(payload.get("id", "old_page_hole7"))
        record, record_path = samplebook.load_entry(sid)
        if record.get("kind") != "legacy":
            raise ValueError(f"{sid!r} is not a legacy mask entry")
        mask = samplebook.load_legacy_mask(record, record_path)
        rois = record["unrecoverable_rois"]
        roi = samplebook.roi_stats(mask, rois[0])
        marked = samplebook.mark_unrecoverable(mask, rois, upscale=4)
        os.makedirs(GUI_LAST_DIR, exist_ok=True)
        marked_path = os.path.join(GUI_LAST_DIR, f"{sid}_marked.png")
        Image.fromarray(marked, "RGB").save(marked_path)
        up = np.round(np.repeat(np.repeat(mask, 4, 0), 4, 1) * 255
                       ).astype(np.uint8)
        self._send_json({
            "ok": True, "id": sid,
            "source_png": os.path.relpath(
                samplebook.legacy_mask_path(record, record_path),
                REPO_ROOT),
            "mask_up_png": _png_b64(up, "L"),
            "marked_png": _png_b64(marked, "RGB"),
            "roi_ink": roi,
            "roi_max_tol": record.get("verification", {})
            .get("roi_ink_max_at_most", 1e-6),
            "provenance": record.get("provenance", {}),
        })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print("精细选区绘画窗口已启动：", url)
    print("按 Ctrl+C 停止。")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n停止。")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
