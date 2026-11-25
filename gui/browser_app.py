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

HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")

SHAPE_BUILDERS = {
    "circle": lambda x, y, w, h: circle_path(x + w / 2, y + h / 2,
                                             min(w, h) / 2),
    "rectangle": lambda x, y, w, h: rectangle_path(x, y, w, h),
    "slanted": lambda x, y, w, h: slanted_path(x, y, w, h, shear=0.25 * w),
    "slit": lambda x, y, w, h: thin_slit_path(x + w / 2, y + h / 2,
                                              max(0.2, min(0.6, 0.04 * w)), h),
}


def _png_b64(arr_uint8: np.ndarray, mode: str) -> str:
    img = Image.fromarray(arr_uint8, mode=mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _coverage_rgba(coverage: np.ndarray) -> np.ndarray:
    """把 float 覆盖率编码为灰度 RGBA（保留软边层次，不做阈值化）。"""
    u8 = np.round(np.clip(coverage, 0, 1) * 255).astype(np.uint8)
    rgba = np.zeros((*u8.shape, 4), dtype=np.uint8)
    rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = u8
    rgba[..., 3] = 255
    return rgba


def _edge_rgba(coverage: np.ndarray) -> np.ndarray:
    """边缘视图：只显示 0<c<1 的亚像素软边，红色越亮覆盖越高。"""
    soft = ((coverage > 0.0) & (coverage < 1.0)).astype(np.uint8)
    rgba = np.zeros((*coverage.shape, 4), dtype=np.uint8)
    vals = np.round(np.clip(coverage, 0, 1) * 255).astype(np.uint8)
    rgba[..., 0] = np.where(soft, vals, 0)
    rgba[..., 3] = np.where(soft, 255, 0)
    return rgba


def build_selection(payload: dict):
    """根据前端请求构造组合后的硬边与羽化覆盖率。"""
    h = int(payload["height"])
    w = int(payload["width"])
    feather = float(payload.get("feather", 0.0))
    if feather < 0 or feather > min(h, w):
        raise ValueError(f"feather radius out of range: {feather}")
    shapes = payload["shapes"]
    if not shapes:
        return (np.zeros((h, w), np.float32),
                np.zeros((h, w), np.float32), [])

    doc = SelectionDocument((h, w))
    all_polys = []
    for idx, sp in enumerate(shapes):
        kind = sp["shape"]
        x, y = float(sp["x"]), float(sp["y"])
        sw, sh = float(sp["w"]), float(sp["h"])
        if sw <= 0 or sh <= 0:
            raise ValueError("shape must have positive width and height")
        path = SHAPE_BUILDERS[kind](x, y, sw, sh)
        op = sp.get("op", CombineOp.UNION)
        sel = Selection(f"s{idx}", (h, w), [path])
        doc.add(sel)
        all_polys.extend(path.flattened())

    ops = [CombineOp(sp.get("op", CombineOp.UNION)) for sp in shapes[1:]]
    hard = doc.combined(ops).coverage
    sdf = polygon_signed_distance(all_polys, w, h, hard_cover=hard)
    feathered = feather_from_distance(sdf, feather, hard_cover=hard)
    return hard, feathered, all_polys


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
        if self.path in ("/", "/index.html"):
            with open(HTML_PATH, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._send_json({"ok": False, "error": "not found"}, 404)

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
