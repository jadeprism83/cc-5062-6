"""精细选区绘画窗口（Tkinter 桌面版，可选）。

启动前提：系统装有 python3-tk（Debian/Ubuntu: apt install python3-tk）。
启动：
    python3 gui/tk_app.py

本环境若无 tkinter 会给出明确提示并退出，请改用零依赖的浏览器版：
    python3 gui/browser_app.py
Tk 画布只承担显示与输入；覆盖率全部由内核在固定高分辨率坐标域计算，
PhotoImage 仅用于预览，导出走内核全分辨率路径，不用显示分辨率蒙版。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import tkinter as tk
    from tkinter import messagebox  # noqa: F401
except Exception:  # pragma: no cover - 环境缺 tkinter
    print("未找到 tkinter。请安装 python3-tk，或使用浏览器版：\n"
          "  python3 gui/browser_app.py", file=sys.stderr)
    raise SystemExit(1)

import numpy as np

from selection_kernel import (
    Canvas, CombineOp, Selection, SelectionDocument, circle_path,
    paint_brush, rectangle_path, slanted_path, thin_slit_path, export,
)
from selection_kernel.distance import feather_from_distance, polygon_signed_distance


SHAPES = {
    "circle": lambda x, y, w, h: circle_path(x + w / 2, y + h / 2, min(w, h) / 2),
    "rectangle": rectangle_path,
    "slanted": lambda x, y, w, h: slanted_path(x, y, w, h, 0.25 * w),
    "slit": lambda x, y, w, h: thin_slit_path(x + w / 2, y + h / 2,
                                              max(0.2, 0.04 * w), h),
}


class SelectionApp:
    def __init__(self, root: "tk.Tk", width: int = 240, height: int = 200):
        self.root = root
        self.w, self.h = width, height
        self.shapes = []          # (kind, op, x, y, w, h)
        self.feather = tk.DoubleVar(value=4.0)
        self.shape_var = tk.StringVar(value="circle")
        self.op_var = tk.StringVar(value="union")
        self.drag_start = None
        root.title("精细选区绘画窗口 (Tk)")

        ctrl = tk.Frame(root)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        for kind in SHAPES:
            tk.Radiobutton(ctrl, text=kind, variable=self.shape_var,
                           value=kind).pack(anchor="w")
        for op in ("union", "subtract", "intersect", "xor"):
            tk.Radiobutton(ctrl, text=op, variable=self.op_var,
                           value=op).pack(anchor="w")
        tk.Label(ctrl, text="羽化半径(画布px)").pack(anchor="w")
        tk.Scale(ctrl, variable=self.feather, from_=0, to=20,
                 resolution=0.25, orient=tk.HORIZONTAL,
                 command=lambda _e: self.render()).pack(anchor="w")
        tk.Button(ctrl, text="清空", command=self.clear).pack(fill=tk.X)
        tk.Button(ctrl, text="瓦片校验", command=self.tile_check).pack(fill=tk.X)
        tk.Button(ctrl, text="导出16位掩码", command=self.export_mask).pack(fill=tk.X)
        tk.Button(ctrl, text="绘制并导出绘画", command=self.paint_and_export).pack(fill=tk.X)
        self.status = tk.Label(ctrl, text="", wraplength=180, justify="left")
        self.status.pack(anchor="w")

        self.canvas = tk.Canvas(root, width=self.w, height=self.h,
                                bg="#222833", highlightthickness=1)
        self.canvas.pack(side=tk.RIGHT, padx=8, pady=8)
        self.canvas.bind("<ButtonPress-1>", self.on_down)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_up)
        self.image = tk.PhotoImage(width=self.w, height=self.h)
        self.canvas.create_image((0, 0), image=self.image, anchor="nw")
        self._last_coverage = None

    def on_down(self, e):
        self.drag_start = (e.x, e.y)

    def on_drag(self, e):
        if self.drag_start:
            self.canvas.delete("rect")
            x0, y0 = self.drag_start
            self.canvas.create_rectangle(x0, y0, e.x, e.y, outline="#5aa2ff",
                                         tags="rect")

    def on_up(self, e):
        if not self.drag_start:
            return
        x0, y0 = self.drag_start
        x, y = min(x0, e.x), min(y0, e.y)
        ww, hh = abs(e.x - x0), abs(e.y - y0)
        self.drag_start = None
        if ww >= 1 and hh >= 1:
            self.shapes.append((self.shape_var.get(), self.op_var.get(),
                                float(x), float(y), float(ww), float(hh)))
            self.render()

    def _build(self):
        doc = SelectionDocument((self.h, self.w))
        polys = []
        for i, (kind, _op, x, y, ww, hh) in enumerate(self.shapes):
            path = SHAPES[kind](x, y, ww, hh)
            doc.add(Selection(f"s{i}", (self.h, self.w), [path]))
            polys.extend(path.flattened())
        if not self.shapes:
            return np.zeros((self.h, self.w), np.float32), polys
        ops = [CombineOp(op) for _k, op, *_ in self.shapes[1:]]
        hard = doc.combined(ops).coverage
        sdf = polygon_signed_distance(polys, self.w, self.h, hard_cover=hard)
        return feather_from_distance(sdf, float(self.feather.get()),
                                     hard_cover=hard), polys

    def render(self):
        try:
            cov, _ = self._build()
        except Exception as exc:  # 失败反馈回界面
            messagebox.showerror("精细选区错误", str(exc))
            return
        self._last_coverage = cov
        self._put_coverage(cov)
        soft = int(np.count_nonzero((cov > 0) & (cov < 1)))
        self.status.config(text=f"Σc={cov.sum():.1f}px²\n软边像素={soft}")

    def _put_coverage(self, cov):
        u8 = np.round(np.clip(cov, 0, 1) * 255).astype(np.uint8)
        # PPM 字节流喂给 PhotoImage，避免 Tk  PhotoImage 对 numpy 的限制
        header = f"P5 {self.w} {self.h} 255 ".encode()
        self.image = tk.PhotoImage(data=header + u8.tobytes(), format="PPM")
        self.canvas.create_image((0, 0), image=self.image, anchor="nw")

    def clear(self):
        self.shapes = []
        self.render()

    def tile_check(self):
        from selection_kernel import assert_tile_consistency
        cov, polys = self._build()
        if not polys:
            return
        err = assert_tile_consistency(polys, self.h, self.w, 37, 41,
                                      feather=float(self.feather.get()))
        self.status.config(text=f"瓦片最大差异={err:.1e}")

    def export_mask(self):
        if self._last_coverage is None:
            self.render()
        path = os.path.abspath("output/tk_selection_mask.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        export.save_mask_png(self._last_coverage, path)
        self.status.config(text=f"已导出 {path}")

    def paint_and_export(self):
        cov, _ = self._build()
        canvas = Canvas(self.h, self.w)
        layer = canvas.new_layer("paint")
        pts = [(self.w * (0.15 + 0.7 * t),
                self.h * 0.5 + 18 * np.sin(t * 6.283))
               for t in np.linspace(0, 1, 30)]
        paint_brush(canvas, layer, pts, radius=10.0,
                    color_rgb=(0.9, 0.25, 0.15), selection=cov, hardness=0.6)
        out = os.path.abspath("output/tk_paint_rgba.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        export.save_rgba(layer.rgba, out)
        outside = float(layer.rgba[cov == 0, 3].max()) if np.any(cov == 0) else 0.0
        self.status.config(text=f"绘画已导出 {out}\n选区外alpha={outside:.1e}")


def main() -> int:
    root = tk.Tk()
    SelectionApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
