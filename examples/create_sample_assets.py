"""创作选区样册的原始素材（一次性/可重复运行的入库工具）。

运行：
    python3 examples/create_sample_assets.py

产出真实入库文件到 selection_samples/：
- outlines/*.json        矢量母形（原路径命令 + 组合算子 + 羽化 + 参考约定）
- projects/*.json        有洞/细丝/半透明内容的绘画工程参数
- legacy/*.json + .png   只留低清蒙版的旧页（模拟旧导出链，来源如实标注）

参考覆盖面积一律由解析几何公式（矩形面积、πr²、相交矩形边长）独立给出，
不读取内核光栅化结果，因此 build_samplebook.py 的核验是真正的交叉检查。
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from selection_kernel.geometry import circle_path, rectangle_path  # noqa: E402
from selection_kernel.pathio import path_to_dict  # noqa: E402
from selection_kernel.samplebook import (  # noqa: E402
    box_downsample, compose_spec,
)

SAMPLE_ROOT = os.path.join(ROOT, "selection_samples")
TODAY = "2026-11-25"


def item(iid, op, path, **extra):
    d = {"id": iid, "op": op, "path": path_to_dict(path)}
    d.update(extra)
    return d


def write_json(rel_path, obj):
    p = os.path.join(SAMPLE_ROOT, rel_path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print("wrote", os.path.relpath(p, ROOT))


def hole_filament_spec():
    w = hf = 0.32
    hair_len = 84.0
    hard_ref = 180.0 * 140.0 - math.pi * 42.0 ** 2 + hair_len * hf
    return {
        "schema": "selection-sample/v1",
        "id": "hole_filament",
        "title": "孔洞与孔边细丝",
        "description": (
            "矩形版材 subtract 圆孔，再 union 一根横跨空孔、宽 0.32px 的细丝。"
            "4x 面积平均预览下细丝峰值仅 0.04，低于可见阈值 0.05，刚好消失；"
            "以 4 倍输出倍率从矢量母形重新光栅化后，细丝峰值 0.64，油墨质量"
            "约为母版的 16 倍。孔内设有远离细丝的空孔核验区。"
        ),
        "tags": ["hole", "filament", "subpixel", "resolution", "union",
                 "subtract", "duotone"],
        "canvas": {"width": 240, "height": 200},
        "feather": 0.0,
        "items": [
            item("plate", "union", rectangle_path(30, 30, 180, 140)),
            item("round_hole", "subtract", circle_path(120, 100, 42)),
            item("filament", "union",
                 rectangle_path(78, 100 - hf / 2, hair_len, hf),
                 resolution_role="filament"),
        ],
        "reference": {
            "hard_mass": {
                "expected": round(hard_ref, 4),
                "tol_abs": 1.5,
                "basis": "180*140 - pi*42^2 + 84*0.32；细丝端点与孔缘抗锯齿"
                         "覆盖按 c=a(1-b) 相乘，约 1px^2 偏差计入容差",
            },
            "rois": {
                "filament_corridor": {
                    "xyxy": [84, 99, 156, 101],
                    "max_min": 0.159,
                    "sum_expected": 23.04,
                    "sum_tol": 0.2,
                    "basis": "0.32px 宽居中 y=100，分占 99/100 两行各 0.16",
                },
                "hole_empty": {
                    "xyxy": [104, 66, 136, 86],
                    "max_at_most": 1e-6,
                    "basis": "圆孔深处、细丝上方，几何上完全在洞内",
                },
            },
        },
        "resolution_lab": {
            "preview_factor": 4,
            "visibility_tau": 0.05,
            "preview_filament_roi": [21, 24, 39, 26],
            "preview_filament_max_below": 0.05,
            "output_scale": 4,
            "output_filament_peak_min": 0.5,
            "mass_scale_tol_rel": 0.02,
        },
        "plates": {
            "plate_a": {"ink": "vermilion", "use_sample_coverage": True},
            "plate_b": {
                "ink": "indigo",
                "items": [
                    item("halo_outer", "union", circle_path(120, 100, 60)),
                    item("halo_inner", "subtract", circle_path(120, 100, 54)),
                ],
            },
            "filament_roi": [84, 99, 156, 101],
            "hole_roi": [104, 66, 136, 86],
        },
        "provenance": {
            "source": "hand-authored for selection sample book",
            "authoring_script": "examples/create_sample_assets.py",
            "created": TODAY,
            "coordinate_space": "240x200 canvas pixels, resolution independent",
            "params": {"hole_center": [120, 100], "hole_r": 42,
                       "filament_width": hf, "filament_length": hair_len},
        },
    }
    return spec


def subpixel_lines_spec():
    lines = [
        ("hair_025", 30, 39.875, 180, 0.25),
        ("hair_050", 30, 70.0, 180, 0.50),
        ("hair_075", 30, 99.625, 180, 0.75),
        ("top_aligned_040", 30, 150.0, 120, 0.40),
    ]
    vertical = ("vertical_035", 59.825, 160.0, 0.35, 25.0)
    total = sum(w * h for _, _, _, w, h in lines)
    total += vertical[3] * vertical[4]
    return {
        "schema": "selection-sample/v1",
        "id": "subpixel_lines",
        "title": "亚像素细线家族",
        "description": (
            "宽度 0.25/0.35/0.4/0.5/0.75px 的水平与竖直线，全部小于一个像素，"
            "没有任何像素被完全覆盖；总面积严格等于线宽×线长。边缘颜色探针"
            "使用边的分数坐标给出逐像素精确覆盖率约定。"
        ),
        "tags": ["subpixel", "thin-line", "edge-color"],
        "canvas": {"width": 240, "height": 200},
        "feather": 0.0,
        "items": [item(i, "union", rectangle_path(x, y, w, h))
                  for i, x, y, w, h in lines] + [
            item(vertical[0], "union",
                 rectangle_path(vertical[1], vertical[2],
                                vertical[3], vertical[4]))
        ],
        "reference": {
            "hard_mass": {"expected": round(total, 4), "tol_abs": 0.5,
                          "basis": "各细线 w*h 之和，互不相交，union 即面积和"},
            "hard_max_below": 0.76,
            "edge_pixels": [
                {"pixel": [50, 39], "expected_hard": 0.125, "tol": 0.01,
                 "why": "y=39.875 的 0.25 线，第 39 行覆盖 0.125"},
                {"pixel": [50, 70], "expected_hard": 0.5, "tol": 0.01,
                 "why": "y=70 起 0.5 高，第 70 行覆盖 0.5"},
                {"pixel": [50, 99], "expected_hard": 0.375, "tol": 0.01,
                 "why": "y=99.625 的 0.75 线，第 99 行覆盖 0.375"},
                {"pixel": [50, 150], "expected_hard": 0.4, "tol": 0.01,
                 "why": "y=150 起 0.4 高，第 150 行覆盖 0.4"},
                {"pixel": [59, 170], "expected_hard": 0.175, "tol": 0.01,
                 "why": "x=59.825 的 0.35 竖线，第 59 列覆盖 0.175"},
            ],
        },
        "provenance": {
            "source": "hand-authored edge-color convention sheet",
            "authoring_script": "examples/create_sample_assets.py",
            "created": TODAY,
        },
    }


def repeat_spokes_spec():
    n = 12
    bar_w, bar_h = 6.0, 54.0
    return {
        "schema": "selection-sample/v1",
        "id": "repeat_spokes",
        "title": "重复变换：十二辐条",
        "description": (
            "一根 6x54 辐条绕 (120,100) 每 30° 旋转复制，共 12 份；"
            "所有副本由同一原路径经 rotation_matrix 现场变换得到。"
            "辐条从半径 26 延伸到 80，内毂留空，互不重叠，总覆盖面积=12*6*54。"
        ),
        "tags": ["repeat", "transform", "rotation"],
        "canvas": {"width": 240, "height": 200},
        "feather": 1.2,
        "items": [
            item("spoke", "union",
                 rectangle_path(117, 20, bar_w, bar_h),
                 repeat={"count": n, "center": [120, 100],
                         "angle_start_deg": 0, "angle_step_deg": 30}),
        ],
        "reference": {
            "hard_mass": {"expected": n * bar_w * bar_h, "tol_abs": 3.0,
                          "basis": "12 * 6 * 54，旋转保面积且辐条互不重叠"},
            "rois": {
                "empty_hub": {"xyxy": [112, 92, 128, 108],
                              "max_at_most": 1e-6,
                              "basis": "内半径 26 以内无辐条"},
            },
        },
        "provenance": {
            "source": "hand-authored repeat-transform sheet",
            "authoring_script": "examples/create_sample_assets.py",
            "created": TODAY,
            "params": {"count": n, "step_deg": 30, "bar": [bar_w, bar_h]},
        },
    }


def intersect_narrow_spec():
    # 交集 x: 92.4 .. 120.35 (27.95), y: 58.25 .. 150.25 (92.0)
    mass = 27.95 * 92.0
    return {
        "schema": "selection-sample/v1",
        "id": "intersect_narrow",
        "title": "相交轮廓与窄羽化",
        "description": (
            "两个带分数坐标边的矩形做 intersect(c=a*b)，交集是边边精确的"
            "27.95x92 矩形；0.8px 窄羽化只沿组合结果的真实边界展开"
            "（edge-aware 边界段筛选排除了落在交集外的轮廓边，无假光晕）。"
            "四个边缘探针给出精确覆盖率与边缘 PNG 颜色约定。"
        ),
        "tags": ["intersect", "narrow-feather", "edge-color"],
        "canvas": {"width": 240, "height": 200},
        "feather": 0.8,
        "items": [
            item("a", "union", rectangle_path(36.35, 36.6, 84, 128)),
            item("b", "intersect", rectangle_path(92.4, 58.25, 110, 92)),
        ],
        "reference": {
            "hard_mass": {"expected": round(mass, 4), "tol_abs": 0.2,
                          "basis": "交集宽 120.35-92.4=27.95，"
                                   "高 150.25-58.25=92.0"},
            "edge_pixels": [
                {"pixel": [92, 100], "expected_hard": 0.6, "tol": 0.01,
                 "expect_rgb": [153, 0, 0, 255],
                 "why": "左边界 x=92.4，第 92 列覆盖 0.6"},
                {"pixel": [120, 100], "expected_hard": 0.35, "tol": 0.01,
                 "expect_rgb": [89, 0, 0, 255],
                 "why": "右边界 x=120.35，第 120 列覆盖 0.35"},
                {"pixel": [105, 58], "expected_hard": 0.75, "tol": 0.01,
                 "expect_rgb": [191, 0, 0, 255],
                 "why": "上边界 y=58.25，第 58 行覆盖 0.75"},
                {"pixel": [105, 150], "expected_hard": 0.25, "tol": 0.01,
                 "expect_rgb": [64, 0, 0, 255],
                 "why": "下边界 y=150.25，第 150 行覆盖 0.25"},
            ],
            "feather_checks": {
                "radius": 0.8,
                "center_pixel": [105, 100],
                "center_min": 0.999,
                "ghost_edge_roi": [118, 42, 123, 50],
                "ghost_edge_max_at_most": 0.02,
                "basis": "该 ROI 贴着 A 矩形位于交集上方的废边；"
                         "edge-aware 羽化在该处必须为 0",
            },
        },
        "provenance": {
            "source": "hand-authored intersection edge-color sheet",
            "authoring_script": "examples/create_sample_assets.py",
            "created": TODAY,
        },
    }


def painting_project():
    return {
        "schema": "painting-project/v1",
        "id": "hole_still_life",
        "title": "有洞、细丝与半透明内容的水彩工程",
        "description": (
            "载入 hole_filament 选区并施加 1.6px 羽化；两层笔画（0.42 透明度"
            "淡水彩 + 0.9 透明度深线）真实调用 paint_brush，写入=笔画覆盖×"
            "选区覆盖×不透明度。笔画扫过空孔时被选区严格挡住。工程 JSON "
            "保存全部笔画生成参数，build_samplebook.py 可重复重建导出图。"
            "这不是抠好的成品图，而是可编辑、可重建的绘画工程。"
        ),
        "canvas": {"width": 240, "height": 200},
        "selection_sample": "hole_filament",
        "feather": 1.6,
        "layers": [
            {
                "name": "wash",
                "strokes": [{
                    "generator": "wave",
                    "radius": 22.0,
                    "hardness": 0.55,
                    "opacity": 0.42,
                    "color": [0.20, 0.45, 0.85],
                    "params": {"x0": 36, "x1": 204, "y": 104,
                               "amp": 15, "cycles": 1.5, "n": 46},
                }],
            },
            {
                "name": "ink_line",
                "strokes": [{
                    "generator": "wave",
                    "radius": 6.5,
                    "hardness": 0.8,
                    "opacity": 0.9,
                    "color": [0.85, 0.22, 0.12],
                    "params": {"x0": 40, "x1": 200, "y": 95,
                               "amp": 11, "cycles": 2.0, "n": 60},
                }],
            },
        ],
        "verify": {
            "hole_roi": [104, 66, 136, 86],
            "outside_alpha_tol": 0.0,
            "semi_transparent_pixels_min": 200,
        },
        "provenance": {
            "source": "hand-authored editable painting project",
            "authoring_script": "examples/create_sample_assets.py",
            "created": TODAY,
        },
    }


def make_legacy_page():
    """从 hero 母版模拟旧导出链：4x 面积平均 + 0.5 二值阈值。"""
    spec = hole_filament_spec()
    res = compose_spec(spec, feather=0.0, edge_aware=False)
    prev = box_downsample(res.hard.coverage, 4)
    binary = (prev >= 0.5).astype(np.uint8) * 255
    lp = os.path.join(SAMPLE_ROOT, "legacy", "old_page_hole7_mask.png")
    os.makedirs(os.path.dirname(lp), exist_ok=True)
    Image.fromarray(binary, mode="L").save(lp)
    print("wrote", os.path.relpath(lp, ROOT), binary.shape)

    record = {
        "schema": "legacy-mask/v1",
        "id": "old_page_hole7",
        "kind": "legacy",
        "title": "旧样册第 7 页：仅存 60x50 低清蒙版",
        "description": (
            "当年的导出链先做 4x 面积平均再按 0.5 阈值二值化；孔边 0.32px "
            "细丝在低清块中峰值仅 0.04，阈值后为 0，矢量母形已佚。"
            "unrecoverable_rois 是当年编辑笔记留存的细丝走廊（低清坐标），"
            "标记程序只圈出不可恢复区域，不虚构其中形状。"
        ),
        "canvas": {"width": 60, "height": 50},
        "mask_png": "old_page_hole7_mask.png",
        "unrecoverable_rois": [[21, 24, 39, 26]],
        "verification": {
            "roi_ink_max_at_most": 1e-6,
            "companion_vector_sample": "hole_filament",
            "note": "现代伴随矢量样册仅用于交叉验证标记确实命中真实佚失特征，"
                    "不参与旧页标记本身",
        },
        "provenance": {
            "source": "simulated legacy page produced from hole_filament",
            "authoring_script": "examples/create_sample_assets.py",
            "created": TODAY,
            "vector_master": "lost",
            "process": "4x box area downsample then coverage >= 0.5 binary",
        },
    }
    write_json(os.path.join("legacy", "old_page_hole7.json"), record)


def main():
    os.makedirs(os.path.join(SAMPLE_ROOT, "outlines"), exist_ok=True)
    os.makedirs(os.path.join(SAMPLE_ROOT, "projects"), exist_ok=True)
    write_json(os.path.join("outlines", "hole_filament.json"),
               hole_filament_spec())
    write_json(os.path.join("outlines", "subpixel_lines.json"),
               subpixel_lines_spec())
    write_json(os.path.join("outlines", "repeat_spokes.json"),
               repeat_spokes_spec())
    write_json(os.path.join("outlines", "intersect_narrow.json"),
               intersect_narrow_spec())
    write_json(os.path.join("projects", "hole_still_life.json"),
               painting_project())
    make_legacy_page()
    index = {
        "schema": "selection-catalog/v1",
        "updated": TODAY,
        "gui_entry": "python3 gui/browser_app.py  → 样册与套版标签",
        "samples": [
            {"id": "hole_filament", "kind": "selection",
             "file": "outlines/hole_filament.json",
             "title": "孔洞与孔边细丝",
             "tags": ["hole", "filament", "resolution", "duotone"]},
            {"id": "subpixel_lines", "kind": "selection",
             "file": "outlines/subpixel_lines.json",
             "title": "亚像素细线家族",
             "tags": ["subpixel", "edge-color"]},
            {"id": "repeat_spokes", "kind": "selection",
             "file": "outlines/repeat_spokes.json",
             "title": "重复变换：十二辐条",
             "tags": ["repeat", "rotation"]},
            {"id": "intersect_narrow", "kind": "selection",
             "file": "outlines/intersect_narrow.json",
             "title": "相交轮廓与窄羽化",
             "tags": ["intersect", "feather", "edge-color"]},
            {"id": "hole_still_life", "kind": "project",
             "file": "projects/hole_still_life.json",
             "title": "孔洞细丝半透明绘画工程",
             "tags": ["painting", "semi-transparent"]},
            {"id": "old_page_hole7", "kind": "legacy",
             "file": "legacy/old_page_hole7.json",
             "title": "旧样册第 7 页（低清蒙版）",
             "tags": ["legacy", "unrecoverable"]},
        ],
    }
    write_json("index.json", index)
    print("sample assets created under", SAMPLE_ROOT)


if __name__ == "__main__":
    main()
