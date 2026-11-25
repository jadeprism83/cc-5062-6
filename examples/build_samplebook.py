"""选区样册重建、核验与双色套版导出（命令行辅助入口）。

    python3 examples/build_samplebook.py
    python3 examples/build_samplebook.py --outdir /tmp/samplebook_build

它做的每一件事都走 selection_kernel 的真实实现：
1. 从 selection_samples/ 载入入库的原路径 JSON，重建覆盖率；
2. 用独立解析参考（面积公式/约定边缘色）核验覆盖量与导出 PNG 的边缘颜色；
3. 分辨率实验：预览降采样细丝消失 -> 4x 矢量母形重新光栅化细丝回归；
4. 双色套版：写印版 PNG -> 从实际文件读回油墨 -> 读细丝/查空孔，
   并故意做一张补满孔洞的坏版对照，证明检查有效；
5. 旧页：读真实低清蒙版 PNG，圈出不可恢复 ROI 并核验该处无墨；
6. 半透明绘画工程：按 project JSON 真实绘制并核验选区外 alpha=0；
7. 四种连续覆盖率组合语义导出。

所有数值写入 <outdir>/evidence.json 与 evidence.md；任一核验失败则
进程以非零码退出，证据不允许伪报。图形界面是这些能力的主入口
（python3 gui/browser_app.py 的“样册与套版”标签），本脚本仅辅助。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from selection_kernel import export  # noqa: E402
from selection_kernel.geometry import rectangle_path  # noqa: E402
from selection_kernel.mask import CombineOp, CoverageMask, combine  # noqa: E402
from selection_kernel.painting import Canvas, paint_brush  # noqa: E402
from selection_kernel.rasterize import rasterize_paths  # noqa: E402
from selection_kernel.plates import (  # noqa: E402
    INK_COLORS, fill_enclosed_holes, load_plate_ink, overprint_proof,
    plate_sheet, save_film_positive, save_plate_ink_png, save_rgb,
    verify_filament_and_hole,
)
from selection_kernel.samplebook import (  # noqa: E402
    box_downsample, catalog_entries, compose_recipe, compose_spec,
    load_entry, load_legacy_mask, load_sample, mark_unrecoverable,
    roi_stats, sample_dir,
)
from selection_kernel.visualize import (  # noqa: E402
    edge_color_rgba, edge_expected_rgba, png_bytes,
)

DEFAULT_OUT = os.path.join(ROOT, "output", "samplebook")


def _rel(p: str) -> str:
    try:
        return os.path.relpath(p, ROOT)
    except ValueError:
        return p


def _sha256(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _ok(checks: List[bool]) -> bool:
    return all(bool(x) for x in checks)


def _save_edge_png(coverage: np.ndarray, path: str) -> None:
    with open(path, "wb") as f:
        f.write(png_bytes(edge_color_rgba(coverage), "RGBA"))


def _read_png_pixel(path: str, i: int, j: int) -> Tuple[int, ...]:
    with Image.open(path) as im:
        return tuple(im.convert("RGBA").getpixel((i, j)))


# --------------------------------------------------------------------------
# 1) 选区样例重建 + 覆盖量/边缘颜色核验
# --------------------------------------------------------------------------

def verify_selection_sample(sid: str, outdir: str) -> Dict[str, Any]:
    spec = load_sample(sid)
    canvas = spec["canvas"]
    res = compose_spec(spec)
    hard = res.hard.coverage
    feathered = res.coverage
    ref = spec.get("reference", {})
    ev: Dict[str, Any] = {"id": sid, "title": spec.get("title"), "checks": []}
    passed: List[bool] = []

    export.save_mask_png(hard, os.path.join(outdir, f"{sid}_hard_u16.png"))
    export.save_mask_png(feathered, os.path.join(outdir, f"{sid}_feather_u16.png"))
    edge_hard_path = os.path.join(outdir, f"{sid}_edge_hard.png")
    edge_fez_path = os.path.join(outdir, f"{sid}_edge_feather.png")
    _save_edge_png(hard, edge_hard_path)
    _save_edge_png(feathered, edge_fez_path)

    mass = float(hard.sum())
    rm = ref.get("hard_mass")
    if rm:
        d = abs(mass - rm["expected"])
        ok = d <= rm["tol_abs"]
        passed.append(ok)
        ev["hard_mass"] = {"measured": mass, "expected": rm["expected"],
                           "abs_diff": abs(mass - rm["expected"]),
                           "tol_abs": rm["tol_abs"], "pass": ok,
                           "basis": rm.get("basis")}

    if "hard_max_below" in ref:
        mx = float(hard.max())
        ok = mx < ref["hard_max_below"]
        passed.append(ok)
        ev["hard_max"] = {"measured": mx,
                          "below": ref["hard_max_below"], "pass": ok}

    edge_checks = []
    for ep in ref.get("edge_pixels", []):
        i, j = ep["pixel"]
        c = float(hard[j, i])
        cov_ok = abs(c - ep["expected_hard"]) <= ep["tol"]
        expect_rgb = tuple(ep.get("expect_rgb")
                           or edge_expected_rgba(ep["expected_hard"]))
        actual_rgb = _read_png_pixel(edge_hard_path, i, j)
        color_err = max(abs(int(a) - int(b))
                        for a, b in zip(actual_rgb, expect_rgb))
        color_ok = color_err <= 2
        passed.extend([cov_ok, color_ok])
        edge_checks.append({
            "pixel": [i, j], "measured_hard": c,
            "expected_hard": ep["expected_hard"], "coverage_pass": cov_ok,
            "edge_png": _rel(edge_hard_path),
            "actual_rgba": list(actual_rgb), "expected_rgba": list(expect_rgb),
            "color_max_err": color_err, "color_pass": color_ok,
            "why": ep.get("why"),
        })
    if edge_checks:
        ev["edge_pixels"] = edge_checks

    roi_ev = {}
    for name, rv in ref.get("rois", {}).items():
        st = roi_stats(hard, rv["xyxy"])
        checks = []
        if "max_at_most" in rv:
            checks.append(st["max"] <= rv["max_at_most"] + 1e-7)
        if "max_min" in rv:
            checks.append(st["max"] >= rv["max_min"] - 1e-7)
        if "sum_expected" in rv:
            checks.append(abs(st["sum"] - rv["sum_expected"])
                          <= rv.get("sum_tol", 0.2))
        ok = _ok(checks)
        passed.append(ok)
        roi_ev[name] = {**st, "pass": ok, "convention": {
            k: v for k, v in rv.items() if k != "xyxy"},
            "xyxy": rv["xyxy"]}
    if roi_ev:
        ev["rois"] = roi_ev

    fc = ref.get("feather_checks")
    if fc:
        i, j = fc["center_pixel"]
        center = float(feathered[j, i])
        ghost = roi_stats(feathered, fc["ghost_edge_roi"])
        ok1 = center >= fc["center_min"]
        ok2 = ghost["max"] <= fc["ghost_edge_max_at_most"] + 1e-7
        passed.extend([ok1, ok2])
        ev["feather_checks"] = {"radius": fc.get("radius"),
                                "center_value": center,
                                "center_pass": ok1,
                                "ghost_roi": ghost,
                                "ghost_pass": ok2 and True,
                                "ghost_threshold": fc["ghost_edge_max_at_most"],
                                "pass": ok1 and ok2}

    ev["files"] = {
        "hard_u16": _rel(os.path.join(outdir, f"{sid}_hard_u16.png")),
        "feather_u16": _rel(os.path.join(outdir, f"{sid}_feather_u16.png")),
        "edge_hard": _rel(edge_hard_path),
        "edge_feather": _rel(edge_fez_path),
    }
    ev["pass"] = _ok(passed)
    ev["canvas"] = [canvas["width"], canvas["height"]]
    return ev


# --------------------------------------------------------------------------
# 2) 分辨率实验：预览消失 + 4x 矢量重新光栅化
# --------------------------------------------------------------------------

def resolution_lab(outdir: str) -> Dict[str, Any]:
    sid = "hole_filament"
    spec = load_sample(sid)
    lab = spec["resolution_lab"]
    res1 = compose_spec(spec)
    hard1 = res1.hard.coverage
    f = int(lab["preview_factor"])
    preview = box_downsample(hard1, f)
    prev_path = os.path.join(outdir, f"{sid}_preview_{preview.shape[1]}x{preview.shape[0]}.png")
    export.save_mask_u8_png(preview, prev_path)
    prev_fil = roi_stats(preview, lab["preview_filament_roi"])
    prev_ok = prev_fil["max"] < lab["preview_filament_max_below"]

    s = int(lab["output_scale"])
    res4 = compose_spec(spec, scale=s)
    hard4 = res4.hard.coverage
    m4_path = os.path.join(outdir, f"{sid}_master_{s}x_mask_u16.png")
    export.save_mask_png(hard4, m4_path)
    fil4 = roi_stats(hard4, spec["plates"]["filament_roi"], scale=s)
    hole4 = roi_stats(hard4, spec["plates"]["hole_roi"], scale=s)
    peak_ok = fil4["max"] >= lab["output_filament_peak_min"]
    ratio = float(hard4.sum()) / float(hard1.sum())
    ratio_ok = abs(ratio - s * s) <= s * s * lab["mass_scale_tol_rel"]
    hole_ok = hole4["max"] <= 0.02

    ev = {"preview_factor": f,
          "visibility_tau": lab["visibility_tau"],
          "preview_filament": prev_fil,
          "preview_disappears": bool(prev_ok),
          "preview_png": _rel(prev_path),
          "output_scale": s,
          "output_filament_peak_min": lab["output_filament_peak_min"],
          "output_filament": fil4,
          "output_peak_recovers": bool(peak_ok),
          "mass_ratio": ratio, "mass_ratio_expected": s * s,
          "mass_ratio_pass": bool(ratio_ok),
          "output_hole": hole4, "output_hole_empty": bool(hole_ok),
          "master4x_png": _rel(m4_path)}
    ev["pass"] = _ok([prev_ok, peak_ok, ratio_ok, hole_ok])
    return ev


# --------------------------------------------------------------------------
# 3) 双色套版 + 实际油墨回读
# --------------------------------------------------------------------------

def plates_lab(outdir: str) -> Dict[str, Any]:
    pdir = os.path.join(outdir, "plates")
    os.makedirs(pdir, exist_ok=True)
    spec = load_sample("hole_filament")
    recipe = spec["plates"]
    h = spec["canvas"]["height"]
    w = spec["canvas"]["width"]

    res_a = compose_spec(spec)
    plate_a = res_a.hard.coverage
    res_b = compose_recipe(recipe["plate_b"]["items"], h, w)
    plate_b = res_b.hard.coverage

    a_path = os.path.join(pdir, "plate_A_vermilion_ink.png")
    b_path = os.path.join(pdir, "plate_B_indigo_ink.png")
    save_plate_ink_png(plate_a, a_path)
    save_plate_ink_png(plate_b, b_path)
    save_film_positive(plate_a, os.path.join(pdir, "film_A_black.png"))
    save_film_positive(plate_b, os.path.join(pdir, "film_B_black.png"))
    proof = overprint_proof([
        (plate_a, INK_COLORS[recipe["plate_a"]["ink"]]),
        (plate_b, INK_COLORS[recipe["plate_b"]["ink"]]),
    ])
    proof_path = os.path.join(pdir, "proof_overprint.png")
    save_rgb(proof, proof_path)
    Image.fromarray(plate_sheet(
        plate_a, "PLATE A vermilion",
        INK_COLORS[recipe["plate_a"]["ink"]]), "RGB").save(
        os.path.join(pdir, "sheet_plate_A.png"))
    Image.fromarray(plate_sheet(
        plate_b, "PLATE B indigo",
        INK_COLORS[recipe["plate_b"]["ink"]]), "RGB").save(
        os.path.join(pdir, "sheet_plate_B.png"))

    # 关键：从实际写盘的印版 PNG 读回油墨，再读细丝/查空孔
    ink_a = load_plate_ink(a_path)
    readback = verify_filament_and_hole(
        ink_a, recipe["filament_roi"], recipe["hole_roi"])

    # 换版：4x 矢量母形重新出正式印版
    s = 4
    a4 = compose_spec(spec, scale=s).hard.coverage
    b4 = compose_recipe(recipe["plate_b"]["items"],
                        h * s, w * s, scale=s).hard.coverage
    a4_path = os.path.join(pdir, "plate_A_vermilion_ink_4x.png")
    b4_path = os.path.join(pdir, "plate_B_indigo_ink_4x.png")
    save_plate_ink_png(a4, a4_path)
    save_plate_ink_png(b4, b4_path)
    save_rgb(overprint_proof([
        (a4, INK_COLORS[recipe["plate_a"]["ink"]]),
        (b4, INK_COLORS[recipe["plate_b"]["ink"]]),
    ]), os.path.join(pdir, "proof_overprint_4x.png"))
    ink_a4 = load_plate_ink(a4_path)
    readback4 = verify_filament_and_hole(
        ink_a4, recipe["filament_roi"], recipe["hole_roi"],
        scale=s, filament_peak_min=0.5)

    # 坏版对照：二值化 + 补满封闭孔洞，检查器必须判失败
    bad = fill_enclosed_holes(plate_a)
    bad_path = os.path.join(pdir, "control_bad_filled_plate_A.png")
    save_plate_ink_png(bad, bad_path)
    bad_ink = load_plate_ink(bad_path)
    bad_check = verify_filament_and_hole(
        bad_ink, recipe["filament_roi"], recipe["hole_roi"])
    control_works = (not bad_check["hole_empty"]) and bad_check["filament_present"]

    ev = {
        "plate_a_png": _rel(a_path), "plate_b_png": _rel(b_path),
        "proof_png": _rel(proof_path),
        "readback_from_actual_file": readback,
        "plate_4x_png": _rel(a4_path),
        "readback_4x_from_actual_file": readback4,
        "bad_control_png": _rel(bad_path),
        "bad_control_report": bad_check,
        "bad_control_detected": bool(control_works),
        "pass": _ok([readback["pass"], readback4["pass"], control_works]),
    }
    return ev


# --------------------------------------------------------------------------
# 4) 旧页不可恢复标记
# --------------------------------------------------------------------------

def legacy_lab(outdir: str) -> Dict[str, Any]:
    ldir = os.path.join(outdir, "legacy")
    os.makedirs(ldir, exist_ok=True)
    legacy_entries = catalog_entries("legacy")
    ev_all = {"entries": []}
    all_pass = True
    for entry in legacy_entries:
        record, record_path = load_entry(entry["id"])
        mask = load_legacy_mask(record, record_path)  # 从真实 PNG 读
        source_png = os.path.join(os.path.dirname(record_path),
                                  record["mask_png"])
        rois = record["unrecoverable_rois"]
        ink = roi_stats(mask, rois[0])
        tol = record.get("verification", {}).get("roi_ink_max_at_most", 1e-6)
        ink_ok = ink["max"] <= tol
        marked = mark_unrecoverable(mask, rois, upscale=4)
        marked_path = os.path.join(ldir, f"{record['id']}_marked.png")
        Image.fromarray(marked, "RGB").save(marked_path)

        companion = record.get("verification", {}).get("companion_vector_sample")
        companion_preview_max = None
        if companion:
            cspec = load_sample(companion)
            chard = compose_spec(cspec).hard.coverage
            cprev = box_downsample(chard, 4)
            companion_preview_max = float(
                roi_stats(cprev, rois[0])["max"])

        passed = bool(ink_ok)
        all_pass &= passed
        ev_all["entries"].append({
            "id": record["id"],
            "source_json": _rel(record_path),
            "source_png": _rel(source_png),
            "source_png_exists": os.path.exists(source_png),
            "source_png_sha256_16": _sha256(source_png),
            "mask_shape": list(mask.shape),
            "unrecoverable_rois_lowres": rois,
            "roi_ink_from_actual_png": ink,
            "roi_ink_max_tol": tol,
            "roi_empty_in_legacy": bool(ink_ok),
            "companion_vector_preview_max": companion_preview_max,
            "companion_proves_lost_feature": bool(
                companion_preview_max is not None
                and companion_preview_max > 0.01 and ink["max"] == 0.0),
            "marked_png": _rel(marked_path),
            "pass": passed,
        })
    ev_all["pass"] = all_pass
    return ev_all


# --------------------------------------------------------------------------
# 5) 半透明绘画工程
# --------------------------------------------------------------------------

def _wave_points(params: Dict[str, Any]) -> List[Tuple[float, float]]:
    ts = np.linspace(0.0, 1.0, int(params.get("n", 40)))
    x0, x1 = float(params["x0"]), float(params["x1"])
    y0, amp = float(params["y"]), float(params["amp"])
    cycles = float(params.get("cycles", 1.0))
    return [(float(x), float(y0 + amp * np.sin(2 * np.pi * cycles * t)))
            for t, x in zip(ts, x0 + (x1 - x0) * ts)]


def painting_lab(outdir: str) -> Dict[str, Any]:
    pdir = os.path.join(outdir, "paintings")
    os.makedirs(pdir, exist_ok=True)
    project, project_path = load_entry("hole_still_life")
    w = int(project["canvas"]["width"])
    h = int(project["canvas"]["height"])
    hero = load_sample(project["selection_sample"])
    sel = compose_spec(hero, feather=float(project["feather"])).coverage

    canvas = Canvas(h, w)
    layer_ev = []
    for ldef in project["layers"]:
        layer = canvas.new_layer(ldef["name"])
        for stroke in ldef["strokes"]:
            pts = _wave_points(stroke["params"]) \
                if stroke.get("generator") == "wave" else stroke["points"]
            paint_brush(canvas, layer, pts,
                        radius=float(stroke["radius"]),
                        color_rgb=tuple(stroke["color"]),
                        selection=sel,
                        spacing=0.25,
                        hardness=float(stroke.get("hardness", 0.6)),
                        opacity=float(stroke.get("opacity", 1.0)))
        export.save_rgba(
            layer.rgba,
            os.path.join(pdir, f"{project['id']}_{ldef['name']}.png"))
        layer_ev.append({"name": ldef["name"]})

    comp = canvas.composite()
    rgba_path = os.path.join(pdir, f"{project['id']}_composite_rgba.png")
    checker_path = os.path.join(pdir, f"{project['id']}_checker.png")
    export.save_rgba(comp, rgba_path)
    export.save_on_checker(comp, checker_path)

    outside = comp[..., 3][sel == 0.0]
    outside_max = float(outside.max()) if outside.size else 0.0
    hole_roi = project["verify"]["hole_roi"]
    hole = roi_stats(comp[..., 3], hole_roi)
    semi = int(np.count_nonzero((comp[..., 3] > 0.02)
                                & (comp[..., 3] < 0.98)))
    semi_min = int(project["verify"]["semi_transparent_pixels_min"])
    outside_ok = outside_max <= project["verify"]["outside_alpha_tol"]
    hole_ok = hole["max"] <= 1e-6
    semi_ok = semi >= semi_min
    ev = {"project_json": _rel(project_path),
          "composite_rgba": _rel(rgba_path),
          "checker_png": _rel(checker_path),
          "layers": layer_ev,
          "outside_selection_alpha_max": outside_max,
          "outside_alpha_pass": outside_ok,
          "hole_alpha": hole, "hole_alpha_pass": hole_ok,
          "semi_transparent_pixels": semi,
          "semi_transparent_min": semi_min,
          "semi_transparent_pass": semi_ok}
    ev["pass"] = _ok([outside_ok, hole_ok, semi_ok])
    return ev


# --------------------------------------------------------------------------
# 6) 四种覆盖率组合语义
# --------------------------------------------------------------------------

def semantics_board(outdir: str) -> Dict[str, Any]:
    sdir = os.path.join(outdir, "semantics")
    os.makedirs(sdir, exist_ok=True)
    w = h = 120
    a = CoverageMask(rasterize_paths(
        rectangle_path(24, 28, 56, 60).flattened(), w, h))
    b = CoverageMask(rasterize_paths(
        rectangle_path(52, 44, 48, 52).flattened(), w, h))
    ev = {"A_mass": float(a.coverage.sum()), "B_mass": float(b.coverage.sum()),
          "ops": {}}
    all_ok = True
    for op in CombineOp:
        c = combine(a, b, op)
        p = os.path.join(sdir, f"combine_{op.value}.png")
        export.save_mask_u8_png(c.coverage, p)
        ev["ops"][op.value] = {"mass": float(c.coverage.sum()),
                               "png": _rel(p)}
    # 恒等式核验：union+subtract 关系与 intersect 非负等
    u = combine(a, b, CombineOp.UNION).coverage
    i = combine(a, b, CombineOp.INTERSECT).coverage
    x = combine(a, b, CombineOp.XOR).coverage
    identity_err = float(np.abs(u + i - (a.coverage + b.coverage)).max())
    xor_err = float(np.abs(x - (u - i)).max())
    ok = identity_err < 1e-6 and xor_err < 1e-6
    all_ok &= ok
    ev["union_plus_intersect_eq_A_plus_B_maxerr"] = identity_err
    ev["xor_eq_union_minus_intersect_maxerr"] = xor_err
    ev["pass"] = ok
    return ev


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------

def build_all(outdir: str = DEFAULT_OUT) -> Dict[str, Any]:
    os.makedirs(outdir, exist_ok=True)
    sample_ids = [e["id"] for e in catalog_entries("selection")]
    samples = {}
    for sid in sample_ids:
        samples[sid] = verify_selection_sample(sid, outdir)

    report: Dict[str, Any] = {
        "schema": "samplebook-evidence/v1",
        "sample_dir": _rel(sample_dir()),
        "samples": samples,
        "resolution_lab": resolution_lab(outdir),
        "plates": plates_lab(outdir),
        "legacy": legacy_lab(outdir),
        "painting": painting_lab(outdir),
        "semantics": semantics_board(outdir),
    }
    sample_pass = _ok([s["pass"] for s in samples.values()])
    report["all_pass"] = _ok([
        sample_pass, report["resolution_lab"]["pass"],
        report["plates"]["pass"], report["legacy"]["pass"],
        report["painting"]["pass"], report["semantics"]["pass"],
    ])
    with open(os.path.join(outdir, "evidence.json"), "w",
              encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(os.path.join(outdir, "evidence.md"), "w",
              encoding="utf-8") as f:
        f.write(render_markdown(report))
    return report


def render_markdown(r: Dict[str, Any]) -> str:
    L: List[str] = ["# 选区样册执行证据", ""]
    L.append(f"总结论：**{'全部通过' if r['all_pass'] else '存在失败'}**")
    L.append("")
    L.append("所有数值均由 `examples/build_samplebook.py` 调用内核真实计算，"
             "边缘颜色与油墨读数来自写盘后重新读回的 PNG。")
    L.append("")
    L.append("## 样例覆盖量与边缘颜色")
    L.append("")
    L.append("| 样例 | Σc 实测 | 参考 | 容差 | 结论 |")
    L.append("|---|---:|---:|---:|:--:|")
    for sid, s in r["samples"].items():
        m = s.get("hard_mass", {})
        L.append("| {id} | {meas:.2f} | {exp:.2f} | ±{tol} | {p} |".format(
            id=sid, meas=m.get("measured", float("nan")),
            exp=m.get("expected", float("nan")),
            tol=m.get("tol_abs", "-"),
            p="PASS" if s["pass"] else "FAIL"))
    L.append("")
    for sid, s in r["samples"].items():
        for ec in s.get("edge_pixels", []):
            L.append(
                f"- `{sid}` 像素 {ec['pixel']}：hard c={ec['measured_hard']:.4f}"
                f"（约定 {ec['expected_hard']}），{os.path.basename(ec['edge_png'])} "
                f"读回 RGBA={ec['actual_rgba']}，约定 {ec['expected_rgba']}，"
                f"色差≤2：{'PASS' if ec['color_pass'] else 'FAIL'}")
    L.append("")
    rl = r["resolution_lab"]
    L.append("## 分辨率实验")
    L.append("")
    L.append(f"- {rl['preview_factor']}x 预览细丝峰值 "
             f"{rl['preview_filament']['max']:.4f} < τ={rl['visibility_tau']}"
             f" → 刚好消失：**{rl['preview_disappears']}**")
    L.append(f"- {rl['output_scale']}x 矢量重新光栅化细丝峰值 "
             f"{rl['output_filament']['max']:.3f} ≥ "
             f"{rl['output_filament_peak_min']}"
             f" → 回归：**{rl['output_peak_recovers']}**")
    L.append(f"- 质量比 {rl['mass_ratio']:.3f} ≈ {rl['mass_ratio_expected']}"
             f"（容差 2%）：**{rl['mass_ratio_pass']}**")
    L.append(f"- 放大版空孔 ROI max={rl['output_hole']['max']:.2e}"
             f" → 未补满：**{rl['output_hole_empty']}**")
    L.append("")
    pl = r["plates"]
    rb, rb4 = pl["readback_from_actual_file"], pl["readback_4x_from_actual_file"]
    L.append("## 双色套版油墨回读")
    L.append("")
    L.append(f"- 母版印版（{pl['plate_a_png']} 读回）：细丝峰值 "
             f"{rb['filament']['max']:.3f}，空孔 max={rb['hole']['max']:.2e}"
             f" → **{'PASS' if rb['pass'] else 'FAIL'}**")
    L.append(f"- 换 {rl['output_scale']}x 版后（{pl['plate_4x_png']} 读回）："
             f"细丝峰值 {rb4['filament']['max']:.3f}，"
             f"空孔 max={rb4['hole']['max']:.2e}"
             f" → **{'PASS' if rb4['pass'] else 'FAIL'}**")
    bc = pl["bad_control_report"]
    L.append(f"- 坏版对照 {pl['bad_control_png']}：空孔 max="
             f"{bc['hole']['max']:.3f}，检查器判定补满并拒绝："
             f"**{pl['bad_control_detected']}**")
    L.append("")
    L.append("## 旧页不可恢复标记")
    L.append("")
    for e in r["legacy"]["entries"]:
        L.append(f"- `{e['id']}` 真实素材 {e['source_png']} "
                 f"(sha256:{e['source_png_sha256_16']}, 形状 {e['mask_shape']})")
        L.append(f"  - 佚失 ROI {e['unrecoverable_rois_lowres']} 内旧蒙版油墨 "
                 f"max={e['roi_ink_from_actual_png']['max']:.2e} → 无墨："
                 f"**{e['roi_empty_in_legacy']}**")
        if e.get("companion_vector_preview_max") is not None:
            L.append(f"  - 伴随矢量母形同 ROI 预览覆盖 "
                     f"{e['companion_vector_preview_max']:.3f}（>0.01），"
                     f"交叉证明确有特征永久丢失："
                     f"**{e['companion_proves_lost_feature']}**")
        L.append(f"  - 不可恢复标记图：{e['marked_png']}")
    L.append("")
    pa = r["painting"]
    L.append("## 半透明绘画工程")
    L.append("")
    L.append(f"- 选区外 alpha 最大值 {pa['outside_selection_alpha_max']:.2e}"
             f"（必须为 0）：**{pa['outside_alpha_pass']}**")
    L.append(f"- 空孔 alpha max={pa['hole_alpha']['max']:.2e}："
             f"**{pa['hole_alpha_pass']}**")
    L.append(f"- 半透明像素数 {pa['semi_transparent_pixels']} ≥ "
             f"{pa['semi_transparent_min']}：**{pa['semi_transparent_pass']}**")
    L.append("")
    L.append("## 组合语义导出")
    L.append("")
    for name, o in r["semantics"]["ops"].items():
        L.append(f"- `{name}` Σc={o['mass']:.2f} → {o['png']}")
    L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="重建选区样册并核验/导出")
    ap.add_argument("--outdir", default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    report = build_all(args.outdir)
    for sid, s in report["samples"].items():
        print(f"[sample] {sid:20s} {'PASS' if s['pass'] else 'FAIL'}")
    print("[resolution]", "PASS" if report["resolution_lab"]["pass"] else "FAIL")
    print("[plates]    ", "PASS" if report["plates"]["pass"] else "FAIL")
    print("[legacy]    ", "PASS" if report["legacy"]["pass"] else "FAIL")
    print("[painting]  ", "PASS" if report["painting"]["pass"] else "FAIL")
    print("[semantics] ", "PASS" if report["semantics"]["pass"] else "FAIL")
    print("evidence:", os.path.join(args.outdir, "evidence.json"))
    if not report["all_pass"]:
        print("样册核验存在失败项", file=sys.stderr)
        return 1
    print("all sample-book checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
