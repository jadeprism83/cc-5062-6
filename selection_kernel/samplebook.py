"""选区样册：资源加载、重复变换展开、统一合成与旧页不可恢复标记。

GUI、命令行重建脚本与测试都走这里的同一套真实实现：
- 样册 JSON 保存原路径命令、组合算子、羽化半径与来源参数；
- 重复变换（repeat）由原路径 + 旋转矩阵现场展开，不存固定位图；
- compose_paths 是界面渲染与脚本导出的唯一入口，禁止用固定数据冒充。

相交/异或等掩码域组合之后，部分输入轮廓边不再是组合结果的边界；
直接对“所有输入边”求距离场会在被消去的轮廓段旁产生假羽化光晕。
compose_paths(edge_aware=True) 用组合后的硬覆盖率逐段核验边的两侧
是否真的发生覆盖跳变，只保留真实结果边界段参与距离场。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .distance import feather_from_distance, polygon_signed_distance
from .geometry import (
    Path, circle_path, rectangle_path, rotation_matrix, slanted_path,
    thin_slit_path,
)
from .mask import CombineOp, CoverageMask
from .pathio import path_from_dict, path_to_dict, scale_path_dict
from .selection import FLATTEN_TOL, Selection, SelectionDocument

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_DIR = os.path.join(REPO_ROOT, "selection_samples")

PRIMITIVE_BUILDERS = {
    "circle": lambda x, y, w, h: circle_path(x + w / 2, y + h / 2,
                                             min(w, h) / 2),
    "rectangle": lambda x, y, w, h: rectangle_path(x, y, w, h),
    "slanted": lambda x, y, w, h: slanted_path(x, y, w, h, shear=0.25 * w),
    "slit": lambda x, y, w, h: thin_slit_path(
        x + w / 2, y + h / 2, max(0.2, min(0.6, 0.04 * w)), h),
}


def sample_dir() -> str:
    return os.environ.get("SELECTION_SAMPLE_DIR", SAMPLE_DIR)


def load_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def catalog() -> Dict[str, Any]:
    return load_json_file(os.path.join(sample_dir(), "index.json"))


def catalog_entries(kind: Optional[str] = None) -> List[Dict[str, Any]]:
    entries = catalog()["samples"]
    if kind is not None:
        entries = [e for e in entries if e.get("kind", "selection") == kind]
    return entries


def _entry_path(entry: Dict[str, Any]) -> str:
    return os.path.join(sample_dir(), entry["file"])


def load_entry(entry_id: str) -> Tuple[Dict[str, Any], str]:
    for e in catalog()["samples"]:
        if e["id"] == entry_id:
            p = _entry_path(e)
            return load_json_file(p), p
    raise KeyError(f"no sample-book entry with id {entry_id!r}")


def load_sample(sample_id: str) -> Dict[str, Any]:
    spec, _ = load_entry(sample_id)
    return spec


def expand_repeats(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把带 repeat 配方的素材项展开为若干变换后的路径项。

    repeat: {"count": N, "center": [x, y],
             "angle_start_deg": a0, "angle_step_deg": da}
    第 0 份使用 angle_start，之后依次叠加 step；全部由原路径矩阵变换得到。
    """
    rep = item.get("repeat")
    if not rep:
        return [item]
    count = int(rep["count"])
    cx, cy = rep["center"]
    a0 = float(rep.get("angle_start_deg", 0.0))
    da = float(rep.get("angle_step_deg", 0.0))
    base = path_from_dict(item["path"])
    out: List[Dict[str, Any]] = []
    for k in range(count):
        angle = np.radians(a0 + k * da)
        moved = base.transformed(rotation_matrix(float(angle), (cx, cy)))
        new_item = dict(item)
        new_item.pop("repeat", None)
        new_item["path"] = path_to_dict(moved)
        new_item["repeat_index"] = k
        out.append(new_item)
    return out


def expanded_items(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for item in spec["items"]:
        items.extend(expand_repeats(item))
    return items


def shape_to_path(shape: Dict[str, Any]) -> Path:
    """GUI 拖选基元或样册 path 项 -> Path。"""
    kind = shape.get("shape", "path")
    if kind == "path":
        return path_from_dict(shape["path"])
    x, y = float(shape["x"]), float(shape["y"])
    w, h = float(shape["w"]), float(shape["h"])
    if w <= 0 or h <= 0:
        raise ValueError("shape must have positive width and height")
    return PRIMITIVE_BUILDERS[kind](x, y, w, h)


def shapes_to_items(shapes: Sequence[Dict[str, Any]]) -> List[Tuple[Path, str]]:
    out = []
    for sp in shapes:
        out.append((shape_to_path(sp), sp.get("op", CombineOp.UNION.value)))
    return out


def _bilinear_sample(arr: np.ndarray, x: float, y: float) -> float:
    h, w = arr.shape
    if x < 0.0 or y < 0.0 or x > w - 1 or y > h - 1:
        return 0.0
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    top = arr[y0, x0] * (1 - fx) + arr[y0, x1] * fx
    bot = arr[y1, x0] * (1 - fx) + arr[y1, x1] * fx
    return float(top * (1 - fy) + bot * fy)


def relevant_segments(polys: Sequence[np.ndarray], hard: np.ndarray,
                      step: float = 1.2, delta: float = 0.7,
                      jump: float = 0.3) -> List[np.ndarray]:
    """筛出组合硬覆盖场上真正构成边界的边段。

    每条边按 ≤step 像素切段，在段中点沿法向 ±delta 采样组合覆盖率；
    两侧覆盖跳变 >jump 才认为该段是组合结果边界。被布尔组合消去的
    轮廓段（如相交后落在对方外部的边）两侧都为 0，自然被排除。
    """
    segs: List[np.ndarray] = []
    for poly in polys:
        n = len(poly)
        if n < 2:
            continue
        for i in range(n):
            p = np.asarray(poly[i], dtype=np.float64)
            q = np.asarray(poly[(i + 1) % n], dtype=np.float64)
            edge = q - p
            length = float(np.hypot(edge[0], edge[1]))
            if length < 1e-9:
                continue
            pieces = max(1, int(np.ceil(length / step)))
            nx, ny = -edge[1] / length, edge[0] / length
            for k in range(pieces):
                t0, t1 = k / pieces, (k + 1) / pieces
                mid = p + edge * ((t0 + t1) * 0.5)
                cp = _bilinear_sample(hard, mid[0] + nx * delta,
                                      mid[1] + ny * delta)
                cm = _bilinear_sample(hard, mid[0] - nx * delta,
                                      mid[1] - ny * delta)
                if abs(cp - cm) > jump:
                    segs.append(np.array([p + edge * t0, p + edge * t1]))
    return segs


@dataclass
class ComposeResult:
    hard: CoverageMask
    coverage: np.ndarray          # 羽化后的最终覆盖率（feather=0 时等于 hard）
    polys: List[np.ndarray]
    boundary_segments: List[np.ndarray]
    feather: float


def compose_paths(height: int, width: int,
                  path_items: Sequence[Tuple[Path, str]],
                  feather: float = 0.0,
                  edge_aware: bool = True) -> ComposeResult:
    """样册/界面共用的真实合成入口：逐 Selection 组合，再做画布坐标羽化。"""
    if feather < 0 or feather > min(height, width):
        raise ValueError(f"feather radius out of range: {feather}")
    doc = SelectionDocument((height, width))
    polys: List[np.ndarray] = []
    for idx, (path, op) in enumerate(path_items):
        doc.add(Selection(f"s{idx}", (height, width), [path]))
        polys.extend(path.flattened(FLATTEN_TOL))
    if not path_items:
        zero = CoverageMask(np.zeros((height, width), np.float32))
        return ComposeResult(zero, zero.coverage, [], [], feather)

    ops = [CombineOp(op) for _, op in path_items[1:]]
    hard = doc.combined(ops)
    if feather <= 0.0:
        return ComposeResult(hard, hard.coverage, polys, [], feather)

    if edge_aware:
        segments = relevant_segments(polys, hard.coverage)
        sdf_polys: Sequence[np.ndarray] = segments if segments else polys
    else:
        segments = []
        sdf_polys = polys
    sdf = polygon_signed_distance(sdf_polys, width, height,
                                  hard_cover=hard.coverage)
    final = feather_from_distance(sdf, feather, hard_cover=hard.coverage)
    return ComposeResult(hard, final, polys, segments, feather)


def compose_shapes(payload_shapes: Sequence[Dict[str, Any]],
                   height: int, width: int, feather: float,
                   edge_aware: bool = True) -> ComposeResult:
    return compose_paths(height, width, shapes_to_items(payload_shapes),
                         feather, edge_aware)


def compose_recipe(recipe_items: Sequence[Dict[str, Any]],
                   height: int, width: int, scale: float = 1.0,
                   edge_aware: bool = False) -> ComposeResult:
    """合成样册 plates 等附属配方里的 items（默认硬边、无羽化）。"""
    path_items = []
    for it in recipe_items:
        pd = it["path"]
        if scale != 1.0:
            pd = scale_path_dict(pd, scale)
        path_items.append((path_from_dict(pd), it.get("op", "union")))
    return compose_paths(height, width, path_items, 0.0, edge_aware)


def compose_spec(spec: Dict[str, Any], feather: Optional[float] = None,
                 scale: float = 1.0,
                 edge_aware: bool = True) -> ComposeResult:
    """按样册 JSON 重建；scale>1 时坐标域与路径一起放大，用于重新光栅化。"""
    cw = int(round(spec["canvas"]["width"] * scale))
    ch = int(round(spec["canvas"]["height"] * scale))
    f = float(spec.get("feather", 0.0)) if feather is None else float(feather)
    if scale != 1.0:
        f = f * scale
    items = expanded_items(spec)
    path_items = []
    for item in items:
        pd = item["path"]
        if scale != 1.0:
            pd = scale_path_dict(pd, scale)
        path_items.append((path_from_dict(pd), item.get("op", "union")))
    return compose_paths(ch, cw, path_items, f, edge_aware)


def box_downsample(coverage: np.ndarray, factor: int) -> np.ndarray:
    """面积平均降采样（预览分辨率）；画布尺寸必须能被 factor 整除。"""
    if factor <= 1:
        return coverage
    h, w = coverage.shape
    if h % factor or w % factor:
        raise ValueError("box_downsample requires divisible dimensions")
    return coverage.reshape(h // factor, factor, w // factor, factor).mean(
        axis=(1, 3))


def roi_slice(roi_xyxy: Sequence[int], scale: float = 1.0
              ) -> Tuple[slice, slice]:
    x0, y0, x1, y1 = [int(round(v * scale)) for v in roi_xyxy]
    return slice(y0, y1), slice(x0, x1)


def roi_stats(coverage: np.ndarray, roi_xyxy: Sequence[int],
              scale: float = 1.0) -> Dict[str, float]:
    sub = coverage[roi_slice(roi_xyxy, scale)]
    if sub.size == 0:
        raise ValueError(f"empty ROI {roi_xyxy} for shape {coverage.shape}")
    return {"max": float(sub.max()), "mean": float(sub.mean()),
            "sum": float(sub.sum()), "pixels": int(sub.size)}


# --------------------------------------------------------------------------
# 旧页：只有低清蒙版、矢量母形已佚
# --------------------------------------------------------------------------

def legacy_mask_path(record: Dict[str, Any], record_path: str) -> str:
    return os.path.join(os.path.dirname(record_path), record["mask_png"])


def load_legacy_mask(record: Dict[str, Any], record_path: str) -> np.ndarray:
    """从真实存在的旧页 PNG 读回覆盖率（不接受内存中的替代品）。"""
    p = legacy_mask_path(record, record_path)
    img = Image.open(p).convert("L")
    arr = np.asarray(img, dtype=np.uint8)
    return arr.astype(np.float32) / 255.0


def mark_unrecoverable(mask: np.ndarray, rois: Sequence[Sequence[int]],
                       upscale: int = 4) -> np.ndarray:
    """生成旧页标记图：低清蒙版最近邻放大 + 红色斜线标出不可恢复区域。

    标记只依据旧页记录中保存的佚失特征 ROI（来源元数据），不重建、
    不猜测 ROI 内的亚像素形状——那部分信息在低清化 + 阈值化时已永久丢失。
    """
    h, w = mask.shape
    u8 = np.round(np.clip(mask, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(u8, "L").resize((w * upscale, h * upscale),
                                          Image.NEAREST)
    rgb = np.stack([np.asarray(img)] * 3, axis=-1).astype(np.float32) / 255.0
    rgb = rgb * 0.85 + 0.15  # 稍微提灰，让红斜线清楚
    yy, xx = np.mgrid[0:h * upscale, 0:w * upscale]
    hatch = ((xx // 6 + yy // 6) % 2 == 0) & (((xx + yy) % 12) < 6)
    mark = np.zeros(rgb.shape[:2], dtype=bool)
    for roi in rois:
        ys, xs = roi_slice(roi, upscale)
        mark[ys, xs] = True
    red = np.array([0.85, 0.12, 0.12])
    rgb[mark & hatch] = red
    rgb[mark & ~hatch] = rgb[mark & ~hatch] * 0.55 + red * 0.45
    # ROI 边框
    for roi in rois:
        x0, y0, x1, y1 = [int(round(v * upscale)) for v in roi]
        rgb[y0:y1, x0:x0 + 2] = red
        rgb[y0:y1, max(x0, x1 - 2):x1] = red
        rgb[y0:y0 + 2, x0:x1] = red
        rgb[max(y0, y1 - 2):y1, x0:x1] = red
    return np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
