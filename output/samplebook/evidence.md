# 选区样册执行证据

总结论：**全部通过**

所有数值均由 `examples/build_samplebook.py` 调用内核真实计算，边缘颜色与油墨读数来自写盘后重新读回的 PNG。

## 样例覆盖量与边缘颜色

| 样例 | Σc 实测 | 参考 | 容差 | 结论 |
|---|---:|---:|---:|:--:|
| hole_filament | 19684.11 | 19685.11 | ±1.5 | PASS |
| subpixel_lines | 326.75 | 326.75 | ±0.5 | PASS |
| repeat_spokes | 3888.00 | 3888.00 | ±3.0 | PASS |
| intersect_narrow | 2571.40 | 2571.40 | ±0.2 | PASS |

- `subpixel_lines` 像素 [50, 39]：hard c=0.1250（约定 0.125），subpixel_lines_edge_hard.png 读回 RGBA=[32, 0, 0, 255]，约定 [32, 0, 0, 255]，色差≤2：PASS
- `subpixel_lines` 像素 [50, 70]：hard c=0.5000（约定 0.5），subpixel_lines_edge_hard.png 读回 RGBA=[128, 0, 0, 255]，约定 [128, 0, 0, 255]，色差≤2：PASS
- `subpixel_lines` 像素 [50, 99]：hard c=0.3750（约定 0.375），subpixel_lines_edge_hard.png 读回 RGBA=[96, 0, 0, 255]，约定 [96, 0, 0, 255]，色差≤2：PASS
- `subpixel_lines` 像素 [50, 150]：hard c=0.4000（约定 0.4），subpixel_lines_edge_hard.png 读回 RGBA=[102, 0, 0, 255]，约定 [102, 0, 0, 255]，色差≤2：PASS
- `subpixel_lines` 像素 [59, 170]：hard c=0.1750（约定 0.175），subpixel_lines_edge_hard.png 读回 RGBA=[45, 0, 0, 255]，约定 [45, 0, 0, 255]，色差≤2：PASS
- `intersect_narrow` 像素 [92, 100]：hard c=0.6000（约定 0.6），intersect_narrow_edge_hard.png 读回 RGBA=[153, 0, 0, 255]，约定 [153, 0, 0, 255]，色差≤2：PASS
- `intersect_narrow` 像素 [120, 100]：hard c=0.3500（约定 0.35），intersect_narrow_edge_hard.png 读回 RGBA=[89, 0, 0, 255]，约定 [89, 0, 0, 255]，色差≤2：PASS
- `intersect_narrow` 像素 [105, 58]：hard c=0.7500（约定 0.75），intersect_narrow_edge_hard.png 读回 RGBA=[191, 0, 0, 255]，约定 [191, 0, 0, 255]，色差≤2：PASS
- `intersect_narrow` 像素 [105, 150]：hard c=0.2500（约定 0.25），intersect_narrow_edge_hard.png 读回 RGBA=[64, 0, 0, 255]，约定 [64, 0, 0, 255]，色差≤2：PASS

## 分辨率实验

- 4x 预览细丝峰值 0.0400 < τ=0.05 → 刚好消失：**True**
- 4x 矢量重新光栅化细丝峰值 0.640 ≥ 0.5 → 回归：**True**
- 质量比 16.000 ≈ 16（容差 2%）：**True**
- 放大版空孔 ROI max=0.00e+00 → 未补满：**True**

## 双色套版油墨回读

- 母版印版（output/samplebook/plates/plate_A_vermilion_ink.png 读回）：细丝峰值 0.161，空孔 max=0.00e+00 → **PASS**
- 换 4x 版后（output/samplebook/plates/plate_A_vermilion_ink_4x.png 读回）：细丝峰值 0.639，空孔 max=0.00e+00 → **PASS**
- 坏版对照 output/samplebook/plates/control_bad_filled_plate_A.png：空孔 max=1.000，检查器判定补满并拒绝：**True**

## 旧页不可恢复标记

- `old_page_hole7` 真实素材 selection_samples/legacy/old_page_hole7_mask.png (sha256:b4c60166892cb09d, 形状 [50, 60])
  - 佚失 ROI [[21, 24, 39, 26]] 内旧蒙版油墨 max=0.00e+00 → 无墨：**True**
  - 伴随矢量母形同 ROI 预览覆盖 0.040（>0.01），交叉证明确有特征永久丢失：**True**
  - 不可恢复标记图：output/samplebook/legacy/old_page_hole7_marked.png

## 半透明绘画工程

- 选区外 alpha 最大值 0.00e+00（必须为 0）：**True**
- 空孔 alpha max=0.00e+00：**True**
- 半透明像素数 5729 ≥ 200：**True**

## 组合语义导出

- `union` Σc=4624.00 → output/samplebook/semantics/combine_union.png
- `subtract` Σc=2128.00 → output/samplebook/semantics/combine_subtract.png
- `intersect` Σc=1232.00 → output/samplebook/semantics/combine_intersect.png
- `xor` Σc=3392.00 → output/samplebook/semantics/combine_xor.png
