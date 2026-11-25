# 精细绘画选区内核（Subpixel Selection Kernel）

面向绘画软件的精细选区实现：曲线轮廓生成**亚像素面积覆盖率掩码**，
在连续覆盖率（而非布尔真假值）上执行合并、相减、相交与可控羽化，
笔刷实际写入受结果限制，并支持跨图层剪切、瓦片一致性与全分辨率导出。

## 启动界面

本环境无系统级 `python3-tk`，默认使用**零外部依赖的浏览器窗口**
（仅用 Python 标准库 `http.server`，内核仍在服务端按高分辨率画布计算）：

```bash
python3 gui/browser_app.py            # 默认 http://127.0.0.1:8765/
python3 gui/browser_app.py --port 9000
```

浏览器打开提示的地址即可：左侧最上方的 **「0. 精细选区样册资源（入库素材）」**
可直接加载 `selection_samples/` 里的真实样册；载入后仍可在画布上拖选区域、
调节羽化，预览区实时显示亚像素选区与红色编码的软边。其下分标签查看选区
结果、亚像素边缘、**样册 / 套版 / 旧页**、棋盘底绘画与两种剪切所有权对比；
所有按钮都请求 `browser_app.py` 后端的同一套 `selection_kernel` 实现，
不存在固定覆盖率数组或冒充亚像素边缘的假数据。内核错误以红色反馈条返回控件。

样册相关按钮：

- **载入选材到绘画窗口**：selection 类从 JSON 原路径经内核重建为可编辑
  矢量 path；project 类（`hole_still_life`）载入其伴随选区与工程羽化；
  legacy 类（`old_page_hole7`）没有矢量母形，不伪造 shapes，而是跳到
  「样册 / 套版 / 旧页」标签标记不可恢复区域。
- **冲印双色套版 / 旧页标记**：在「样册 / 套版 / 旧页」标签写真实印版 PNG
  到 `output/samplebook/gui_last/`，再从磁盘读回油墨覆盖并给出 1×/4× 核验。

桌面 Tk 版（需先 `apt install python3-tk`）：

```bash
python3 gui/tk_app.py
```

命令行仅作为辅助的重建 / 取证入口（图形窗口仍是首选入口）：

```bash
# 从 selection_samples/ 原路径重建全部样册、套版、旧页与证据到 output/samplebook/
python3 examples/build_samplebook.py
# 重新生成入库素材（JSON 原路径 + 模拟旧页 PNG，含 provenance）
python3 examples/create_sample_assets.py
# 旧的端到端绘画/剪切工作流
python3 examples/demo_pipeline.py --width 220 --height 200 --feather 4.0
```

## 运行测试

```bash
python3 -m unittest discover -s tests -v
```

共 **52 个测试**：原 35 个覆盖圆/斜边/极细缝/接近轮廓的截面、对称性、
选区外像素保持、组合语义、旋转-羽化次序、瓦片一致性与导出往返；
新增 17 个样册测试覆盖入库素材真实性、Path JSON 编解码、四个样例的
面积/ROI/边缘颜色约定、分辨率实验室、印版写盘回读与坏版对照、旧页
标记、`build_all` 全流程、GUI HTTP 端到端（含一次性几何数值防伪）
以及旧 `demo_pipeline.py` 子进程回归。

## 选区样册与双色套版工程

`selection_samples/` 是**真实入库、可直接加载**的绘画工程素材，而不是
抠好的成品图。每个 JSON 保存矢量原路径命令（M/L/C/Z）、组合算子、
羽化参数、参考约定与 provenance；`index.json` 是目录，GUI 下拉与
CLI/测试都从这里发现素材。

| ID | 覆盖的要求 | 锁定的独立参考约定（节录） |
| --- | --- | --- |
| `hole_filament` | 孔洞 + 孔边亚像素细丝 | hard 质量 ≈ `150×110 − π·42² + 84×0.32 = 19685.11`（容差 1.5）；细丝走廊 ROI 峰值 ≥0.159、sum `23.04±0.2`；空孔 ROI max ≤1e-6 |
| `subpixel_lines` | 五条不同相位/宽度的亚像素细线 | 手工面积合计 `326.75±0.5`（实测精确）；边缘像素 `[50,70]=0.5`、`[50,39]=0.125` 等，边缘 PNG 红色通道 = `round(c·255)` |
| `repeat_spokes` | 同一原路径重复旋转变换 12 份 | JSON 只存 1 个基础矩形 + `repeat` 配方（中心、30° 步进），现场展开；质量 `12×6×54=3888±3`；空毂 ROI max ≤1e-6 |
| `intersect_narrow` | 相交轮廓 + 0.8px 窄羽化 | 交集面积 `27.95×92=2571.4±0.2`；四个边缘探针颜色逐一约定；被消去轮廓边的 ghost ROI 羽化 max ≤0.02（实测 0） |
| `hole_still_life` | 有洞、半透明内容的可编辑绘画工程 | 两层参数化波浪笔画，opacity 0.42 / 0.9 真实走 `paint_brush`；孔 ROI 与选区外 alpha max=0；半透明像素 ≥200（实测 5729） |
| `old_page_hole7` | 只剩低清蒙版的旧页 | 真实 60×50、仅 0/255 的 PNG；佚失细丝 ROI 油墨 max=0；provenance 如实写明是“模拟旧导出链”（4× box 降采样 + 阈值化），矢量母形已佚 |

### 预览消失 → 4× 矢量恢复 → 旧页不可恢复

同一条孔边细丝有完整证据链（数值来自最近一次 `build_samplebook.py`，
完整机器可读证据见 `output/samplebook/evidence.json` / `evidence.md`）：

1. **预览分辨率刚好消失**：hard 覆盖经 4× 面积平均降采样后，细丝走廊
   峰值 `0.04 < τ=0.05`，即在预览中不可见但尚未被二值删除。
2. **有矢量母形时放大输出**：`compose_spec(..., scale=4)` 把 JSON 原路径
   坐标整体 ×4 后**重新解析光栅化**（不是放大位图），细丝峰值恢复到
   ≈0.64，总覆盖质量按 `s²=16` 缩放（实测比值 16.000，容差 2%）。
3. **只留低清蒙版的旧页**：`old_page_hole7_mask.png` 中该 ROI 油墨 max=0；
   `mark_unrecoverable()` 只依据记录在案的 ROI 画红斜线与边框，**不重建、
   不猜测** ROI 内形状——这部分信息在降采样 + 阈值化时已永久丢失。

### 双色套版与“换版后从实际油墨读回”

`plates.py` 把覆盖率直接写成 8-bit 灰度印版 PNG（像素值即油墨覆盖），
随后 `load_plate_ink()` **从磁盘文件重新读回**再核验：

- A 版：`hole_filament` 朱红；B 版：独立圆环配方（r60 减 r54）靛蓝；
  另写菲林阳片、减法混色套印校样、带标题/色标的规线贴版与 4× 版本。
- `verify_filament_and_hole()` 检查读回油墨里细丝存在且空孔没被补满：
  母版细丝峰值 ≈0.161，4× 版 ≈0.639，两版空孔 max 均为 0。
- **坏版对照证明检查有效**：`fill_enclosed_holes()` 生成二值化后封闭孔
  被填满的对照印版，检查器明确给出 `hole_empty=False / pass=False`；
  测试断言坏版必被拒绝，避免“恒真检查”和伪报。

### 边缘颜色与覆盖语义核验

`visualize.edge_color_rgba()` 与 GUI 共用同一编码：仅对 `0<c<1` 像素写
`RGBA(round(c·255), 0, 0, 255)`，所以导出的 `<id>_edge_*.png` 可逐像素
反查约定颜色，而不是“看起来有红边”。相交样例还使用 edge-aware 边界筛选
（`samplebook.relevant_segments`）：对每条输入边分段、在两侧采样组合
hard 覆盖，只保留真实覆盖跳变的边段进入距离场，避免相交后无关轮廓边
在窄羽化下产生鬼影。

`output/samplebook/semantics/` 另导出 union/subtract/intersect/xor 四图，
证据中核对质量数值与组合恒等式误差 `<1e-6`，演示不同覆盖语义。

### 证据与可复现

```
selection_samples/
  index.json                     目录 + gui_entry 说明
  outlines/*.json                4 个选区原路径/参数/参考约定
  projects/hole_still_life.json  半透明绘画工程（笔画参数，可重建）
  legacy/old_page_hole7.json     旧页 provenance + 不可恢复 ROI
  legacy/old_page_hole7_mask.png 真实低清二值蒙版
output/samplebook/
  evidence.json / evidence.md    实际测量值、文件 sha256、PASS/FAIL
  <id>_hard_u16.png / _feather_u16.png / _edge_*.png
  plates/  paintings/  legacy/  semantics/
```

`python3 examples/build_samplebook.py` 任一项约定不符都会以非零码退出；
测试在临时目录调用同一个 `build_all()` 并断言 `all_pass is True`。
素材目录可用环境变量 `SELECTION_SAMPLE_DIR` 覆盖（仅测试/隔离用途）。

## 核心语义（覆盖率不是布尔值）

- **解析光栅化**（`rasterize.py`）：逐像素条带对多边形做梯形面积积分，
  输出每像素真实面积覆盖率 c∈[0,1]；非零环绕支持带洞轮廓。不做
  超采样，也不对抗锯齿选区反复阈值化。
- **组合运算**（`mask.py`，全程连续值）：
  - 合并 `c = 1-(1-a)(1-b)`；相减 `c = a(1-b)`；
  - 相交 `c = ab`；异或 `c = a+b-2ab`。
- **羽化**（`distance.py`）：带符号距离场上 `c(d)=0.5-d/(2r)`，
  半径 r 用**画布坐标**定义，界面缩放不改变实际宽度；r=0 时返回
  解析硬边覆盖本身（保留 0<c<1 细边），不是二值蒙版。
- **操作次序**（`selection.py`）：选区保留矢量轮廓、坐标域与操作日志。
  “先旋转再羽化”旋转矢量后重新解析光栅化；“先羽化再旋转”是对已
  羽化图像做图像域变换，两者边缘不同，测试分别核查。
- **笔刷限制**（`painting.py`）：整条笔画先在覆盖率域取并集，再
  `写入 = 笔画覆盖 × 选区覆盖`，选区外 alpha 严格为 0。
- **跨图层剪切所有权**：覆盖率相同的软边在叠加后归属可能不同——
  `merge_first`（先合成内容再乘 c）与 `cut_each`（每层先乘 c 再合成）
  在非线性 alpha 合成下结果不同，`ownership_difference` 直接给出差值；
  原层留洞由 `alpha×(1-c)` 裁决，剪下图块放棋盘底由实际透明度呈现。
- **瓦片一致性**（`tiling.py`）：距离场从矢量轮廓按瓦片原点重算，
  组合运算逐像素独立；`assert_tile_consistency` 断言瓦片拼接与整幅
  参考逐元素相同（测试中最大差为 0）。
- **导出**（`export.py`）：掩码默认 16-bit PNG（量化误差 1/65535），
  按画布原始 W×H 导出，显示分辨率不参与，预览与导出不会出现不同锯齿。

## 目录

```
selection_kernel/  内核：geometry/rasterize/distance/mask/selection/
                   painting/tiling/export，以及样册层
                   pathio（矢量 JSON 编解码）、samplebook（加载/展开/合成/
                   ROI/旧页标记）、plates（套版油墨写读与核验）、
                   visualize（GUI/CLI 共用的 PNG 与边缘颜色编码）
selection_samples/ 入库样册素材：原路径 JSON、羽化参数、provenance、旧页 PNG
gui/               browser_app.py + index.html（默认图形入口）、tk_app.py（可选）
examples/          create_sample_assets.py（造素材）、build_samplebook.py
                   （重建+证据+套版）、demo_pipeline.py（旧工作流）
tests/             52 个单元测试（含 test_samplebook.py 的 HTTP 端到端）
output/            演示导出物；samplebook/ 子目录为样册证据、套版、绘画与标记图
```
