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

浏览器打开提示的地址即可：左侧拖选轮廓、调羽化半径、选组合算子与
笔刷，右侧分标签查看选区结果、亚像素边缘、棋盘底绘画与两种剪切
所有权对比；内核错误以红色反馈条返回控件。

桌面 Tk 版（需先 `apt install python3-tk`）：

```bash
python3 gui/tk_app.py
```

命令行完整工作流（生成真实绘画与掩码导出物到 `output/`）：

```bash
python3 examples/demo_pipeline.py --width 220 --height 200 --feather 4.0
```

## 运行测试

```bash
python3 -m unittest discover -s tests -v
```

共 35 个测试，覆盖圆/斜边/极细缝/接近轮廓的截面、对称性、选区外
像素保持、组合语义、旋转-羽化次序、瓦片一致性与导出往返。

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
selection_kernel/  内核（geometry/rasterize/distance/mask/selection/painting/tiling/export）
gui/               browser_app.py + index.html（默认）、tk_app.py（可选桌面）
examples/          demo_pipeline.py 端到端工作流
tests/             35 个单元测试
output/            演示导出物（掩码、绘画、剪切、差异、瓦片校验）
```
