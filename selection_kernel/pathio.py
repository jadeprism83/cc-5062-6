"""矢量轮廓的 JSON 编解码：样册资源以原路径（命令序列）入库。

JSON 形式：
    {"commands": [["M", [x, y]], ["L", [x, y]],
                  ["C", [c1x, c1y], [c2x, c2y], [x, y]], ["Z"]]}
一条 Path 含多个子路径时，用多个 "M" 分段。坐标全部是画布坐标浮点，
因此样册在任意输出倍率下都能从矢量母形重新光栅化，而不是放大旧蒙版。
"""
from __future__ import annotations

from typing import Any, Dict, List

from .geometry import Path, Point, SubPath

_CMD_TO_STR = {0: "M", 1: "L", 2: "C", 3: "Z"}
_STR_TO_CMD = {"M": 0, "L": 1, "C": 2, "Z": 3}


def _pt_to_json(p: Point) -> List[float]:
    return [float(p[0]), float(p[1])]


def path_to_dict(path: Path) -> Dict[str, Any]:
    out: List[list] = []
    for sp in path.subpaths:
        for cmd, pts in sp.segments:
            name = _CMD_TO_STR[cmd]
            if name == "Z":
                out.append(["Z"])
            elif name == "C":
                out.append(["C", _pt_to_json(pts[0]), _pt_to_json(pts[1]),
                            _pt_to_json(pts[2])])
            else:
                out.append([name, _pt_to_json(pts[0])])
    return {"commands": out}


def path_from_dict(data: Dict[str, Any]) -> Path:
    path = Path()
    sp: SubPath | None = None
    for entry in data["commands"]:
        name = entry[0]
        cmd = _STR_TO_CMD[name]
        if name == "M":
            sp = SubPath()
            path.subpaths.append(sp)
            sp.move_to((float(entry[1][0]), float(entry[1][1])))
        elif sp is None:
            raise ValueError("path command before M")
        elif name == "L":
            sp.line_to((float(entry[1][0]), float(entry[1][1])))
        elif name == "C":
            sp.curve_to((float(entry[1][0]), float(entry[1][1])),
                        (float(entry[2][0]), float(entry[2][1])),
                        (float(entry[3][0]), float(entry[3][1])))
        elif name == "Z":
            sp.close()
    return path


def scale_path_dict(data: Dict[str, Any], factor: float) -> Dict[str, Any]:
    """把路径中的所有坐标乘以 factor（生成放大输出母形用）。"""
    def sc(p):
        return [p[0] * factor, p[1] * factor]

    commands = []
    for entry in data["commands"]:
        name = entry[0]
        if name == "Z":
            commands.append(["Z"])
        elif name == "C":
            commands.append(["C", sc(entry[1]), sc(entry[2]), sc(entry[3])])
        else:
            commands.append([name, sc(entry[1])])
    return {"commands": commands}
