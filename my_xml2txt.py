# python
#
#  食用方法
# python3 my_xml2txt.py -h
#
# # 最小示例
# python3 my_xml2txt.py --xml-dir ./annotations --out-dir ./labels
#
# # 使用类别映射与可见性
# python3 my_xml2txt.py --xml-dir ./annotations --out-dir ./labels \
#   --names ./names.txt --default-class 0 --vis 2




import os
import sys
import glob
import argparse
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional

def load_class_map(names_path: Optional[str]) -> Dict[str, int]:
    if not names_path:
        return {}
    with open(names_path, "r", encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    return {name: i for i, name in enumerate(names)}

def clamp01(v: float) -> float:
    return 0.0 if v < 0 else 1.0 if v > 1 else v

def norm_xy(x: float, y: float, W: int, H: int) -> Tuple[float, float]:
    return clamp01(x / W), clamp01(y / H)

def parse_quad(obj_el: ET.Element) -> Optional[List[Tuple[float, float]]]:
    bb = obj_el.find("bndbox")
    if bb is None:
        return None
    keys = ["x1","y1","x2","y2","x3","y3","x4","y4"]
    if not all(bb.find(k) is not None for k in keys):
        return None
    pts = []
    for i in range(1, 5):
        x = float(bb.findtext(f"x{i}"))
        y = float(bb.findtext(f"y{i}"))
        pts.append((x, y))
    return pts  # 保持 x1..x4 顺序

def aabb_from_pts(pts: List[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)

def yolo_bbox_from_xyxy(xmin: float, ymin: float, xmax: float, ymax: float, W: int, H: int) -> Optional[Tuple[float, float, float, float]]:
    w = xmax - xmin
    h = ymax - ymin
    if w <= 0 or h <= 0:
        return None
    xc = (xmin + xmax) / 2.0
    yc = (ymin + ymax) / 2.0
    nx, ny = norm_xy(xc, yc, W, H)
    nw = clamp01(w / W)
    nh = clamp01(h / H)
    return nx, ny, nw, nh

def convert_xml_to_yolo_pose(xml_path: str, out_dir: str, class_map: Dict[str, int], default_cid: int, vis_flag: int = 2):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_el = root.find("size")
    if size_el is None:
        print(f"[WARN] no <size> in {xml_path}, skip", file=sys.stderr)
        return
    W = int(float(size_el.findtext("width", "0")))
    H = int(float(size_el.findtext("height", "0")))
    if W <= 0 or H <= 0:
        print(f"[WARN] invalid size in {xml_path}, skip", file=sys.stderr)
        return

    objects = root.findall("object")
    if not objects:
        return

    # 文件名基于 xml 名称（与 <filename> 一致通常更安全）
    base = os.path.splitext(os.path.basename(xml_path))[0]
    out_path = os.path.join(out_dir, base + ".txt")
    os.makedirs(out_dir, exist_ok=True)

    lines: List[str] = []
    for obj in objects:
        name = (obj.findtext("name") or "obj").strip()
        cid = class_map.get(name, default_cid)

        pts = parse_quad(obj)
        if not pts:
            # 若没有四点，跳过该目标
            continue

        # bbox
        xmin, ymin, xmax, ymax = aabb_from_pts(pts)
        bbox = yolo_bbox_from_xyxy(xmin, ymin, xmax, ymax, W, H)
        if bbox is None:
            continue
        xc, yc, bw, bh = bbox

        # 关键点（保持 x1..x4 顺序），归一化并附加可见性
        kv: List[float] = []
        for (x, y) in pts:
            nx, ny = norm_xy(x, y, W, H)
            kv += [f"{nx:.6f}", f"{ny:.6f}", str(vis_flag)]

        line = f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} " + " ".join(kv)
        lines.append(line)

    if lines:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

def main():
    ap = argparse.ArgumentParser(description="Convert XML (x1..x4/y1..y4) to YOLO Pose labels (4 keypoints).")
    ap.add_argument("--xml-dir", required=True, help="输入 XML 目录")
    ap.add_argument("--out-dir", required=True, help="输出 labels 目录")
    ap.add_argument("--names", default=None, help="可选类别文件，每行一个类名（行号为 id）")
    ap.add_argument("--default-class", type=int, default=0, help="未命中 names 时的默认类 id，默认 0")
    ap.add_argument("--vis", type=int, default=2, choices=[0,1,2], help="关键点可见性标志，默认 2")
    args = ap.parse_args()

    class_map = load_class_map(args.names)

    xml_files = sorted(glob.glob(os.path.join(args.xml_dir, "*.xml")))
    if not xml_files:
        print(f"[ERROR] no xml under {args.xml_dir}", file=sys.stderr)
        sys.exit(1)

    for xp in xml_files:
        convert_xml_to_yolo_pose(xp, args.out_dir, class_map, args.default_class, vis_flag=args.vis)

    print("[DONE]")

if __name__ == "__main__":
    main()
