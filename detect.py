"""Command-line entry point.

Examples:
  python detect.py --source assets/sample.jpg                        # image: detect + count per class
  python detect.py --source traffic.mp4 --classes car truck --save outputs/out.mp4
  python detect.py --source 0 --show                                 # webcam, live window
  python detect.py --source video.mp4 --line 0,400,1280,400          # custom counting line
"""
import argparse
import json
from pathlib import Path

import cv2

from visioncount import VisionCounter

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_line(s: str):
    x1, y1, x2, y2 = map(int, s.split(","))
    return (x1, y1), (x2, y2)


def main():
    p = argparse.ArgumentParser(description="YOLO object detection, tracking and line counting")
    p.add_argument("--source", required=True, help="image, video path, URL or webcam index (0)")
    p.add_argument("--model", default="yolov8n.pt", help="any Ultralytics YOLO weights")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--classes", nargs="*", help="only these classes, e.g. person car")
    p.add_argument("--line", type=parse_line, help="counting line x1,y1,x2,y2 (default: horizontal middle)")
    p.add_argument("--save", help="output path for annotated image/video")
    p.add_argument("--show", action="store_true", help="open a live preview window")
    p.add_argument("--json", help="write summary JSON here")
    a = p.parse_args()

    vc = VisionCounter(a.model, a.conf, a.classes, a.line)
    if Path(a.source).suffix.lower() in IMAGE_EXT:
        img = cv2.imread(a.source)
        if img is None:
            raise FileNotFoundError(a.source)
        out, counts = vc.detect_image(img)
        save = a.save or str(Path("outputs") / f"{Path(a.source).stem}_detected.jpg")
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save, out)
        summary = {"counts": counts, "saved": save}
    else:
        if a.save:
            Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        summary = vc.run(a.source, save_path=a.save, show=a.show)
    print(json.dumps(summary, indent=2))
    if a.json:
        Path(a.json).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
