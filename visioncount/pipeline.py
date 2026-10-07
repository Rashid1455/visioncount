"""YOLO detection + ByteTrack tracking + OpenCV drawing."""
from __future__ import annotations

import time
from collections import Counter
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from .counter import LineCounter

GREEN, RED, WHITE, DARK = (60, 200, 80), (60, 60, 230), (255, 255, 255), (30, 30, 30)


def _colour(track_id: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    return tuple(int(c) for c in rng.integers(80, 255, 3))


class VisionCounter:
    def __init__(
        self,
        model: str = "yolov8n.pt",
        conf: float = 0.35,
        classes: Optional[Sequence[str]] = None,
        line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
        tracker: str = "bytetrack.yaml",
    ):
        self.model = YOLO(model)
        self.names = self.model.names
        self.conf = conf
        self.class_ids = None
        if classes:
            lookup = {v.lower(): k for k, v in self.names.items()}
            missing = [c for c in classes if c.lower() not in lookup]
            if missing:
                raise ValueError(f"Unknown classes {missing}. Options: {sorted(lookup)}")
            self.class_ids = [lookup[c.lower()] for c in classes]
        self.line_pts = line
        self.tracker = tracker
        self.counter: Optional[LineCounter] = None
        self.trails: dict[int, list] = {}
        self.fps = 0.0

    # ---------- single image (detection only) ----------
    def detect_image(self, image: np.ndarray):
        res = self.model.predict(image, conf=self.conf, classes=self.class_ids, verbose=False)[0]
        counts = Counter(self.names[int(c)] for c in res.boxes.cls.tolist())
        out = image.copy()
        for box, cls, cf in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            self._draw_box(out, box, f"{self.names[int(cls)]} {cf:.2f}", _colour(int(cls) + 7))
        self._draw_panel(out, [f"{k}: {v}" for k, v in counts.most_common()] or ["no objects"])
        return out, dict(counts)

    # ---------- video / stream (tracking + counting) ----------
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if self.counter is None:
            a, b = self.line_pts or ((0, h // 2), (w, h // 2))
            self.counter = LineCounter(a, b)
        t0 = time.perf_counter()
        res = self.model.track(
            frame, conf=self.conf, classes=self.class_ids, persist=True,
            tracker=self.tracker, verbose=False,
        )[0]
        out = frame.copy()
        boxes = res.boxes
        if boxes.id is not None:
            for box, tid, cls in zip(boxes.xyxy.tolist(), boxes.id.int().tolist(), boxes.cls.int().tolist()):
                label = self.names[cls]
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                hit = self.counter.update(tid, (cx, cy), label)
                col = _colour(tid)
                trail = self.trails.setdefault(tid, [])
                trail.append((int(cx), int(cy)))
                del trail[:-30]
                for p, q in zip(trail, trail[1:]):
                    cv2.line(out, p, q, col, 2)
                self._draw_box(out, box, f"#{tid} {label}", GREEN if hit else col)
        dt = time.perf_counter() - t0
        self.fps = 0.9 * self.fps + 0.1 * (1 / dt) if self.fps else 1 / dt
        a, b = self.counter.start, self.counter.end
        cv2.line(out, tuple(map(int, a)), tuple(map(int, b)), RED, 3)
        self._draw_panel(out, [
            f"IN: {self.counter.in_count}   OUT: {self.counter.out_count}",
            f"TOTAL: {self.counter.total}",
            f"FPS: {self.fps:.1f}",
        ])
        return out

    def run(self, source, save_path: Optional[str] = None, show: bool = False,
            max_frames: Optional[int] = None) -> dict:
        cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open source: {source}")
        writer = None
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames and n >= max_frames):
                break
            out = self.process_frame(frame)
            if save_path:
                if writer is None:
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25
                    writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                             (out.shape[1], out.shape[0]))
                writer.write(out)
            if show:
                cv2.imshow("VisionCount  (q to quit)", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            n += 1
        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()
        return self.summary(frames=n)

    def summary(self, frames: int = 0) -> dict:
        c = self.counter
        return {
            "frames": frames,
            "in": c.in_count if c else 0,
            "out": c.out_count if c else 0,
            "total": c.total if c else 0,
            "per_class": {k: dict(v) for k, v in (c.per_class.items() if c else [])},
            "avg_fps": round(self.fps, 1),
        }

    # ---------- drawing helpers ----------
    @staticmethod
    def _draw_box(img, box, text, col):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, max(y1 - th - 8, 0)), (x1 + tw + 6, y1), col, -1)
        cv2.putText(img, text, (x1 + 3, max(y1 - 5, th)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, DARK, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_panel(img, lines: Iterable[str]):
        lines = list(lines)
        h = 28 * len(lines) + 12
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (290, 10 + h), DARK, -1)
        cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
        for i, t in enumerate(lines):
            cv2.putText(img, t, (20, 38 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
