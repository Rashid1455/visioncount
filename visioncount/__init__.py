"""VisionCount: real-time object detection, tracking and line-crossing counting."""
from .counter import LineCounter
from .pipeline import VisionCounter

__all__ = ["LineCounter", "VisionCounter"]
__version__ = "1.0.0"
