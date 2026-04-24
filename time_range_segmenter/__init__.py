"""Time Range Segmenter – persistent-identifier scaffold for time-range pipelines."""

from .models import Event, Range
from .segmenter import TimeRangeSegmenter

__all__ = ["Event", "Range", "TimeRangeSegmenter"]
