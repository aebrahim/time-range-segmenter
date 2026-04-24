"""Data models for the time-range segmenter."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class Event:
    """An event that belongs to a parent entity, occurring at a given timestamp.

    Attributes:
        parent_id: Identifier for the parent entity that owns this event.
        ts: Timestamp at which the event occurred.
        data: Optional dict of additional attributes carried by the event.
    """

    parent_id: int
    ts: datetime
    data: Optional[Dict[str, Any]] = field(default=None)


@dataclass
class Range:
    """A half-open time interval ``[start_time, end_time)`` assigned to a parent entity.

    Attributes:
        parent_id: Identifier for the parent entity that owns this range.
        start_time: Inclusive start of the interval.
        end_time: Exclusive end of the interval.
        range_id: Optional persistent identifier.  ``None`` means the range has not
            yet been assigned an identifier by an external system.
    """

    parent_id: int
    start_time: datetime
    end_time: datetime
    range_id: Optional[str] = None
