# time-range-segmenter

A Python scaffold for labelling time segments of streaming data with **persistent
(sticky) identifiers**.

## Overview

Events arrive in a stream grouped by `parent_id`.  A user-supplied *label
function* converts each group of events into a set of non-overlapping time
intervals (ranges).  When the pipeline is re-run with new events, the scaffold
attempts to carry over the `range_id` from the previous run's ranges so that
downstream consumers see as few id changes as possible.

```
Events  ─┐
          ├─► CoGroupByKey(parent_id) ─► TimeRangeSegmenter ─► output_ranges
Ranges  ─┘                                                  └─► retired_ranges
```

## Data models

```python
@dataclass
class Event:
    parent_id: int
    ts: datetime
    data: dict | None = None

@dataclass
class Range:
    parent_id: int
    start_time: datetime
    end_time: datetime
    range_id: str | None = None   # None = "needs a new id"
```

## ID-assignment algorithm

After the label function produces output ranges, the scaffold builds a bipartite
*overlap graph* between the old (input) ranges and the new (output) ranges.  Each
connected component of this graph falls into one of four cases:

| Case | Input ranges | Output ranges | Behaviour |
|------|-------------|---------------|-----------|
| **1-to-1** | 1 | 1 | `range_id` is always preserved. |
| **split** | 1 | N > 1 | `split_fn` picks which output range inherits the id. |
| **merge** | N > 1 | 1 | `merge_fn` picks which input range donates its id. |
| **complex** | M > 1 | N > 1 | `complex_fn` may assign ids arbitrarily. |

Output ranges that cannot be matched receive `range_id = None` so an external
system can assign a new one.  Input ranges whose ids were not carried over are
emitted as **retired ranges**.

## Quick start

```python
from time_range_segmenter import Event, Range, TimeRangeSegmenter

def my_label_fn(events: list[Event]) -> list[Range]:
    # Your business logic here
    ...
    return ranges

segmenter = TimeRangeSegmenter(label_fn=my_label_fn)

output_ranges, retired_ranges = segmenter.process(
    parent_id=42,
    events=current_events,
    input_ranges=previous_ranges,
)
```

### Custom hooks

```python
def my_split_fn(input_range: Range, output_ranges: list[Range]) -> Range:
    """Pick the longest output range to inherit the id."""
    return max(output_ranges, key=lambda r: r.end_time - r.start_time)

def my_merge_fn(input_ranges: list[Range], output_range: Range) -> Range:
    """Use the id from the input range that overlaps the most with the output."""
    return max(
        input_ranges,
        key=lambda r: min(r.end_time, output_range.end_time)
                    - max(r.start_time, output_range.start_time),
    )

segmenter = TimeRangeSegmenter(
    label_fn=my_label_fn,
    split_fn=my_split_fn,
    merge_fn=my_merge_fn,
)
```

## Apache Beam pipeline

```python
import apache_beam as beam
from time_range_segmenter.pipeline import build_pipeline

with beam.Pipeline() as p:
    events = p | beam.io.ReadFromText("events.jsonl") | beam.Map(parse_event)
    ranges = p | beam.io.ReadFromText("ranges.jsonl") | beam.Map(parse_range)

    output_ranges, retired_ranges = build_pipeline(
        pipeline=p,
        events_pcollection=events,
        ranges_pcollection=ranges,
        segmenter=segmenter,
    )

    output_ranges  | beam.io.WriteToText("output_ranges.jsonl")
    retired_ranges | beam.io.WriteToText("retired_ranges.jsonl")
```

> **Note**: all hook functions passed to `TimeRangeSegmenter` must be
> **picklable** (module-level functions, not lambdas or closures) when used
> inside a Beam pipeline.

## Installation

```bash
pip install apache-beam
pip install -e .
```

## Running tests

```bash
pip install pytest
pytest
```
