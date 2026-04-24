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

## Generating ranges for a specific `parent_id`

A common use-case is taking a collection of `Event`s, filtering them to a
single `parent_id`, and producing the `Range`s for that parent.  There are two
ways to do this depending on whether you are using the direct Python API or an
Apache Beam pipeline.

### Direct Python API

Group your events by `parent_id`, then call `segmenter.process()` for each
group.  The segmenter processes **one parent at a time**, so you must supply
only the events (and any prior ranges) that belong to the target parent.

```python
from collections import defaultdict
from time_range_segmenter import Event, Range, TimeRangeSegmenter

def my_label_fn(events: list[Event]) -> list[Range]:
    """Convert a single parent's events into ranges — your business logic goes here."""
    ...

segmenter = TimeRangeSegmenter(label_fn=my_label_fn)

# ---- 1. Gather events and (optionally) prior ranges ----
all_events: list[Event] = [...]       # events from your data source
prior_ranges: list[Range] = [...]     # ranges from the previous run (if any)

# ---- 2. Group by parent_id ----
events_by_parent: dict[int, list[Event]] = defaultdict(list)
for event in all_events:
    events_by_parent[event.parent_id].append(event)

ranges_by_parent: dict[int, list[Range]] = defaultdict(list)
for r in prior_ranges:
    ranges_by_parent[r.parent_id].append(r)

# ---- 3. Process a single parent_id ----
target_parent_id = 42

output_ranges, retired_ranges = segmenter.process(
    parent_id=target_parent_id,
    events=events_by_parent[target_parent_id],
    input_ranges=ranges_by_parent.get(target_parent_id, []),
)
# output_ranges  — new Range objects for parent 42 with sticky range_ids
# retired_ranges — prior Range objects whose range_ids were not preserved

# ---- 4. Or process every parent_id at once ----
all_parent_ids = set(events_by_parent) | set(ranges_by_parent)
for pid in all_parent_ids:
    output, retired = segmenter.process(
        parent_id=pid,
        events=events_by_parent[pid],
        input_ranges=ranges_by_parent.get(pid, []),
    )
    # ... handle output and retired ranges for each parent ...
```

> **Key points**
>
> * `segmenter.process()` accepts the events and ranges for **exactly one**
>   `parent_id` per call.  The caller is responsible for grouping data
>   beforehand.
> * `input_ranges` can be omitted (or set to `None` / `[]`) on the first run
>   when there are no prior ranges.
> * The `parent_id` field on every returned `Range` is always set to the
>   `parent_id` you pass in, regardless of what the `label_fn` returns.

### Apache Beam pipeline

When using Apache Beam, the `build_pipeline` helper handles the grouping
automatically.  It keys both `Event`s and `Range`s by their `parent_id`,
performs a `CoGroupByKey`, and then applies the segmenter to each group.  You
do **not** need to group or filter manually.

```python
import apache_beam as beam
from time_range_segmenter import Event, Range, TimeRangeSegmenter
from time_range_segmenter.pipeline import build_pipeline

segmenter = TimeRangeSegmenter(label_fn=my_label_fn)

with beam.Pipeline() as p:
    # parse_event and parse_range are user-supplied functions that convert
    # JSON strings into Event and Range objects respectively.
    events = p | "ReadEvents" >> beam.io.ReadFromText("events.jsonl") | beam.Map(parse_event)
    ranges = p | "ReadRanges" >> beam.io.ReadFromText("ranges.jsonl") | beam.Map(parse_range)

    # build_pipeline groups events and ranges by parent_id internally,
    # so the output PCollections contain Range objects for ALL parent_ids.
    output_ranges, retired_ranges = build_pipeline(
        pipeline=p,
        events_pcollection=events,
        ranges_pcollection=ranges,
        segmenter=segmenter,
    )

    # To inspect or write ranges for a specific parent_id, filter the output:
    parent_42_ranges = output_ranges | beam.Filter(lambda r: r.parent_id == 42)
```

> **Key points**
>
> * `build_pipeline` performs a `CoGroupByKey` on `parent_id`, so each
>   invocation of the segmenter automatically receives only the events and
>   ranges that share the same `parent_id`.
> * The returned `PCollection`s contain `Range` objects for **all**
>   `parent_id`s.  Use `beam.Filter` if you need only a subset.
> * All callables (`label_fn`, `split_fn`, etc.) must be **picklable**
>   (module-level functions, not lambdas or closures) when used in a Beam
>   pipeline.

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
