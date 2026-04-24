"""Apache Beam pipeline integration for time-range segmentation.

This module provides a :class:`SegmenterDoFn` that can be dropped into any Beam
pipeline, and a :func:`build_pipeline` convenience function that wires together
the co-group-by-key and DoFn into a reusable sub-graph.

Pipeline sketch
---------------
::

    events  ──┐
               ├─► CoGroupByKey(parent_id) ──► SegmenterDoFn ──► output_ranges
    ranges  ──┘                                               └──► retired_ranges

Output tags
-----------
* **main output** (``"output_ranges"``) – :class:`~time_range_segmenter.models.Range`
  objects produced by the label function with sticky ``range_id`` values where
  possible.
* **side output** (``RETIRED_TAG = "retired_ranges"``) – input
  :class:`~time_range_segmenter.models.Range` objects whose ``range_id`` was not
  carried over to any output range.  These should be forwarded to a system that
  can mark the identifier as no longer active.
"""

from typing import Iterator

import apache_beam as beam
from apache_beam import pvalue

from .models import Event, Range
from .segmenter import TimeRangeSegmenter

RETIRED_TAG = "retired_ranges"
"""Side-output tag for retired input ranges."""


class SegmenterDoFn(beam.DoFn):
    """Beam ``DoFn`` that processes one ``(parent_id, {events, ranges})`` element.

    Main output
        :class:`~time_range_segmenter.models.Range` objects with sticky
        ``range_id`` values (or ``None`` when no match was found).

    Tagged output ``RETIRED_TAG``
        Input :class:`~time_range_segmenter.models.Range` objects whose
        ``range_id`` was not preserved.

    Parameters
    ----------
    segmenter:
        A configured :class:`~time_range_segmenter.segmenter.TimeRangeSegmenter`
        instance.  All callables stored on the segmenter must be picklable.
    """

    def __init__(self, segmenter: TimeRangeSegmenter) -> None:
        self._segmenter = segmenter

    def process(self, element) -> Iterator:  # type: ignore[override]
        parent_id, grouped = element
        events = list(grouped.get("events", []))
        input_ranges = list(grouped.get("ranges", []))

        output_ranges, retired_ranges = self._segmenter.process(
            parent_id=parent_id,
            events=events,
            input_ranges=input_ranges,
        )

        for r in output_ranges:
            yield r

        for r in retired_ranges:
            yield pvalue.TaggedOutput(RETIRED_TAG, r)


def build_pipeline(
    pipeline: beam.Pipeline,
    events_pcollection: beam.PCollection,
    ranges_pcollection: beam.PCollection,
    segmenter: TimeRangeSegmenter,
):
    """Wire segmentation transforms into *pipeline* and return output PCollections.

    Parameters
    ----------
    pipeline:
        The :class:`beam.Pipeline` instance (used only for label namespacing).
    events_pcollection:
        ``PCollection`` of :class:`~time_range_segmenter.models.Event` objects.
    ranges_pcollection:
        ``PCollection`` of :class:`~time_range_segmenter.models.Range` objects
        from a prior run (may be empty).
    segmenter:
        A configured :class:`~time_range_segmenter.segmenter.TimeRangeSegmenter`.

    Returns
    -------
    output_ranges : beam.PCollection
        New :class:`~time_range_segmenter.models.Range` objects with sticky ids.
    retired_ranges : beam.PCollection
        Input :class:`~time_range_segmenter.models.Range` objects whose ids were
        not preserved.
    """
    events_by_parent = events_pcollection | "KeyEvents" >> beam.Map(
        lambda e: (e.parent_id, e)
    )
    ranges_by_parent = ranges_pcollection | "KeyRanges" >> beam.Map(
        lambda r: (r.parent_id, r)
    )

    grouped = (
        {"events": events_by_parent, "ranges": ranges_by_parent}
        | "CoGroupByParent" >> beam.CoGroupByKey()
    )

    result = grouped | "ProcessGroups" >> beam.ParDo(
        SegmenterDoFn(segmenter)
    ).with_outputs(RETIRED_TAG, main="output_ranges")

    return result.output_ranges, result[RETIRED_TAG]
