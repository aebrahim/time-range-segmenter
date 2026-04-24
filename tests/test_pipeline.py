"""Integration tests for the Apache Beam pipeline."""

from datetime import datetime, timedelta

import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to

from time_range_segmenter.models import Event, Range
from time_range_segmenter.pipeline import RETIRED_TAG, build_pipeline
from time_range_segmenter.segmenter import TimeRangeSegmenter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def dt(hour: int) -> datetime:
    return datetime(2024, 1, 1, hour, 0, 0)


def make_event(parent_id: int, hour: int) -> Event:
    return Event(parent_id=parent_id, ts=dt(hour))


def make_range(
    parent_id: int, start_hour: int, end_hour: int, range_id: str | None = None
) -> Range:
    return Range(
        parent_id=parent_id,
        start_time=dt(start_hour),
        end_time=dt(end_hour),
        range_id=range_id,
    )


# ---------------------------------------------------------------------------
# Module-level label functions (must be picklable for Beam serialisation)
# ---------------------------------------------------------------------------


def single_range_label(events):
    """Create one range covering all events, padded to at least one hour."""
    if not events:
        return []
    min_ts = min(e.ts for e in events)
    max_ts = max(e.ts for e in events)
    if min_ts >= max_ts:
        max_ts = min_ts + timedelta(hours=1)
    return [Range(parent_id=0, start_time=min_ts, end_time=max_ts)]


def two_fixed_ranges_label(events):
    """Always return two fixed ranges regardless of events."""
    return [
        Range(parent_id=0, start_time=dt(0), end_time=dt(5)),
        Range(parent_id=0, start_time=dt(5), end_time=dt(10)),
    ]


def empty_label(events):
    return []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBeamPipelineBasic:
    def test_single_parent_one_to_one(self):
        """Sticky id is preserved through the Beam pipeline (1-to-1 case)."""
        segmenter = TimeRangeSegmenter(label_fn=single_range_label)

        events = [make_event(1, 1), make_event(1, 3)]
        input_ranges = [make_range(1, 1, 3, range_id="sticky")]

        with TestPipeline() as p:
            events_pc = p | "CreateEvents" >> beam.Create(events)
            ranges_pc = p | "CreateRanges" >> beam.Create(input_ranges)

            output_ranges, retired_ranges = build_pipeline(
                pipeline=p,
                events_pcollection=events_pc,
                ranges_pcollection=ranges_pc,
                segmenter=segmenter,
            )

            assert_that(
                output_ranges | "GetIds" >> beam.Map(lambda r: r.range_id),
                equal_to(["sticky"]),
                label="CheckOutputId",
            )
            assert_that(
                retired_ranges | "GetRetiredIds" >> beam.Map(lambda r: r.range_id),
                equal_to([]),
                label="CheckRetiredEmpty",
            )

    def test_multiple_parents(self):
        """Events from different parents are processed independently."""
        segmenter = TimeRangeSegmenter(label_fn=single_range_label)

        events = [make_event(1, 1), make_event(2, 3), make_event(3, 5)]
        input_ranges = []

        with TestPipeline() as p:
            events_pc = p | "CreateEvents" >> beam.Create(events)
            ranges_pc = p | "CreateRanges" >> beam.Create(input_ranges)

            output_ranges, _ = build_pipeline(
                pipeline=p,
                events_pcollection=events_pc,
                ranges_pcollection=ranges_pc,
                segmenter=segmenter,
            )

            parent_ids = output_ranges | beam.Map(lambda r: r.parent_id)
            assert_that(parent_ids, equal_to([1, 2, 3]))

    def test_no_events_produces_no_outputs(self):
        """A parent that appears only in ranges (no events) produces no output."""
        segmenter = TimeRangeSegmenter(label_fn=empty_label)

        events = []
        input_ranges = [make_range(1, 0, 5, range_id="old")]

        with TestPipeline() as p:
            events_pc = p | "CreateEvents" >> beam.Create(events)
            ranges_pc = p | "CreateRanges" >> beam.Create(input_ranges)

            output_ranges, retired_ranges = build_pipeline(
                pipeline=p,
                events_pcollection=events_pc,
                ranges_pcollection=ranges_pc,
                segmenter=segmenter,
            )

            assert_that(
                output_ranges | beam.Map(lambda r: r.range_id),
                equal_to([]),
                label="CheckOutputEmpty",
            )
            assert_that(
                retired_ranges | beam.Map(lambda r: r.range_id),
                equal_to(["old"]),
                label="CheckRetiredHasOld",
            )

    def test_retired_ranges_emitted(self):
        """Input ranges that do not overlap any output are retired."""
        segmenter = TimeRangeSegmenter(label_fn=single_range_label)

        events = [make_event(1, 12)]  # event at hour 12
        # input range is at hours 0–5, far from the output range [12, 13)
        input_ranges = [make_range(1, 0, 5, range_id="old-id")]

        with TestPipeline() as p:
            events_pc = p | "CreateEvents" >> beam.Create(events)
            ranges_pc = p | "CreateRanges" >> beam.Create(input_ranges)

            output_ranges, retired_ranges = build_pipeline(
                pipeline=p,
                events_pcollection=events_pc,
                ranges_pcollection=ranges_pc,
                segmenter=segmenter,
            )

            assert_that(
                output_ranges | "OutIds" >> beam.Map(lambda r: r.range_id),
                equal_to([None]),
                label="CheckOutputNone",
            )
            assert_that(
                retired_ranges | "RetiredIds" >> beam.Map(lambda r: r.range_id),
                equal_to(["old-id"]),
                label="CheckRetiredOldId",
            )

    def test_empty_pipeline(self):
        """Pipeline with no events and no ranges produces empty outputs."""
        segmenter = TimeRangeSegmenter(label_fn=empty_label)

        with TestPipeline() as p:
            events_pc = p | "CreateEvents" >> beam.Create([])
            ranges_pc = p | "CreateRanges" >> beam.Create([])

            output_ranges, retired_ranges = build_pipeline(
                pipeline=p,
                events_pcollection=events_pc,
                ranges_pcollection=ranges_pc,
                segmenter=segmenter,
            )

            assert_that(output_ranges, equal_to([]), label="CheckOutputEmpty")
            assert_that(retired_ranges, equal_to([]), label="CheckRetiredEmpty")


class TestBeamPipelineSplitMerge:
    def test_split_case_one_id_preserved(self):
        """Split case: one input range → two output ranges; only one keeps the id."""
        segmenter = TimeRangeSegmenter(label_fn=two_fixed_ranges_label)

        events = [make_event(1, 2)]
        input_ranges = [make_range(1, 0, 10, range_id="parent-id")]

        with TestPipeline() as p:
            events_pc = p | "CreateEvents" >> beam.Create(events)
            ranges_pc = p | "CreateRanges" >> beam.Create(input_ranges)

            output_ranges, retired_ranges = build_pipeline(
                pipeline=p,
                events_pcollection=events_pc,
                ranges_pcollection=ranges_pc,
                segmenter=segmenter,
            )

            output_ids = output_ranges | beam.Map(lambda r: r.range_id)
            # Default split: first output gets the id
            assert_that(
                output_ids,
                equal_to(["parent-id", None]),
                label="CheckSplitIds",
            )
            assert_that(
                retired_ranges | beam.Map(lambda r: r.range_id),
                equal_to([]),
                label="CheckNoRetired",
            )

    def test_merge_case_one_id_preserved_one_retired(self):
        """Merge case: two inputs → one output; one id preserved, one retired."""

        def merge_label(events):
            return [Range(parent_id=0, start_time=dt(0), end_time=dt(10))]

        segmenter = TimeRangeSegmenter(label_fn=merge_label)

        events = [make_event(1, 2)]
        input_ranges = [
            make_range(1, 0, 5, range_id="first"),
            make_range(1, 5, 10, range_id="second"),
        ]

        with TestPipeline() as p:
            events_pc = p | "CreateEvents" >> beam.Create(events)
            ranges_pc = p | "CreateRanges" >> beam.Create(input_ranges)

            output_ranges, retired_ranges = build_pipeline(
                pipeline=p,
                events_pcollection=events_pc,
                ranges_pcollection=ranges_pc,
                segmenter=segmenter,
            )

            assert_that(
                output_ranges | beam.Map(lambda r: r.range_id),
                equal_to(["first"]),
                label="CheckMergeOutput",
            )
            assert_that(
                retired_ranges | beam.Map(lambda r: r.range_id),
                equal_to(["second"]),
                label="CheckMergeRetired",
            )
