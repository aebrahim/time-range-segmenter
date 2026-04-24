"""Unit tests for the TimeRangeSegmenter core logic."""

from datetime import datetime, timedelta

import pytest

from time_range_segmenter.models import Event, Range
from time_range_segmenter.segmenter import TimeRangeSegmenter, ranges_overlap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def dt(hour: int, minute: int = 0) -> datetime:
    """Return a datetime on 2024-01-01 at the given hour and minute."""
    return datetime(2024, 1, 1, hour, minute, 0)


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
# ranges_overlap
# ---------------------------------------------------------------------------


class TestRangesOverlap:
    def test_fully_overlapping(self):
        r1 = make_range(1, 0, 10)
        r2 = make_range(1, 2, 8)
        assert ranges_overlap(r1, r2)
        assert ranges_overlap(r2, r1)

    def test_partially_overlapping(self):
        r1 = make_range(1, 0, 5)
        r2 = make_range(1, 3, 8)
        assert ranges_overlap(r1, r2)
        assert ranges_overlap(r2, r1)

    def test_adjacent_not_overlapping(self):
        """Ranges that merely touch at a boundary do not overlap."""
        r1 = make_range(1, 0, 5)
        r2 = make_range(1, 5, 10)
        assert not ranges_overlap(r1, r2)
        assert not ranges_overlap(r2, r1)

    def test_gap_not_overlapping(self):
        r1 = make_range(1, 0, 3)
        r2 = make_range(1, 5, 10)
        assert not ranges_overlap(r1, r2)

    def test_identical_ranges_overlap(self):
        r1 = make_range(1, 0, 5)
        r2 = make_range(1, 0, 5)
        assert ranges_overlap(r1, r2)


# ---------------------------------------------------------------------------
# 1-to-1 case
# ---------------------------------------------------------------------------


class TestOneToOne:
    """One input range overlaps with exactly one output range."""

    def test_range_id_preserved(self):
        def label_fn(events):
            return [make_range(0, 0, 5)]

        seg = TimeRangeSegmenter(label_fn=label_fn)
        outputs, retired = seg.process(
            parent_id=1,
            events=[make_event(1, 2)],
            input_ranges=[make_range(1, 0, 5, range_id="abc")],
        )

        assert len(outputs) == 1
        assert outputs[0].range_id == "abc"
        assert retired == []

    def test_partial_overlap_preserves_range_id(self):
        """The ranges partially overlap but it is still a 1-to-1 component."""

        def label_fn(events):
            return [make_range(0, 3, 8)]

        seg = TimeRangeSegmenter(label_fn=label_fn)
        outputs, retired = seg.process(
            parent_id=1,
            events=[make_event(1, 4)],
            input_ranges=[make_range(1, 0, 6, range_id="xyz")],
        )

        assert outputs[0].range_id == "xyz"
        assert retired == []

    def test_input_with_none_range_id_passes_none(self):
        """If the input range has no id, the output also gets None (no id to preserve)."""

        def label_fn(events):
            return [make_range(0, 0, 5)]

        seg = TimeRangeSegmenter(label_fn=label_fn)
        outputs, retired = seg.process(
            parent_id=1,
            events=[make_event(1, 2)],
            input_ranges=[make_range(1, 0, 5, range_id=None)],
        )

        assert outputs[0].range_id is None
        assert retired == []  # range_id=None inputs are never retired

    def test_parent_id_set_on_output(self):
        """The segmenter always overwrites parent_id on output ranges."""

        def label_fn(events):
            return [Range(parent_id=999, start_time=dt(0), end_time=dt(5))]

        seg = TimeRangeSegmenter(label_fn=label_fn)
        outputs, _ = seg.process(parent_id=42, events=[], input_ranges=[])

        assert outputs[0].parent_id == 42

    def test_multiple_independent_components(self):
        """Two separate 1-to-1 components each preserve their own id."""

        def label_fn(events):
            return [make_range(0, 0, 4), make_range(0, 6, 10)]

        seg = TimeRangeSegmenter(label_fn=label_fn)
        outputs, retired = seg.process(
            parent_id=1,
            events=[make_event(1, 1), make_event(1, 7)],
            input_ranges=[
                make_range(1, 0, 4, range_id="first"),
                make_range(1, 6, 10, range_id="second"),
            ],
        )

        ids = {r.range_id for r in outputs}
        assert ids == {"first", "second"}
        assert retired == []


# ---------------------------------------------------------------------------
# Split case
# ---------------------------------------------------------------------------


class TestSplitCase:
    """One input range overlaps with multiple output ranges."""

    def test_default_split_first_output_gets_id(self):
        def label_fn(events):
            return [make_range(0, 0, 3), make_range(0, 3, 6)]

        seg = TimeRangeSegmenter(label_fn=label_fn)
        outputs, retired = seg.process(
            parent_id=1,
            events=[make_event(1, 1), make_event(1, 4)],
            input_ranges=[make_range(1, 0, 6, range_id="parent-id")],
        )

        assert len(outputs) == 2
        first = min(outputs, key=lambda r: r.start_time)
        second = max(outputs, key=lambda r: r.start_time)
        assert first.range_id == "parent-id"
        assert second.range_id is None
        assert retired == []

    def test_custom_split_fn_last_output(self):
        def label_fn(events):
            return [make_range(0, 0, 3), make_range(0, 3, 6)]

        def last_output(input_range, output_ranges):
            return max(output_ranges, key=lambda r: r.start_time)

        seg = TimeRangeSegmenter(label_fn=label_fn, split_fn=last_output)
        outputs, retired = seg.process(
            parent_id=1,
            events=[make_event(1, 1)],
            input_ranges=[make_range(1, 0, 6, range_id="parent-id")],
        )

        last = max(outputs, key=lambda r: r.start_time)
        first = min(outputs, key=lambda r: r.start_time)
        assert last.range_id == "parent-id"
        assert first.range_id is None
        assert retired == []

    def test_split_three_ways(self):
        def label_fn(events):
            return [make_range(0, 0, 2), make_range(0, 2, 4), make_range(0, 4, 6)]

        seg = TimeRangeSegmenter(label_fn=label_fn)
        outputs, retired = seg.process(
            parent_id=1,
            events=[make_event(1, 1)],
            input_ranges=[make_range(1, 0, 6, range_id="big")],
        )

        id_counts = sum(1 for r in outputs if r.range_id == "big")
        assert id_counts == 1
        none_counts = sum(1 for r in outputs if r.range_id is None)
        assert none_counts == 2
        assert retired == []


# ---------------------------------------------------------------------------
# Merge case
# ---------------------------------------------------------------------------


class TestMergeCase:
    """Multiple input ranges all overlap with one output range."""

    def test_default_merge_first_input_donates_id(self):
        def label_fn(events):
            return [make_range(0, 0, 8)]

        seg = TimeRangeSegmenter(label_fn=label_fn)
        outputs, retired = seg.process(
            parent_id=1,
            events=[make_event(1, 2)],
            input_ranges=[
                make_range(1, 0, 4, range_id="first"),
                make_range(1, 4, 8, range_id="second"),
            ],
        )

        assert len(outputs) == 1
        assert outputs[0].range_id == "first"
        assert len(retired) == 1
        assert retired[0].range_id == "second"

    def test_custom_merge_fn_last_input(self):
        def label_fn(events):
            return [make_range(0, 0, 8)]

        def last_input(input_ranges, output_range):
            return max(input_ranges, key=lambda r: r.start_time)

        seg = TimeRangeSegmenter(label_fn=label_fn, merge_fn=last_input)
        outputs, retired = seg.process(
            parent_id=1,
            events=[make_event(1, 2)],
            input_ranges=[
                make_range(1, 0, 4, range_id="first"),
                make_range(1, 4, 8, range_id="second"),
            ],
        )

        assert outputs[0].range_id == "second"
        assert len(retired) == 1
        assert retired[0].range_id == "first"

    def test_merge_three_inputs(self):
        def label_fn(events):
            return [make_range(0, 0, 9)]

        seg = TimeRangeSegmenter(label_fn=label_fn)
        outputs, retired = seg.process(
            parent_id=1,
            events=[make_event(1, 2)],
            input_ranges=[
                make_range(1, 0, 3, range_id="r1"),
                make_range(1, 3, 6, range_id="r2"),
                make_range(1, 6, 9, range_id="r3"),
            ],
        )

        assert len(outputs) == 1
        assert outputs[0].range_id == "r1"
        assert {r.range_id for r in retired} == {"r2", "r3"}


# ---------------------------------------------------------------------------
# Complex case
# ---------------------------------------------------------------------------


class TestComplexCase:
    """M-to-N overlap component (M > 1, N > 1)."""

    def test_default_complex_all_none(self):
        """Default behavior: no ids assigned, all input ids retired."""

        def label_fn(events):
            # Two overlapping output ranges that together form an M:N component
            # with two overlapping input ranges
            return [make_range(0, 0, 5), make_range(0, 3, 8)]

        seg = TimeRangeSegmenter(label_fn=label_fn)
        outputs, retired = seg.process(
            parent_id=1,
            events=[make_event(1, 2)],
            input_ranges=[
                make_range(1, 0, 4, range_id="r1"),
                make_range(1, 4, 8, range_id="r2"),
            ],
        )

        assert all(r.range_id is None for r in outputs)
        assert {r.range_id for r in retired} == {"r1", "r2"}

    def test_custom_complex_fn(self):
        def label_fn(events):
            return [make_range(0, 0, 5), make_range(0, 3, 8)]

        def my_complex(input_ranges, output_ranges):
            # Map first input id to first output, second to second
            return [
                input_ranges[0].range_id if len(input_ranges) > 0 else None,
                input_ranges[1].range_id if len(input_ranges) > 1 else None,
            ]

        seg = TimeRangeSegmenter(label_fn=label_fn, complex_fn=my_complex)
        outputs, retired = seg.process(
            parent_id=1,
            events=[make_event(1, 2)],
            input_ranges=[
                make_range(1, 0, 4, range_id="r1"),
                make_range(1, 4, 8, range_id="r2"),
            ],
        )

        output_ids = {r.range_id for r in outputs}
        assert output_ids == {"r1", "r2"}
        assert retired == []

    def test_custom_complex_partial_assignment(self):
        """Complex fn assigns only some ids; others are retired."""

        def label_fn(events):
            return [make_range(0, 0, 5), make_range(0, 3, 8)]

        def partial_complex(input_ranges, output_ranges):
            return [input_ranges[0].range_id, None]

        seg = TimeRangeSegmenter(label_fn=label_fn, complex_fn=partial_complex)
        outputs, retired = seg.process(
            parent_id=1,
            events=[make_event(1, 2)],
            input_ranges=[
                make_range(1, 0, 4, range_id="r1"),
                make_range(1, 4, 8, range_id="r2"),
            ],
        )

        assigned = [r.range_id for r in outputs]
        assert "r1" in assigned
        assert None in assigned
        assert len(retired) == 1
        assert retired[0].range_id == "r2"


# ---------------------------------------------------------------------------
# No input ranges
# ---------------------------------------------------------------------------


class TestNoInputRanges:
    def test_outputs_all_get_none_id(self):
        def label_fn(events):
            return [make_range(0, 0, 5), make_range(0, 5, 10)]

        seg = TimeRangeSegmenter(label_fn=label_fn)
        outputs, retired = seg.process(
            parent_id=1, events=[make_event(1, 2)], input_ranges=[]
        )

        assert all(r.range_id is None for r in outputs)
        assert retired == []

    def test_none_input_ranges_treated_as_empty(self):
        def label_fn(events):
            return [make_range(0, 0, 5)]

        seg = TimeRangeSegmenter(label_fn=label_fn)
        outputs, retired = seg.process(
            parent_id=1, events=[make_event(1, 2)], input_ranges=None
        )

        assert outputs[0].range_id is None
        assert retired == []


# ---------------------------------------------------------------------------
# Retired ranges
# ---------------------------------------------------------------------------


class TestRetiredRanges:
    def test_non_overlapping_input_is_retired(self):
        def label_fn(events):
            return [make_range(0, 10, 20)]

        seg = TimeRangeSegmenter(label_fn=label_fn)
        outputs, retired = seg.process(
            parent_id=1,
            events=[make_event(1, 12)],
            input_ranges=[make_range(1, 0, 5, range_id="old")],
        )

        assert outputs[0].range_id is None
        assert len(retired) == 1
        assert retired[0].range_id == "old"

    def test_input_with_none_range_id_not_in_retired(self):
        """Inputs with no id are not emitted to the retired stream."""

        def label_fn(events):
            return [make_range(0, 10, 20)]

        seg = TimeRangeSegmenter(label_fn=label_fn)
        outputs, retired = seg.process(
            parent_id=1,
            events=[make_event(1, 12)],
            input_ranges=[make_range(1, 0, 5, range_id=None)],
        )

        assert retired == []

    def test_empty_label_fn_retires_all_inputs(self):
        """No output ranges → all input range ids are retired."""

        def label_fn(events):
            return []

        seg = TimeRangeSegmenter(label_fn=label_fn)
        outputs, retired = seg.process(
            parent_id=1,
            events=[],
            input_ranges=[
                make_range(1, 0, 5, range_id="a"),
                make_range(1, 5, 10, range_id="b"),
            ],
        )

        assert outputs == []
        assert {r.range_id for r in retired} == {"a", "b"}

    def test_mixed_retired_and_preserved(self):
        """Some inputs are matched (preserved), others retired."""

        def label_fn(events):
            # Only one output range, overlapping with first input
            return [make_range(0, 0, 5)]

        seg = TimeRangeSegmenter(label_fn=label_fn)
        outputs, retired = seg.process(
            parent_id=1,
            events=[make_event(1, 2)],
            input_ranges=[
                make_range(1, 0, 5, range_id="kept"),
                make_range(1, 10, 15, range_id="dropped"),
            ],
        )

        assert outputs[0].range_id == "kept"
        assert len(retired) == 1
        assert retired[0].range_id == "dropped"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_events_empty_inputs(self):
        def label_fn(events):
            return []

        seg = TimeRangeSegmenter(label_fn=label_fn)
        outputs, retired = seg.process(parent_id=1, events=[], input_ranges=[])
        assert outputs == []
        assert retired == []

    def test_single_event_single_output(self):
        def label_fn(events):
            e = events[0]
            return [Range(parent_id=0, start_time=e.ts, end_time=e.ts + timedelta(hours=1))]

        seg = TimeRangeSegmenter(label_fn=label_fn)
        outputs, retired = seg.process(
            parent_id=7,
            events=[make_event(7, 5)],
            input_ranges=[],
        )

        assert len(outputs) == 1
        assert outputs[0].parent_id == 7
        assert outputs[0].range_id is None

    def test_multiple_parents_independent(self):
        """Calling process twice for different parents is independent."""

        def label_fn(events):
            return [make_range(0, 0, 5)]

        seg = TimeRangeSegmenter(label_fn=label_fn)

        out1, ret1 = seg.process(
            parent_id=1,
            events=[make_event(1, 2)],
            input_ranges=[make_range(1, 0, 5, range_id="id-for-parent-1")],
        )
        out2, ret2 = seg.process(
            parent_id=2,
            events=[make_event(2, 2)],
            input_ranges=[make_range(2, 0, 5, range_id="id-for-parent-2")],
        )

        assert out1[0].range_id == "id-for-parent-1"
        assert out2[0].range_id == "id-for-parent-2"
