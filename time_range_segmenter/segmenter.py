"""Core scaffold for time-range segmentation with sticky identifier assignment.

The :class:`TimeRangeSegmenter` takes a user-provided *label function* that maps
a list of :class:`~time_range_segmenter.models.Event` objects to a list of
:class:`~time_range_segmenter.models.Range` objects, and attempts to preserve
range identifiers across pipeline re-runs.

Algorithm
---------
Given a set of *input ranges* (from the previous run) and *output ranges* (from the
label function), the segmenter builds a bipartite overlap graph and finds its
connected components.  Each component is classified as one of four cases:

* **1-to-1** – exactly one input range overlaps with exactly one output range.
  The ``range_id`` is preserved automatically.
* **split** – exactly one input range overlaps with *N > 1* output ranges.
  The configurable ``split_fn`` selects which output range inherits the identifier.
* **merge** – *N > 1* input ranges all overlap with exactly one output range.
  The configurable ``merge_fn`` selects which input range donates its identifier.
* **complex** – any other M-to-N relationship.  The configurable ``complex_fn``
  may apply arbitrary logic to assign identifiers.

Output ranges that receive no identifier have ``range_id = None``, indicating that
an external system should assign a new one.  Input ranges whose identifiers were
not preserved are emitted as *retired ranges*.
"""

from collections import deque
from typing import Callable, Dict, List, Optional, Tuple

from .models import Event, Range

# ---------------------------------------------------------------------------
# Type aliases for the user-supplied hook functions
# ---------------------------------------------------------------------------

LabelFn = Callable[[List[Event]], List[Range]]
"""``fn(events) -> ranges`` — converts events to output ranges for a parent."""

SplitFn = Callable[[Range, List[Range]], Range]
"""``fn(input_range, output_ranges) -> chosen_output_range``

Called when a single input range overlaps with *multiple* output ranges (a split).
The function returns the output range that should inherit the input's ``range_id``.
All other output ranges in the component receive ``range_id = None``.

The default implementation picks the chronologically first output range.
"""

MergeFn = Callable[[List[Range], Range], Range]
"""``fn(input_ranges, output_range) -> chosen_input_range``

Called when multiple input ranges all overlap with a single output range (a merge).
The function returns the input range whose ``range_id`` should be used.
All other input ranges in the component are retired.

The default implementation picks the chronologically first input range.
"""

ComplexFn = Callable[[List[Range], List[Range]], List[Optional[str]]]
"""``fn(input_ranges, output_ranges) -> list[range_id | None]``

Called for complex M-to-N overlap components where neither the split nor the merge
pattern applies.  Returns a list of ``range_id`` values (or ``None``) to assign to
the output ranges, in the same order as *output_ranges*.

The default implementation assigns ``None`` to all output ranges, causing them all
to be treated as new (no sticky identifier).
"""


# ---------------------------------------------------------------------------
# Helper: overlap detection
# ---------------------------------------------------------------------------


def ranges_overlap(r1: Range, r2: Range) -> bool:
    """Return ``True`` if the two ranges have overlapping time intervals.

    Ranges are treated as half-open ``[start, end)`` intervals, so ranges that
    merely touch at a single point (one's end equals the other's start) do **not**
    overlap.
    """
    return r1.start_time < r2.end_time and r2.start_time < r1.end_time


# ---------------------------------------------------------------------------
# Helper: connected-component detection in a bipartite overlap graph
# ---------------------------------------------------------------------------


def _find_connected_components(
    n_inputs: int,
    n_outputs: int,
    input_to_outputs: Dict[int, List[int]],
    output_to_inputs: Dict[int, List[int]],
) -> List[Tuple[List[int], List[int]]]:
    """Return connected components of the bipartite input/output overlap graph.

    Each component is a ``(input_indices, output_indices)`` pair.  Output ranges
    that do not overlap with any input range are returned as singleton components
    with an empty input list: ``([], [j])``.
    """
    visited_inputs: set = set()
    visited_outputs: set = set()
    components: List[Tuple[List[int], List[int]]] = []

    def _bfs(start_input: int) -> Tuple[List[int], List[int]]:
        comp_inputs: List[int] = []
        comp_outputs: List[int] = []
        queue: deque = deque([("input", start_input)])
        while queue:
            kind, idx = queue.popleft()
            if kind == "input":
                if idx in visited_inputs:
                    continue
                visited_inputs.add(idx)
                comp_inputs.append(idx)
                for j in input_to_outputs[idx]:
                    if j not in visited_outputs:
                        queue.append(("output", j))
            else:
                if idx in visited_outputs:
                    continue
                visited_outputs.add(idx)
                comp_outputs.append(idx)
                for i in output_to_inputs[idx]:
                    if i not in visited_inputs:
                        queue.append(("input", i))
        return comp_inputs, comp_outputs

    for i in range(n_inputs):
        if i not in visited_inputs:
            components.append(_bfs(i))

    # Output ranges that are not reachable from any input
    for j in range(n_outputs):
        if j not in visited_outputs:
            components.append(([], [j]))

    return components


# ---------------------------------------------------------------------------
# Default hook implementations
# ---------------------------------------------------------------------------


def _default_split(input_range: Range, output_ranges: List[Range]) -> Range:
    """Default split strategy: chronologically first output range inherits the id."""
    return min(output_ranges, key=lambda r: r.start_time)


def _default_merge(input_ranges: List[Range], output_range: Range) -> Range:
    """Default merge strategy: chronologically first input range donates the id."""
    return min(input_ranges, key=lambda r: r.start_time)


def _default_complex(
    input_ranges: List[Range], output_ranges: List[Range]
) -> List[Optional[str]]:
    """Default complex strategy: no identifiers are carried over."""
    return [None] * len(output_ranges)


# ---------------------------------------------------------------------------
# Main scaffold
# ---------------------------------------------------------------------------


class TimeRangeSegmenter:
    """Scaffold that assigns sticky identifiers to output time ranges.

    Parameters
    ----------
    label_fn:
        Required.  Converts a list of events into a list of output ranges.
        The ``parent_id`` field of the returned ranges is overwritten by
        :meth:`process` so callers do not need to set it.
    split_fn:
        Optional.  Called when exactly one input range overlaps with multiple
        output ranges.  See :data:`SplitFn`.  Defaults to :func:`_default_split`.
    merge_fn:
        Optional.  Called when multiple input ranges all overlap with a single
        output range.  See :data:`MergeFn`.  Defaults to :func:`_default_merge`.
    complex_fn:
        Optional.  Called for any M-to-N overlap component (M > 1, N > 1).
        See :data:`ComplexFn`.  Defaults to :func:`_default_complex`.

    Notes
    -----
    All hook functions (``label_fn``, ``split_fn``, ``merge_fn``, ``complex_fn``)
    must be picklable when the segmenter is used inside an Apache Beam pipeline.
    Prefer module-level functions over lambdas or closures.
    """

    def __init__(
        self,
        label_fn: LabelFn,
        split_fn: Optional[SplitFn] = None,
        merge_fn: Optional[MergeFn] = None,
        complex_fn: Optional[ComplexFn] = None,
    ) -> None:
        self.label_fn = label_fn
        self.split_fn = split_fn if split_fn is not None else _default_split
        self.merge_fn = merge_fn if merge_fn is not None else _default_merge
        self.complex_fn = complex_fn if complex_fn is not None else _default_complex

    def process(
        self,
        parent_id: int,
        events: List[Event],
        input_ranges: Optional[List[Range]] = None,
    ) -> Tuple[List[Range], List[Range]]:
        """Produce labeled output ranges for *parent_id*.

        Parameters
        ----------
        parent_id:
            The identifier of the parent entity being processed.
        events:
            All events belonging to *parent_id* in this processing window.
        input_ranges:
            Ranges assigned during the previous run (may be empty or ``None``).

        Returns
        -------
        output_ranges:
            Ranges produced by ``label_fn`` with sticky ``range_id`` values where
            possible.  Ranges that could not be matched receive ``range_id = None``.
        retired_ranges:
            Input ranges whose ``range_id`` was not carried over to any output range.
            Only ranges with a non-``None`` ``range_id`` are included here.
        """
        if input_ranges is None:
            input_ranges = []

        output_ranges: List[Range] = self.label_fn(events)
        for r in output_ranges:
            r.parent_id = parent_id

        n_in = len(input_ranges)
        n_out = len(output_ranges)

        # If there are no inputs (or no outputs) there is nothing to match.
        if n_in == 0 or n_out == 0:
            retired = [ir for ir in input_ranges if ir.range_id is not None]
            return output_ranges, retired

        # ----------------------------------------------------------------
        # Build the bipartite overlap graph
        # ----------------------------------------------------------------
        input_to_outputs: Dict[int, List[int]] = {i: [] for i in range(n_in)}
        output_to_inputs: Dict[int, List[int]] = {j: [] for j in range(n_out)}

        for i, ir in enumerate(input_ranges):
            for j, or_ in enumerate(output_ranges):
                if ranges_overlap(ir, or_):
                    input_to_outputs[i].append(j)
                    output_to_inputs[j].append(i)

        components = _find_connected_components(
            n_in, n_out, input_to_outputs, output_to_inputs
        )

        # ----------------------------------------------------------------
        # Assign range IDs per connected component
        # ----------------------------------------------------------------
        id_assignments: Dict[int, Optional[str]] = {j: None for j in range(n_out)}

        for comp_input_idxs, comp_output_idxs in components:
            if not comp_input_idxs:
                # No overlapping inputs → output keeps range_id = None
                continue

            in_ranges = [input_ranges[i] for i in comp_input_idxs]
            out_ranges = [output_ranges[j] for j in comp_output_idxs]
            n_in_c = len(in_ranges)
            n_out_c = len(out_ranges)

            if n_out_c == 0:
                # Input range(s) have no overlapping outputs → all will be retired
                pass

            elif n_in_c == 1 and n_out_c == 1:
                # 1-to-1: preserve unconditionally
                id_assignments[comp_output_idxs[0]] = in_ranges[0].range_id

            elif n_in_c == 1:
                # Split: one input → multiple outputs
                chosen_out = self.split_fn(in_ranges[0], out_ranges)
                chosen_j = comp_output_idxs[out_ranges.index(chosen_out)]
                id_assignments[chosen_j] = in_ranges[0].range_id

            elif n_out_c == 1:
                # Merge: multiple inputs → one output
                chosen_in = self.merge_fn(in_ranges, out_ranges[0])
                id_assignments[comp_output_idxs[0]] = chosen_in.range_id

            else:
                # Complex: M-to-N (M > 1, N > 1)
                assigned_ids = self.complex_fn(in_ranges, out_ranges)
                for k, j in enumerate(comp_output_idxs):
                    id_assignments[j] = assigned_ids[k] if k < len(assigned_ids) else None

        # ----------------------------------------------------------------
        # Apply assignments and collect retired ranges
        # ----------------------------------------------------------------
        used_range_ids: set = set()
        for j, out_r in enumerate(output_ranges):
            out_r.range_id = id_assignments[j]
            if out_r.range_id is not None:
                used_range_ids.add(out_r.range_id)

        retired = [
            ir
            for ir in input_ranges
            if ir.range_id is not None and ir.range_id not in used_range_ids
        ]

        return output_ranges, retired
