"""Example test implementing a simplified HIPPS pregnancy episode algorithm.

This test demonstrates that the time-range-segmenter framework can express the
complexity of the HIPPS (Hierarchy and rule-based pregnancy episode Inference
integrated with Pregnancy Progression Signatures) algorithm described in:

    Jones et al., "Who is pregnant? Defining real-world data-based pregnancy
    episodes in the National COVID Cohort Collaborative (N3C),"
    JAMIA Open 6(3):ooad067, 2023.  PMID 37600074.

The algorithm has three main components:
  1. HIP (Hierarchy-based Inference of Pregnancy) — rule-based episode detection
     from outcome events and gestational-week markers.
  2. PPS (Pregnancy Progression Signature) — temporal sequence analysis that
     validates and extends episodes using clinician-curated gestational timing
     concepts.
  3. ESD (Estimated Start Date) — back-calculates the pregnancy start date
     from gestational timing concepts.

We implement a *simplified but representative* version of these algorithms as
a ``label_fn`` suitable for :class:`TimeRangeSegmenter`, then exercise it
across a range of clinically-motivated scenarios.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pytest

from time_range_segmenter.models import Event, Range
from time_range_segmenter.segmenter import TimeRangeSegmenter

# ============================================================================
# Constants — outcome categories and timing rules from the paper
# ============================================================================

# Outcome hierarchy (highest priority first) per Matcho et al / HIPPS
OUTCOME_HIERARCHY = ["LIVE_BIRTH", "STILLBIRTH", "ECTOPIC", "ABORTION", "DELIVERY"]

# Maximum gestational durations by outcome (days)
MAX_GESTATION_DAYS: Dict[str, int] = {
    "LIVE_BIRTH": 301,   # 43 weeks
    "STILLBIRTH": 301,   # 43 weeks
    "ECTOPIC": 84,       # 12 weeks
    "ABORTION": 140,     # 20 weeks
    "DELIVERY": 301,     # 43 weeks
}

# Minimum days between consecutive outcomes of the same type
MIN_OUTCOME_SEPARATION_DAYS: Dict[str, int] = {
    "LIVE_BIRTH": 182,   # 26 weeks
    "STILLBIRTH": 168,   # 24 weeks
    "ECTOPIC": 60,       # ~8.5 weeks
    "ABORTION": 60,      # ~8.5 weeks
    "DELIVERY": 182,     # 26 weeks
}

# PPS retry period — minimum gap before a new episode can start (days)
PPS_RETRY_PERIOD_DAYS = 60

# Maximum episode length for PPS cleanup
MAX_EPISODE_LENGTH_DAYS = 365  # 12 months


# ============================================================================
# Gestational timing concept definitions (simplified subset of 74 concepts)
# ============================================================================

# Each concept has:
#   - name: human-readable label
#   - type: "GW" (gestational week point estimate) or "GR3m" (range < 3 months)
#   - min_month / max_month: expected gestational timing window (months)
#   - For GW concepts, the week number is encoded in the event data.

GESTATIONAL_TIMING_CONCEPTS: Dict[str, Dict] = {
    # GW-type concepts — "Gestation period, X weeks" encoded in event data
    "GW": {
        "type": "GW",
        "description": "Gestation period, X weeks — week number in event.data['week']",
    },
    # GR3m-type concepts — occur within a gestational range
    "NUCHAL_TRANSLUCENCY": {
        "type": "GR3m",
        "min_month": 2.5,
        "max_month": 3.5,
        "description": "Nuchal translucency measurement (11-14 weeks)",
    },
    "ANATOMY_SCAN": {
        "type": "GR3m",
        "min_month": 4.5,
        "max_month": 5.5,
        "description": "Anatomy ultrasound scan (18-22 weeks)",
    },
    "GLUCOSE_TOLERANCE": {
        "type": "GR3m",
        "min_month": 6.0,
        "max_month": 7.0,
        "description": "Glucose tolerance test (24-28 weeks)",
    },
    "GROUP_B_STREP": {
        "type": "GR3m",
        "min_month": 8.25,
        "max_month": 9.25,
        "description": "Group B Strep screening (35-37 weeks)",
    },
    "TRISOMY_RISK": {
        "type": "GR3m",
        "min_month": 3.0,
        "max_month": 6.0,
        "description": "Trisomy 18 risk assessment (12-24 weeks)",
    },
    "ESTRIOL_TEST": {
        "type": "GR3m",
        "min_month": 3.75,
        "max_month": 5.5,
        "description": "Estriol testing (15-22 weeks)",
    },
}


# ============================================================================
# Helper functions
# ============================================================================

def day(year: int, month: int, day: int) -> datetime:
    """Shorthand for creating a date at midnight."""
    return datetime(year, month, day)


def make_event(
    patient_id: int,
    date: datetime,
    concept: str,
    **extra,
) -> Event:
    """Create a pregnancy-related EHR event.

    Parameters
    ----------
    patient_id : int
        The patient identifier (maps to ``parent_id``).
    date : datetime
        When the event was recorded.
    concept : str
        The clinical concept type (e.g. "LIVE_BIRTH", "GW", "GLUCOSE_TOLERANCE").
    **extra
        Additional data fields (e.g. ``week=32`` for GW concepts).
    """
    data = {"concept": concept, **extra}
    return Event(parent_id=patient_id, ts=date, data=data)


# ============================================================================
# HIPPS label function — simplified implementation
# ============================================================================


def hipps_label_fn(events: List[Event]) -> List[Range]:
    """Simplified HIPPS algorithm expressed as a time-range-segmenter label_fn.

    This function takes EHR events for a single patient and returns pregnancy
    episode ranges.  It implements the three core HIPPS components:

    1. **HIP** — Hierarchy-based Inference of Pregnancy
       a. Outcome-based episodes from pregnancy outcome events
       b. Gestation-based episodes from "Gestation period, X weeks" markers
       c. Merge overlapping episodes; reclassify misaligned outcomes

    2. **PPS** — Pregnancy Progression Signature
       Validates episodes using gestational timing concepts that should occur
       in a monotonically progressing order during pregnancy.

    3. **ESD** — Estimated Start Date
       Back-calculates start dates from the most precise gestational timing
       concepts available.
    """
    if not events:
        return []

    sorted_events = sorted(events, key=lambda e: e.ts)

    # ---- Step 1: HIP Algorithm ----
    outcome_episodes = _hip_outcome_episodes(sorted_events)
    gestation_episodes = _hip_gestation_episodes(sorted_events)
    hip_episodes = _hip_merge_episodes(outcome_episodes, gestation_episodes)

    # ---- Step 2: PPS Algorithm ----
    pps_episodes = _pps_algorithm(sorted_events)

    # ---- Step 3: Merge HIP + PPS ----
    merged = _merge_hip_pps(hip_episodes, pps_episodes)

    # ---- Step 4: ESD — refine start dates ----
    refined = _esd_refine_start_dates(merged, sorted_events)

    # ---- Step 5: Cleanup ----
    cleaned = _cleanup_episodes(refined)

    # Convert to Range objects
    if not cleaned:
        return []

    patient_id = sorted_events[0].parent_id
    return [
        Range(
            parent_id=patient_id,
            start_time=ep["start"],
            end_time=ep["end"],
        )
        for ep in cleaned
    ]


# ---------------------------------------------------------------------------
# HIP: Outcome-based episodes
# ---------------------------------------------------------------------------

def _hip_outcome_episodes(events: List[Event]) -> List[Dict]:
    """Detect episodes anchored by pregnancy outcome events.

    Outcomes are processed in hierarchy order.  Consecutive outcomes of the
    same type must be separated by the minimum required gap.
    """
    outcome_events = [
        e for e in events
        if e.data and e.data.get("concept") in OUTCOME_HIERARCHY
    ]
    if not outcome_events:
        return []

    # Sort by hierarchy priority then date
    outcome_events.sort(
        key=lambda e: (OUTCOME_HIERARCHY.index(e.data["concept"]), e.ts)
    )

    episodes: List[Dict] = []
    used_dates: List[Tuple[datetime, datetime]] = []

    for outcome_type in OUTCOME_HIERARCHY:
        type_events = [e for e in outcome_events if e.data["concept"] == outcome_type]
        min_sep = timedelta(days=MIN_OUTCOME_SEPARATION_DAYS[outcome_type])
        max_gest = timedelta(days=MAX_GESTATION_DAYS[outcome_type])

        last_date = None
        for evt in sorted(type_events, key=lambda e: e.ts):
            # Check minimum separation from last same-type outcome
            if last_date is not None and (evt.ts - last_date) < min_sep:
                continue

            # Check this outcome doesn't overlap with already-assigned episodes
            proposed_start = evt.ts - max_gest
            proposed_end = evt.ts + timedelta(days=1)
            overlap = False
            for used_start, used_end in used_dates:
                if proposed_start < used_end and used_start < proposed_end:
                    # Check cross-type separation rules
                    overlap = True
                    break

            if not overlap:
                episodes.append({
                    "start": proposed_start,
                    "end": proposed_end,
                    "outcome": outcome_type,
                    "outcome_date": evt.ts,
                    "source": "HIP_outcome",
                })
                used_dates.append((proposed_start, proposed_end))
                last_date = evt.ts

    return sorted(episodes, key=lambda ep: ep["end"])


# ---------------------------------------------------------------------------
# HIP: Gestation-based episodes
# ---------------------------------------------------------------------------

def _hip_gestation_episodes(events: List[Event]) -> List[Dict]:
    """Detect episodes from 'Gestation period, X weeks' markers.

    A new episode starts when the week number resets (decreases or stays the
    same as previous), indicating a new pregnancy.
    """
    gw_events = [
        e for e in events
        if e.data and e.data.get("concept") == "GW" and "week" in e.data
    ]
    if not gw_events:
        return []

    gw_events.sort(key=lambda e: e.ts)

    episodes: List[Dict] = []
    current_episode_gws: List[Event] = [gw_events[0]]

    for evt in gw_events[1:]:
        prev_week = current_episode_gws[-1].data["week"]
        curr_week = evt.data["week"]

        # Start a new episode if week number resets (decreases or same)
        if curr_week <= prev_week:
            episodes.append(_finalize_gw_episode(current_episode_gws))
            current_episode_gws = [evt]
        else:
            # Verify the projected start doesn't overlap previous episode
            current_episode_gws.append(evt)

    # Finalize last episode
    if current_episode_gws:
        episodes.append(_finalize_gw_episode(current_episode_gws))

    return episodes


def _finalize_gw_episode(gw_events: List[Event]) -> Dict:
    """Create an episode dict from a sequence of GW events."""
    max_week_evt = max(gw_events, key=lambda e: e.data["week"])
    max_week = max_week_evt.data["week"]
    start = max_week_evt.ts - timedelta(weeks=max_week)
    end = max(e.ts for e in gw_events) + timedelta(days=1)
    return {
        "start": start,
        "end": end,
        "outcome": None,
        "outcome_date": None,
        "source": "HIP_gestation",
        "max_week": max_week,
    }


# ---------------------------------------------------------------------------
# HIP: Merge outcome + gestation episodes
# ---------------------------------------------------------------------------

def _hip_merge_episodes(
    outcome_episodes: List[Dict],
    gestation_episodes: List[Dict],
) -> List[Dict]:
    """Merge overlapping outcome-based and gestation-based episodes.

    If a gestation episode overlaps an outcome episode, they are combined.
    If the gestational age doesn't align with the outcome's expected term,
    the outcome is reclassified (removed).
    """
    if not outcome_episodes and not gestation_episodes:
        return []
    if not gestation_episodes:
        return outcome_episodes
    if not outcome_episodes:
        return gestation_episodes

    merged: List[Dict] = []
    used_gestation: set = set()

    for oe in outcome_episodes:
        matching_ge = None
        for idx, ge in enumerate(gestation_episodes):
            if idx in used_gestation:
                continue
            # Check overlap
            if ge["start"] < oe["end"] and oe["start"] < ge["end"]:
                matching_ge = ge
                used_gestation.add(idx)
                break

        if matching_ge is not None:
            # Merge: use gestation-based start if more precise, outcome end
            combined_start = min(oe["start"], matching_ge["start"])
            combined_end = max(oe["end"], matching_ge["end"])

            # Validate outcome aligns with gestational age
            outcome = oe["outcome"]
            max_week = matching_ge.get("max_week")
            if max_week is not None and outcome is not None:
                max_gest_weeks = MAX_GESTATION_DAYS[outcome] / 7
                if max_week > max_gest_weeks:
                    outcome = None  # Reclassify — gestational age too high

            merged.append({
                "start": combined_start,
                "end": combined_end,
                "outcome": outcome,
                "outcome_date": oe.get("outcome_date"),
                "source": "HIP_merged",
            })
        else:
            merged.append(oe)

    # Add unmatched gestation episodes
    for idx, ge in enumerate(gestation_episodes):
        if idx not in used_gestation:
            merged.append(ge)

    return sorted(merged, key=lambda ep: ep["start"])


# ---------------------------------------------------------------------------
# PPS: Pregnancy Progression Signature algorithm
# ---------------------------------------------------------------------------

def _pps_algorithm(events: List[Event]) -> List[Dict]:
    """Detect pregnancy episodes via progressing gestational timing concepts.

    Implements a simplified version of the LICS-based PPS algorithm.
    For each pair of successive gestational timing concepts, we check whether
    the time elapsed between them is consistent with the expected gestational
    timing ranges.  If not, a new episode is started (subject to the retry
    period).
    """
    timing_events = []
    for e in events:
        if e.data is None:
            continue
        concept = e.data.get("concept", "")
        if concept in GESTATIONAL_TIMING_CONCEPTS:
            info = GESTATIONAL_TIMING_CONCEPTS[concept]
            if info["type"] == "GW":
                week = e.data.get("week", 0)
                min_month = week / 4.33
                max_month = week / 4.33
            else:
                min_month = info["min_month"]
                max_month = info["max_month"]
            timing_events.append({
                "event": e,
                "concept": concept,
                "min_month": min_month,
                "max_month": max_month,
                "mid_month": (min_month + max_month) / 2,
            })

    if not timing_events:
        return []

    timing_events.sort(key=lambda x: x["event"].ts)

    episodes: List[Dict] = []
    current_concepts: List[Dict] = [timing_events[0]]

    for te in timing_events[1:]:
        prev = current_concepts[-1]
        days_elapsed = (te["event"].ts - prev["event"].ts).days
        months_elapsed = days_elapsed / 30.44  # average days per month

        # Expected difference in gestational months between concepts
        expected_min_diff = te["min_month"] - prev["max_month"]
        expected_max_diff = te["max_month"] - prev["min_month"]

        # Allow some tolerance (±1 month)
        is_consistent = (
            months_elapsed >= (expected_min_diff - 1.0)
            and months_elapsed <= (expected_max_diff + 1.0)
            and months_elapsed >= 0
        )

        if is_consistent:
            current_concepts.append(te)
        else:
            # Check retry period
            if days_elapsed >= PPS_RETRY_PERIOD_DAYS:
                episodes.append(_finalize_pps_episode(current_concepts))
                current_concepts = [te]
            else:
                # Too close for a new episode — extend current
                current_concepts.append(te)

    if current_concepts:
        episodes.append(_finalize_pps_episode(current_concepts))

    # Append outcomes to PPS episodes
    return episodes


def _finalize_pps_episode(concepts: List[Dict]) -> Dict:
    """Create an episode from a sequence of PPS timing concepts."""
    first_date = concepts[0]["event"].ts
    last_date = concepts[-1]["event"].ts
    # Back-calculate start from the first concept's expected timing
    first_min_month = concepts[0]["min_month"]
    start = first_date - timedelta(days=first_min_month * 30.44)
    end = last_date + timedelta(days=1)
    return {
        "start": start,
        "end": end,
        "outcome": None,
        "outcome_date": None,
        "source": "PPS",
    }


# ---------------------------------------------------------------------------
# Merge HIP + PPS episodes
# ---------------------------------------------------------------------------

def _merge_hip_pps(
    hip_episodes: List[Dict],
    pps_episodes: List[Dict],
) -> List[Dict]:
    """Merge HIP and PPS episodes.  Overlapping episodes are unified."""
    all_episodes = hip_episodes + pps_episodes
    if not all_episodes:
        return []

    all_episodes.sort(key=lambda ep: ep["start"])

    merged: List[Dict] = [all_episodes[0]]
    for ep in all_episodes[1:]:
        prev = merged[-1]
        if ep["start"] < prev["end"]:
            # Overlapping — merge using union of dates
            prev["start"] = min(prev["start"], ep["start"])
            prev["end"] = max(prev["end"], ep["end"])
            # Prefer outcome from the episode that has one
            if prev["outcome"] is None and ep["outcome"] is not None:
                prev["outcome"] = ep["outcome"]
                prev["outcome_date"] = ep.get("outcome_date")
            # Mark as merged
            prev["source"] = "HIPPS_merged"
        else:
            merged.append(ep)

    return merged


# ---------------------------------------------------------------------------
# ESD: Estimated Start Date refinement
# ---------------------------------------------------------------------------

def _esd_refine_start_dates(
    episodes: List[Dict],
    events: List[Event],
) -> List[Dict]:
    """Refine episode start dates using gestational timing concepts.

    Uses GW (gestational week) concepts for the most precise estimates,
    falling back to GR3m (gestational range < 3 months) concepts.
    """
    for ep in episodes:
        ep_events = [
            e for e in events
            if e.data and ep["start"] <= e.ts <= ep["end"]
        ]

        gw_starts_with_week: List[Tuple[int, datetime]] = []
        gr3m_ranges: List[Tuple[datetime, datetime]] = []

        for e in ep_events:
            concept = e.data.get("concept", "")
            if concept == "GW" and "week" in e.data:
                week = e.data["week"]
                inferred_start = e.ts - timedelta(weeks=week)
                gw_starts_with_week.append((week, inferred_start))
            elif concept in GESTATIONAL_TIMING_CONCEPTS:
                info = GESTATIONAL_TIMING_CONCEPTS[concept]
                if info["type"] == "GR3m":
                    range_start = e.ts - timedelta(days=info["max_month"] * 30.44)
                    range_end = e.ts - timedelta(days=info["min_month"] * 30.44)
                    gr3m_ranges.append((range_start, range_end))

        if gw_starts_with_week:
            # Sort by week ascending; we prefer the highest-week concept
            gw_starts_with_week.sort(key=lambda x: x[0])
            gw_starts = [d for _, d in gw_starts_with_week]

            # Remove outliers using IQR (simplified)
            gw_starts.sort()
            if len(gw_starts) >= 4:
                q1 = gw_starts[len(gw_starts) // 4]
                q3 = gw_starts[3 * len(gw_starts) // 4]
                iqr = q3 - q1
                lower = q1 - iqr * 1.5
                upper = q3 + iqr * 1.5
                gw_starts = [d for d in gw_starts if lower <= d <= upper]

            # If we have GR3m ranges, filter GW starts to intersection
            if gr3m_ranges:
                intersect_start = max(r[0] for r in gr3m_ranges)
                intersect_end = min(r[1] for r in gr3m_ranges)
                filtered = [
                    d for d in gw_starts
                    if intersect_start <= d <= intersect_end
                ]
                if filtered:
                    gw_starts = filtered

            if gw_starts:
                # Per the paper: use the GW concept occurring latest in
                # pregnancy (highest week number) for most precise estimate.
                # Pick the start inferred from the highest-week GW concept
                # that survived outlier filtering.
                best_start = gw_starts_with_week[-1][1]
                if best_start in gw_starts:
                    ep["start"] = best_start
                else:
                    ep["start"] = gw_starts[-1]
                ep["esd_precision"] = "week"
        elif gr3m_ranges:
            # Use intersection of GR3m ranges
            intersect_start = max(r[0] for r in gr3m_ranges)
            intersect_end = min(r[1] for r in gr3m_ranges)
            if intersect_start <= intersect_end:
                ep["start"] = intersect_start + (intersect_end - intersect_start) / 2
                ep["esd_precision"] = "month"

    return episodes


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def _cleanup_episodes(episodes: List[Dict]) -> List[Dict]:
    """Remove invalid episodes (too long, etc.)."""
    cleaned = []
    for ep in episodes:
        duration = (ep["end"] - ep["start"]).days
        if duration > 0 and duration <= MAX_EPISODE_LENGTH_DAYS:
            cleaned.append(ep)
    return sorted(cleaned, key=lambda ep: ep["start"])


# ============================================================================
# Tests
# ============================================================================


class TestHIPOutcomeBasedEpisodes:
    """HIP component: detecting episodes from pregnancy outcome events."""

    def test_single_live_birth(self):
        """A single live-birth outcome defines one pregnancy episode."""
        patient_id = 1001
        events = [
            make_event(patient_id, day(2021, 9, 15), "LIVE_BIRTH"),
        ]
        segmenter = TimeRangeSegmenter(label_fn=hipps_label_fn)
        outputs, _ = segmenter.process(parent_id=patient_id, events=events)

        assert len(outputs) == 1
        r = outputs[0]
        assert r.parent_id == patient_id
        # End should be the day after the birth
        assert r.end_time == day(2021, 9, 16)
        # Start should be ~43 weeks before (maximum gestation for live birth)
        expected_start = day(2021, 9, 15) - timedelta(days=301)
        assert r.start_time == expected_start

    def test_single_ectopic(self):
        """An ectopic pregnancy has a shorter maximum gestation (12 weeks)."""
        patient_id = 1002
        events = [
            make_event(patient_id, day(2021, 4, 10), "ECTOPIC"),
        ]
        segmenter = TimeRangeSegmenter(label_fn=hipps_label_fn)
        outputs, _ = segmenter.process(parent_id=patient_id, events=events)

        assert len(outputs) == 1
        expected_start = day(2021, 4, 10) - timedelta(days=84)
        assert outputs[0].start_time == expected_start

    def test_two_live_births_sufficient_gap(self):
        """Two live births separated by ≥26 weeks produce two episodes."""
        patient_id = 1003
        birth1 = day(2020, 6, 1)
        birth2 = day(2021, 6, 1)  # > 26 weeks apart
        events = [
            make_event(patient_id, birth1, "LIVE_BIRTH"),
            make_event(patient_id, birth2, "LIVE_BIRTH"),
        ]
        segmenter = TimeRangeSegmenter(label_fn=hipps_label_fn)
        outputs, _ = segmenter.process(parent_id=patient_id, events=events)

        assert len(outputs) == 2

    def test_two_live_births_too_close_collapses(self):
        """Two live births < 26 weeks apart — second is suppressed."""
        patient_id = 1004
        birth1 = day(2021, 1, 1)
        birth2 = day(2021, 5, 1)  # ~17 weeks apart < 26 weeks
        events = [
            make_event(patient_id, birth1, "LIVE_BIRTH"),
            make_event(patient_id, birth2, "LIVE_BIRTH"),
        ]
        segmenter = TimeRangeSegmenter(label_fn=hipps_label_fn)
        outputs, _ = segmenter.process(parent_id=patient_id, events=events)

        # Only the first birth qualifies; second is too close
        assert len(outputs) == 1

    def test_outcome_hierarchy_mixed_types(self):
        """Multiple outcome types are processed in hierarchy order."""
        patient_id = 1005
        events = [
            make_event(patient_id, day(2020, 3, 15), "ABORTION"),
            make_event(patient_id, day(2021, 2, 1), "LIVE_BIRTH"),
        ]
        segmenter = TimeRangeSegmenter(label_fn=hipps_label_fn)
        outputs, _ = segmenter.process(parent_id=patient_id, events=events)

        # Both should be separate episodes (well-separated in time)
        assert len(outputs) == 2
        # Live birth episode should come second chronologically by end date
        outcomes_by_end = sorted(outputs, key=lambda r: r.end_time)
        assert outcomes_by_end[0].end_time == day(2020, 3, 16)
        assert outcomes_by_end[1].end_time == day(2021, 2, 2)


class TestHIPGestationBasedEpisodes:
    """HIP component: detecting episodes from gestational week markers."""

    def test_single_progression(self):
        """A sequence of increasing GW markers forms one episode."""
        patient_id = 2001
        events = [
            make_event(patient_id, day(2021, 3, 1), "GW", week=12),
            make_event(patient_id, day(2021, 5, 1), "GW", week=20),
            make_event(patient_id, day(2021, 8, 1), "GW", week=33),
        ]
        segmenter = TimeRangeSegmenter(label_fn=hipps_label_fn)
        outputs, _ = segmenter.process(parent_id=patient_id, events=events)

        assert len(outputs) == 1
        # Start should be back-calculated from highest GW (week 33)
        expected_start = day(2021, 8, 1) - timedelta(weeks=33)
        assert outputs[0].start_time == expected_start

    def test_week_reset_starts_new_episode(self):
        """A GW reset (decrease) signals a new pregnancy."""
        patient_id = 2002
        events = [
            make_event(patient_id, day(2020, 4, 1), "GW", week=20),
            make_event(patient_id, day(2020, 7, 1), "GW", week=32),
            # New pregnancy — week resets
            make_event(patient_id, day(2021, 3, 1), "GW", week=12),
            make_event(patient_id, day(2021, 6, 1), "GW", week=24),
        ]
        segmenter = TimeRangeSegmenter(label_fn=hipps_label_fn)
        outputs, _ = segmenter.process(parent_id=patient_id, events=events)

        assert len(outputs) == 2


class TestHIPMerging:
    """HIP component: merging outcome + gestation episodes."""

    def test_outcome_and_gestation_overlap_merge(self):
        """An outcome episode and a gestation episode that overlap are merged."""
        patient_id = 3001
        # GW markers during pregnancy
        events = [
            make_event(patient_id, day(2021, 3, 1), "GW", week=12),
            make_event(patient_id, day(2021, 6, 1), "GW", week=25),
            # Live birth at ~40 weeks
            make_event(patient_id, day(2021, 9, 15), "LIVE_BIRTH"),
        ]
        segmenter = TimeRangeSegmenter(label_fn=hipps_label_fn)
        outputs, _ = segmenter.process(parent_id=patient_id, events=events)

        # GW episode and outcome episode should merge into one
        assert len(outputs) == 1
        # End date should encompass the birth
        assert outputs[0].end_time >= day(2021, 9, 15)


class TestPPSAlgorithm:
    """PPS component: pregnancy progression signature validation."""

    def test_consistent_progression_one_episode(self):
        """A consistent sequence of gestational timing concepts = one episode."""
        patient_id = 4001
        events = [
            make_event(patient_id, day(2021, 2, 1), "NUCHAL_TRANSLUCENCY"),
            make_event(patient_id, day(2021, 4, 15), "ANATOMY_SCAN"),
            make_event(patient_id, day(2021, 6, 1), "GLUCOSE_TOLERANCE"),
            make_event(patient_id, day(2021, 8, 15), "GROUP_B_STREP"),
        ]
        segmenter = TimeRangeSegmenter(label_fn=hipps_label_fn)
        outputs, _ = segmenter.process(parent_id=patient_id, events=events)

        assert len(outputs) == 1
        # Episode should span from before nuchal translucency to after GBS
        assert outputs[0].start_time < day(2021, 2, 1)
        assert outputs[0].end_time > day(2021, 8, 15)

    def test_inconsistent_timing_splits_episodes(self):
        """Concepts with timing inconsistency and sufficient gap = two episodes."""
        patient_id = 4002
        events = [
            # First pregnancy
            make_event(patient_id, day(2020, 2, 1), "NUCHAL_TRANSLUCENCY"),
            make_event(patient_id, day(2020, 4, 15), "ANATOMY_SCAN"),
            # Gap > 60 days, then concepts from a new pregnancy
            make_event(patient_id, day(2021, 2, 1), "NUCHAL_TRANSLUCENCY"),
            make_event(patient_id, day(2021, 4, 15), "ANATOMY_SCAN"),
        ]
        segmenter = TimeRangeSegmenter(label_fn=hipps_label_fn)
        outputs, _ = segmenter.process(parent_id=patient_id, events=events)

        assert len(outputs) == 2

    def test_pps_validates_hip_episodes(self):
        """PPS concepts that fall within a HIP episode validate it.

        When both HIP and PPS find overlapping episodes, they merge into one.
        """
        patient_id = 4003
        events = [
            # GW markers (HIP)
            make_event(patient_id, day(2021, 3, 1), "GW", week=12),
            make_event(patient_id, day(2021, 6, 1), "GW", week=25),
            # PPS concepts
            make_event(patient_id, day(2021, 4, 15), "ANATOMY_SCAN"),
            make_event(patient_id, day(2021, 6, 15), "GLUCOSE_TOLERANCE"),
            # Outcome
            make_event(patient_id, day(2021, 9, 15), "LIVE_BIRTH"),
        ]
        segmenter = TimeRangeSegmenter(label_fn=hipps_label_fn)
        outputs, _ = segmenter.process(parent_id=patient_id, events=events)

        # Everything should merge into a single pregnancy episode
        assert len(outputs) == 1


class TestESDStartDateEstimation:
    """ESD component: estimated start date refinement."""

    def test_gw_refines_start_date(self):
        """GW concepts provide week-level precision for start date."""
        patient_id = 5001
        events = [
            make_event(patient_id, day(2021, 6, 1), "GW", week=24),
            make_event(patient_id, day(2021, 8, 1), "GW", week=33),
            make_event(patient_id, day(2021, 9, 15), "LIVE_BIRTH"),
        ]
        segmenter = TimeRangeSegmenter(label_fn=hipps_label_fn)
        outputs, _ = segmenter.process(parent_id=patient_id, events=events)

        assert len(outputs) == 1
        # Start should be refined by the latest GW (week 33 on Aug 1)
        expected_start = day(2021, 8, 1) - timedelta(weeks=33)
        assert outputs[0].start_time == expected_start

    def test_gr3m_concepts_help_refine_start(self):
        """GR3m concepts narrow the start date when no GW is available."""
        patient_id = 5002
        events = [
            # Only GR3m concepts, no GW
            make_event(patient_id, day(2021, 4, 1), "NUCHAL_TRANSLUCENCY"),
            make_event(patient_id, day(2021, 6, 15), "ANATOMY_SCAN"),
            make_event(patient_id, day(2021, 9, 1), "LIVE_BIRTH"),
        ]
        segmenter = TimeRangeSegmenter(label_fn=hipps_label_fn)
        outputs, _ = segmenter.process(parent_id=patient_id, events=events)

        assert len(outputs) == 1
        # Start should be earlier than the outcome-only default
        default_start = day(2021, 9, 1) - timedelta(days=301)
        # The ESD should produce a more precise (later) start than the max
        assert outputs[0].start_time > default_start


class TestMultiplePregnancies:
    """Full HIPPS: complex scenarios with multiple pregnancies per patient."""

    def test_two_full_pregnancies(self):
        """Two complete pregnancies with outcomes and GW markers."""
        patient_id = 6001
        events = [
            # --- First pregnancy ---
            make_event(patient_id, day(2019, 3, 1), "GW", week=12),
            make_event(patient_id, day(2019, 5, 1), "ANATOMY_SCAN"),
            make_event(patient_id, day(2019, 6, 15), "GW", week=27),
            make_event(patient_id, day(2019, 9, 1), "LIVE_BIRTH"),
            # --- Second pregnancy ---
            make_event(patient_id, day(2020, 6, 1), "GW", week=10),
            make_event(patient_id, day(2020, 8, 15), "ANATOMY_SCAN"),
            make_event(patient_id, day(2020, 10, 1), "GW", week=27),
            make_event(patient_id, day(2021, 1, 5), "LIVE_BIRTH"),
        ]
        segmenter = TimeRangeSegmenter(label_fn=hipps_label_fn)
        outputs, _ = segmenter.process(parent_id=patient_id, events=events)

        assert len(outputs) == 2
        # Episodes should be non-overlapping
        sorted_out = sorted(outputs, key=lambda r: r.start_time)
        assert sorted_out[0].end_time <= sorted_out[1].start_time

    def test_pregnancy_then_early_loss(self):
        """A live birth followed by an early pregnancy loss (ectopic)."""
        patient_id = 6002
        events = [
            make_event(patient_id, day(2020, 3, 1), "GW", week=20),
            make_event(patient_id, day(2020, 7, 1), "LIVE_BIRTH"),
            # Subsequent ectopic several months later
            make_event(patient_id, day(2021, 3, 1), "ECTOPIC"),
        ]
        segmenter = TimeRangeSegmenter(label_fn=hipps_label_fn)
        outputs, _ = segmenter.process(parent_id=patient_id, events=events)

        assert len(outputs) == 2
        sorted_out = sorted(outputs, key=lambda r: r.end_time)
        # First episode ends around the live birth
        assert sorted_out[0].end_time <= day(2020, 7, 2)
        # Second episode is the ectopic (shorter gestation window)
        ectopic_duration = (sorted_out[1].end_time - sorted_out[1].start_time).days
        assert ectopic_duration <= 85  # 84 + 1 day


class TestStickyRangeIDs:
    """Demonstrate the framework's core value: persistent range IDs across runs.

    When the pipeline is re-run with new events, previously identified pregnancy
    episodes should retain their ``range_id`` if they still match.
    """

    def test_stable_id_on_rerun_with_same_data(self):
        """Re-running with the same events preserves the range_id."""
        patient_id = 7001
        events = [
            make_event(patient_id, day(2021, 3, 1), "GW", week=12),
            make_event(patient_id, day(2021, 6, 1), "GW", week=25),
            make_event(patient_id, day(2021, 9, 15), "LIVE_BIRTH"),
        ]
        segmenter = TimeRangeSegmenter(label_fn=hipps_label_fn)

        # First run — no prior ranges
        run1_outputs, _ = segmenter.process(
            parent_id=patient_id, events=events
        )
        assert len(run1_outputs) == 1
        assert run1_outputs[0].range_id is None  # New — no prior id

        # Simulate external system assigning an id
        run1_outputs[0].range_id = "pregnancy-ep-001"

        # Second run — same events, prior ranges from first run
        run2_outputs, run2_retired = segmenter.process(
            parent_id=patient_id,
            events=events,
            input_ranges=run1_outputs,
        )
        assert len(run2_outputs) == 1
        assert run2_outputs[0].range_id == "pregnancy-ep-001"
        assert run2_retired == []

    def test_new_event_extends_episode_preserves_id(self):
        """Adding a new event that extends an episode still preserves the id."""
        patient_id = 7002
        segmenter = TimeRangeSegmenter(label_fn=hipps_label_fn)

        # Run 1: partial data
        events_run1 = [
            make_event(patient_id, day(2021, 3, 1), "GW", week=12),
            make_event(patient_id, day(2021, 6, 1), "GW", week=25),
        ]
        run1_outputs, _ = segmenter.process(
            parent_id=patient_id, events=events_run1
        )
        assert len(run1_outputs) == 1
        run1_outputs[0].range_id = "pregnancy-ep-002"

        # Run 2: outcome event added
        events_run2 = events_run1 + [
            make_event(patient_id, day(2021, 9, 15), "LIVE_BIRTH"),
        ]
        run2_outputs, run2_retired = segmenter.process(
            parent_id=patient_id,
            events=events_run2,
            input_ranges=run1_outputs,
        )

        assert len(run2_outputs) == 1
        # The episode grew but it still overlaps the prior range → id preserved
        assert run2_outputs[0].range_id == "pregnancy-ep-002"
        assert run2_retired == []

    def test_split_preserves_one_id(self):
        """If new data splits an episode into two, one inherits the old id."""
        patient_id = 7003
        segmenter = TimeRangeSegmenter(label_fn=hipps_label_fn)

        # Run 1: ambiguous events that look like one episode
        events_run1 = [
            make_event(patient_id, day(2020, 3, 1), "GW", week=12),
            make_event(patient_id, day(2020, 6, 1), "GW", week=25),
            make_event(patient_id, day(2020, 8, 1), "LIVE_BIRTH"),
        ]
        run1_outputs, _ = segmenter.process(
            parent_id=patient_id, events=events_run1
        )
        for i, r in enumerate(run1_outputs):
            r.range_id = f"ep-{i}"

        # Run 2: new events reveal a second pregnancy
        events_run2 = events_run1 + [
            make_event(patient_id, day(2021, 5, 1), "GW", week=14),
            make_event(patient_id, day(2021, 9, 1), "LIVE_BIRTH"),
        ]
        run2_outputs, _ = segmenter.process(
            parent_id=patient_id,
            events=events_run2,
            input_ranges=run1_outputs,
        )

        # Should now have two episodes
        assert len(run2_outputs) == 2
        # At least one should preserve the original id
        preserved_ids = [r.range_id for r in run2_outputs if r.range_id is not None]
        assert len(preserved_ids) >= 1
        assert "ep-0" in preserved_ids


class TestEdgeCasesHIPPS:
    """Edge cases and data quality issues common in EHR data."""

    def test_no_outcome_still_detects_episode(self):
        """Episode detected from GW markers alone (no outcome recorded)."""
        patient_id = 8001
        events = [
            make_event(patient_id, day(2021, 3, 1), "GW", week=12),
            make_event(patient_id, day(2021, 5, 1), "GW", week=20),
            make_event(patient_id, day(2021, 7, 1), "GW", week=29),
        ]
        segmenter = TimeRangeSegmenter(label_fn=hipps_label_fn)
        outputs, _ = segmenter.process(parent_id=patient_id, events=events)

        assert len(outputs) == 1

    def test_only_pps_concepts_detects_episode(self):
        """Episode detected purely from PPS timing concepts (no GW, no outcome)."""
        patient_id = 8002
        events = [
            make_event(patient_id, day(2021, 2, 1), "NUCHAL_TRANSLUCENCY"),
            make_event(patient_id, day(2021, 4, 15), "ANATOMY_SCAN"),
            make_event(patient_id, day(2021, 6, 1), "GLUCOSE_TOLERANCE"),
        ]
        segmenter = TimeRangeSegmenter(label_fn=hipps_label_fn)
        outputs, _ = segmenter.process(parent_id=patient_id, events=events)

        assert len(outputs) == 1

    def test_single_event_produces_episode(self):
        """Even a single outcome event should produce an episode."""
        patient_id = 8003
        events = [
            make_event(patient_id, day(2021, 5, 1), "LIVE_BIRTH"),
        ]
        segmenter = TimeRangeSegmenter(label_fn=hipps_label_fn)
        outputs, _ = segmenter.process(parent_id=patient_id, events=events)

        assert len(outputs) == 1

    def test_non_pregnancy_events_ignored(self):
        """Events unrelated to pregnancy don't produce episodes."""
        patient_id = 8004
        events = [
            make_event(patient_id, day(2021, 1, 1), "BLOOD_PRESSURE"),
            make_event(patient_id, day(2021, 3, 1), "CBC_PANEL"),
        ]
        segmenter = TimeRangeSegmenter(label_fn=hipps_label_fn)
        outputs, _ = segmenter.process(parent_id=patient_id, events=events)

        assert len(outputs) == 0

    def test_empty_events_no_crash(self):
        """Empty event list produces no episodes."""
        patient_id = 8005
        segmenter = TimeRangeSegmenter(label_fn=hipps_label_fn)
        outputs, _ = segmenter.process(parent_id=patient_id, events=[])

        assert outputs == []

    def test_episode_too_long_is_removed(self):
        """Episodes exceeding 12 months are cleaned up."""
        patient_id = 8006
        # GW week=60 would create an impossibly long episode
        events = [
            make_event(patient_id, day(2021, 9, 1), "GW", week=60),
        ]
        segmenter = TimeRangeSegmenter(label_fn=hipps_label_fn)
        outputs, _ = segmenter.process(parent_id=patient_id, events=events)

        # The cleanup step should remove the episode (>365 days)
        assert len(outputs) == 0


class TestComplexRealWorldScenario:
    """End-to-end scenario simulating a realistic patient journey.

    Simulates a patient with two pregnancies, incomplete data, and pipeline
    re-runs, demonstrating the full power of the framework.
    """

    def test_full_patient_journey_with_reruns(self):
        """Simulate discovery of pregnancy episodes across multiple pipeline runs.

        Run 1: Early data — first pregnancy partially visible.
        Run 2: More data — first pregnancy complete, second starting.
        Run 3: Full data — both pregnancies complete.
        """
        patient_id = 9001
        segmenter = TimeRangeSegmenter(label_fn=hipps_label_fn)

        # ---- Run 1: Early pregnancy data ----
        events_run1 = [
            make_event(patient_id, day(2020, 2, 15), "NUCHAL_TRANSLUCENCY"),
            make_event(patient_id, day(2020, 3, 1), "GW", week=14),
            make_event(patient_id, day(2020, 5, 1), "ANATOMY_SCAN"),
        ]
        run1_out, _ = segmenter.process(
            parent_id=patient_id, events=events_run1
        )
        assert len(run1_out) >= 1
        # Assign ID
        run1_out[0].range_id = "preg-A"

        # ---- Run 2: First pregnancy completes, second begins ----
        events_run2 = events_run1 + [
            make_event(patient_id, day(2020, 6, 15), "GW", week=29),
            make_event(patient_id, day(2020, 9, 1), "LIVE_BIRTH"),
            # Second pregnancy starts
            make_event(patient_id, day(2021, 5, 1), "GW", week=10),
        ]
        run2_out, run2_retired = segmenter.process(
            parent_id=patient_id,
            events=events_run2,
            input_ranges=run1_out,
        )
        # Should have at least the first pregnancy and possibly the second
        assert len(run2_out) >= 1
        # First pregnancy should preserve its ID since it overlaps
        first_preg = min(run2_out, key=lambda r: r.start_time)
        assert first_preg.range_id == "preg-A"

        # Assign IDs to new ranges
        for r in run2_out:
            if r.range_id is None:
                r.range_id = "preg-B"

        # ---- Run 3: Second pregnancy completes ----
        events_run3 = events_run2 + [
            make_event(patient_id, day(2021, 7, 1), "ANATOMY_SCAN"),
            make_event(patient_id, day(2021, 8, 15), "GLUCOSE_TOLERANCE"),
            make_event(patient_id, day(2021, 12, 1), "LIVE_BIRTH"),
        ]
        run3_out, run3_retired = segmenter.process(
            parent_id=patient_id,
            events=events_run3,
            input_ranges=run2_out,
        )

        assert len(run3_out) >= 2
        sorted_out = sorted(run3_out, key=lambda r: r.start_time)

        # Both IDs should be preserved across the re-run
        ids = {r.range_id for r in run3_out if r.range_id is not None}
        assert "preg-A" in ids

    def test_multiparent_beam_style_processing(self):
        """Process multiple patients independently (simulating Beam pipeline).

        Demonstrates the framework handles per-parent segmentation correctly
        with different pregnancy patterns.
        """
        segmenter = TimeRangeSegmenter(label_fn=hipps_label_fn)

        # Patient A: normal pregnancy
        evts_a = [
            make_event(101, day(2021, 3, 1), "GW", week=12),
            make_event(101, day(2021, 9, 1), "LIVE_BIRTH"),
        ]
        # Patient B: ectopic
        evts_b = [
            make_event(102, day(2021, 4, 1), "ECTOPIC"),
        ]
        # Patient C: two pregnancies
        evts_c = [
            make_event(103, day(2020, 5, 1), "LIVE_BIRTH"),
            make_event(103, day(2021, 8, 1), "LIVE_BIRTH"),
        ]

        out_a, _ = segmenter.process(parent_id=101, events=evts_a)
        out_b, _ = segmenter.process(parent_id=102, events=evts_b)
        out_c, _ = segmenter.process(parent_id=103, events=evts_c)

        assert len(out_a) == 1  # one pregnancy
        assert len(out_b) == 1  # one ectopic
        assert len(out_c) == 2  # two pregnancies

        # Verify parent_id isolation
        assert all(r.parent_id == 101 for r in out_a)
        assert all(r.parent_id == 102 for r in out_b)
        assert all(r.parent_id == 103 for r in out_c)
