"""Evaluation and scoring utilities for segments and regions."""

from ciao.scoring.region import (
    RegionResult,
    calculate_region_deltas,
    calculate_region_probability_drops,
    select_top_regions,
)
from ciao.scoring.segments import (
    calculate_segment_scores,
    create_surrogate_dataset,
)


__all__ = [
    "RegionResult",
    "calculate_region_deltas",
    "calculate_region_probability_drops",
    "calculate_segment_scores",
    "create_surrogate_dataset",
    "select_top_regions",
]
