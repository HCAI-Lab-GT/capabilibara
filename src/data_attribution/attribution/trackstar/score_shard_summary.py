"""Per-shard summaries for a fused TrackStar score artifact."""

from __future__ import annotations

import math
from bisect import bisect_right

from data_attribution.attribution.trackstar.score_artifact_types import (
    ScoreShardSummary,
)


def create_summary_states(roster):
    states, ends, offset = [], [], 0
    for shard_id, row_count in roster:
        states.append([shard_id, row_count, offset, math.inf, -math.inf, 0])
        offset += row_count
        ends.append(offset)
    return states, ends


def update_summary(states, ends, position, mean, median) -> None:
    state = states[bisect_right(ends, position)]
    state[3] = min(state[3], mean, median)
    state[4] = max(state[4], mean, median)
    state[5] += int(mean != 0.0) + int(median != 0.0)


def finish_summaries(states) -> tuple[ScoreShardSummary, ...]:
    return tuple(
        ScoreShardSummary(item[0], item[1], item[1] * 2, item[5], item[3], item[4])
        for item in states
    )
