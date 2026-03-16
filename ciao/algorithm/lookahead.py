"""Greedy lookahead hyperpixel building with bitmask operations.

Rolling horizon strategy: Look ahead multiple steps but only commit one step at a time.
"""

import logging
from collections import deque

import numpy as np
import torch

from ciao.algorithm.bitmask_graph import (
    add_node,
    get_frontier,
    iter_bits,
    mask_to_ids,
)
from ciao.model.predictor import ModelPredictor
from ciao.scoring.hyperpixel import calculate_hyperpixel_deltas


logger = logging.getLogger(__name__)


def build_hyperpixel_greedy_lookahead(
    predictor: ModelPredictor,
    input_batch: torch.Tensor,
    segments: np.ndarray,
    replacement_image: torch.Tensor,
    adj_masks: tuple[int, ...],
    target_class_idx: int,
    seed_idx: int,
    desired_length: int,
    lookahead_distance: int,
    optimization_sign: int,
    used_mask: int,
    batch_size: int = 64,
) -> dict[str, object]:
    """Build a single hyperpixel using greedy lookahead with rolling horizon.

    Strategy: Look ahead up to lookahead_distance steps, evaluate all candidates,
    but only commit the first step of the best path found.

    Args:
        predictor: Model predictor
        input_batch: Preprocessed image
        segments: Segmentation map
        replacement_image: Replacement tensor [C, H, W]
        adj_masks: Adjacency bitmasks
        target_class_idx: Target class
        seed_idx: Starting segment
        desired_length: Target hyperpixel size
        lookahead_distance: How many steps to look ahead (1=greedy, 2+=lookahead)
        optimization_sign: +1 to maximize, -1 to minimize
        used_mask: Globally excluded segments
        batch_size: Batch size for evaluation

    Returns:
        Dict with segments, sign, score, final mask, and stats
    """
    current_mask = add_node(0, seed_idx)
    total_evaluations = 0
    num_steps = 0

    logger.info(f"Starting greedy lookahead from seed {seed_idx}")

    # Grow hyperpixel one step at a time
    while current_mask.bit_count() < desired_length:
        num_steps += 1
        current_size = current_mask.bit_count()

        # Generate all candidate masks via BFS up to lookahead_distance
        candidates = _generate_lookahead_candidates(
            current_mask=current_mask,
            adj_masks=adj_masks,
            used_mask=used_mask,
            lookahead_distance=lookahead_distance,
            desired_length=desired_length,
        )

        if not candidates:
            logger.info(
                f"Step {num_steps}: No candidates available, stopping at size {current_size}"
            )
            break

        logger.debug(
            f"Step {num_steps}: Size={current_size}/{desired_length}, evaluating {len(candidates)} candidates"
        )

        # Batch evaluate all candidates
        candidate_masks = list(candidates.keys())
        segment_id_lists = [mask_to_ids(mask) for mask in candidate_masks]
        total_evaluations += len(candidate_masks)

        scores_list = calculate_hyperpixel_deltas(
            predictor=predictor,
            input_batch=input_batch,
            segments=segments,
            hyperpixel_segment_ids_list=segment_id_lists,
            replacement_image=replacement_image,
            target_class_idx=target_class_idx,
            batch_size=batch_size,
        )

        # Find best candidate (maximize optimization_sign * score)
        best_idx = max(
            range(len(scores_list)), key=lambda i: scores_list[i] * optimization_sign
        )
        best_mask = candidate_masks[best_idx]
        best_score = scores_list[best_idx]
        first_step = candidates[best_mask]

        logger.debug(
            f"Step {num_steps}: Best lookahead candidate score={best_score:.4f}, adding segment {first_step}"
        )

        # Commit only the first step
        # (it is an open question whether we should add only the first step or the entire best_mask)
        current_mask = add_node(current_mask, first_step)

    # Re-evaluate the final built mask to get its exact score
    # (since the last loop might have evaluated a lookahead candidate, not the exact current_mask)
    final_segments = mask_to_ids(current_mask)
    final_score = calculate_hyperpixel_deltas(
        predictor=predictor,
        input_batch=input_batch,
        segments=segments,
        hyperpixel_segment_ids_list=[final_segments],
        replacement_image=replacement_image,
        target_class_idx=target_class_idx,
        batch_size=1,
    )[0]
    total_evaluations += 1

    logger.info(
        f"Built hyperpixel with {len(final_segments)} segments, final exact score={final_score:.4f}"
    )

    return {
        "mask": current_mask,
        "segments": final_segments,
        "sign": optimization_sign,
        "score": final_score,
        "size": len(final_segments),
        "stats": {
            "total_evaluations": total_evaluations,
        },
    }


def _generate_lookahead_candidates(
    current_mask: int,
    adj_masks: tuple[int, ...],
    used_mask: int,
    lookahead_distance: int,
    desired_length: int,
) -> dict[int, int]:
    """Generate all connected supersets up to lookahead_distance steps via BFS.

    Returns:
        Dict mapping candidate_mask -> first_step_segment_id
    """
    candidates: dict[int, int] = {}  # mask -> first_step

    # Queue stores tuples of: (current_mask, first_step_that_led_here, current_depth)
    queue: deque[tuple[int, int | None, int]] = deque([(current_mask, None, 0)])
    visited = {current_mask}

    while queue:
        mask, first_step, depth = queue.popleft()

        # Store valid candidates (depth > 0)
        if depth > 0 and first_step is not None and mask not in candidates:
            # Only add if not already seen (shortest path wins)
            candidates[mask] = first_step

        # Stop expanding if we reached the lookahead limit or maximum size
        if depth >= lookahead_distance or mask.bit_count() >= desired_length:
            continue

        frontier = get_frontier(mask, adj_masks, used_mask)
        for seg_id in iter_bits(frontier):
            new_mask = add_node(mask, seg_id)

            if new_mask not in visited:
                visited.add(new_mask)
                # If at the first layer (depth 0), this seg_id is our first_step.
                # Otherwise, pass along the first_step inherited from the parent.
                next_first_step = seg_id if depth == 0 else first_step
                queue.append((new_mask, next_first_step, depth + 1))

    return candidates
