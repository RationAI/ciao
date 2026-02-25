import logging

import numpy as np
import torch

from ciao.structures.bitmask_graph import (
    add_node,
    get_frontier,
    iter_bits,
    mask_to_ids,
)
from ciao.utils.calculations import ModelPredictor, calculate_hyperpixel_deltas


logger = logging.getLogger(__name__)

"""Greedy lookahead hyperpixel building with bitmask operations.

Rolling horizon strategy: Look ahead multiple steps but only commit one step at a time.
"""


def build_hyperpixel_greedy_lookahead(
    predictor: ModelPredictor,
    input_batch: torch.Tensor,
    segments: np.ndarray,
    adj_masks: tuple[int, ...],
    target_class_idx: int,
    scores: dict[int, float],
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
        adj_masks: Adjacency bitmasks
        target_class_idx: Target class
        scores: Segment scores (for determining sign)
        seed_idx: Starting segment
        desired_length: Target hyperpixel size
        lookahead_distance: How many steps to look ahead (1=greedy, 2+=lookahead)
        optimization_sign: +1 to maximize, -1 to minimize
        used_mask: Globally excluded segments
        batch_size: Batch size for evaluation

    Returns:
        Dict with segments, sign, scores, final mask, and stats
    """
    current_mask = add_node(0, seed_idx)
    path = [seed_idx]  # Track the path for prefix evaluation
    total_evaluations = 0  # Track total number of evaluations
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
            max_total_size=desired_length,
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
            f"Step {num_steps}: Best score={best_score:.4f}, adding segment {first_step}"
        )

        # Commit only the first step
        current_mask = add_node(current_mask, first_step)
        path.append(first_step)

    # Evaluate all prefixes and find the best one
    logger.debug(f"Evaluating {len(path)} prefixes to find best subset")
    num_prefix_evaluations = len(path)
    total_evaluations += num_prefix_evaluations

    best_prefix_mask, best_score = _evaluate_prefixes(
        path=path,
        predictor=predictor,
        input_batch=input_batch,
        segments=segments,
        target_class_idx=target_class_idx,
        optimization_sign=optimization_sign,
        batch_size=batch_size,
    )

    best_segments = mask_to_ids(best_prefix_mask)
    logger.info(
        f"Best prefix has {len(best_segments)} segments with score={best_score:.4f}"
    )

    return {
        "mask": best_prefix_mask,
        "segments": best_segments,
        "sign": optimization_sign,
        "score": best_score,
        "size": len(best_segments),
        "stats": {
            "method": "lookahead",
            "lookahead_distance": lookahead_distance,
            "num_steps": num_steps,
            "total_evaluations": total_evaluations,
            "prefix_evaluations": num_prefix_evaluations,
        },
    }


def _generate_lookahead_candidates(
    current_mask: int,
    adj_masks: tuple[int, ...],
    used_mask: int,
    lookahead_distance: int,
    max_total_size: int,
) -> dict[int, int]:
    """Generate all connected supersets up to lookahead_distance steps via BFS.

    Returns:
        Dict mapping candidate_mask -> first_step_segment
    """
    candidates: dict[int, int] = {}  # mask -> first_step

    # BFS: Track masks at each depth
    current_depth_masks = {current_mask}

    for depth in range(1, lookahead_distance + 1):
        next_depth_masks = set()

        for mask in current_depth_masks:
            if mask.bit_count() >= max_total_size:
                continue

            frontier = get_frontier(mask, adj_masks, used_mask)
            if frontier == 0:
                continue

            # Expand to all frontier neighbors
            for seg_id in iter_bits(frontier):
                new_mask = add_node(mask, seg_id)

                # Determine first_step for this candidate
                if depth == 1:
                    first_step = seg_id
                else:
                    # Inherit first_step from parent mask
                    # Find which first_step led to this mask
                    # We need to track this through the BFS
                    # Since we're at depth > 1, the parent mask should already be in candidates
                    if mask in candidates:
                        first_step = candidates[mask]
                    else:
                        # This shouldn't happen in proper BFS - raise error if it does
                        raise RuntimeError(
                            f"BFS inconsistency: mask {mask} not found in candidates at depth {depth}. "
                            "This indicates a logic error in the BFS traversal."
                        )

                # Only add if not already seen (first path wins)
                if new_mask not in candidates:
                    candidates[new_mask] = first_step
                    next_depth_masks.add(new_mask)

        current_depth_masks = next_depth_masks
        if not current_depth_masks:
            break

    return candidates


def _find_first_step(base_mask: int, target_mask: int) -> int:
    """Find the first segment added from base_mask to reach target_mask."""
    diff = target_mask & ~base_mask
    # Return the first bit in the difference
    for seg_id in iter_bits(diff):
        return seg_id
    raise ValueError("Could not find first step between base and target mask.")


def _evaluate_prefixes(
    path: list[int],
    predictor: ModelPredictor,
    input_batch: torch.Tensor,
    segments: np.ndarray,
    target_class_idx: int,
    optimization_sign: int,
    batch_size: int,
) -> tuple[int, float]:
    """Evaluate all prefixes of the path and return the best one.

    Returns:
        (best_mask, best_score)
    """
    # Generate all prefix masks
    prefix_masks = []
    mask = 0
    for seg_id in path:
        mask = add_node(mask, seg_id)
        prefix_masks.append(mask)

    # Batch evaluate
    segment_id_lists = [mask_to_ids(mask) for mask in prefix_masks]
    scores = calculate_hyperpixel_deltas(
        predictor=predictor,
        input_batch=input_batch,
        segments=segments,
        hyperpixel_segment_ids_list=segment_id_lists,
        target_class_idx=target_class_idx,
        batch_size=batch_size,
    )

    # Find best prefix
    best_idx = max(range(len(scores)), key=lambda i: scores[i] * optimization_sign)

    return prefix_masks[best_idx], scores[best_idx]


def build_all_hyperpixels_greedy_lookahead(
    predictor: ModelPredictor,
    input_batch: torch.Tensor,
    segments: np.ndarray,
    adj_masks: tuple[int, ...],
    target_class_idx: int,
    scores: dict[int, float],
    max_hyperpixels: int,
    desired_length: int,
    lookahead_distance: int,
    batch_size: int = 64,
) -> list[dict[str, object]]:
    """Build multiple hyperpixels using greedy lookahead.

    Returns:
        List of hyperpixel dicts sorted by absolute score
    """
    hyperpixels = []
    used_mask = 0

    for i in range(max_hyperpixels):
        # Find best unprocessed seed
        available_segments = [
            seg_id for seg_id in scores if not (used_mask & (1 << seg_id))
        ]

        if not available_segments:
            break

        seed_idx = max(available_segments, key=lambda x: abs(scores[x]))
        seed_score = scores[seed_idx]
        optimization_sign = 1 if seed_score >= 0 else -1

        logger.info(f"\n--- Hyperpixel {i + 1}/{max_hyperpixels} ---")
        logger.info(
            f"Seed: {seed_idx}, score: {seed_score:.4f}, sign: {optimization_sign}"
        )

        result = build_hyperpixel_greedy_lookahead(
            predictor=predictor,
            input_batch=input_batch,
            segments=segments,
            adj_masks=adj_masks,
            target_class_idx=target_class_idx,
            scores=scores,
            seed_idx=seed_idx,
            desired_length=desired_length,
            lookahead_distance=lookahead_distance,
            optimization_sign=optimization_sign,
            used_mask=used_mask,
            batch_size=batch_size,
        )

        # Update used_mask
        used_mask = result["mask"] | used_mask  # type: ignore[operator]

        # Format for compatibility
        hyperpixel = {
            "segments": result["segments"],
            "sign": result["sign"],
            "size": result["size"],
            "hyperpixel_score": result["score"],
            "stats": result.get("stats", {}),  # Include lookahead statistics
        }
        hyperpixels.append(hyperpixel)

        logger.info(
            f"Built hyperpixel with {len(result['segments'])} segments, score={result['score']:.4f}"  # type: ignore[arg-type]
        )

    # Sort by absolute score
    hyperpixels.sort(key=lambda x: abs(x["hyperpixel_score"]), reverse=True)  # type: ignore[arg-type]
    return hyperpixels
