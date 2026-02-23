import time

from ciao.structures.bitmask_graph import (
    add_node,
    get_frontier,
    iter_bits,
    mask_to_ids,
    remove_node,
    sample_connected_superset,
)
from ciao.utils.calculations import calculate_hyperpixel_deltas


def compute_potentials(
    cache: dict[int, list[tuple[int, float]]],
) -> dict[int, list[float]]:
    """Compute potential vectors for each neighbor.

    NOTE: This uses a lexicographical dominance rule based on sorted scores.
    This is a heuristic choice specific to this project's requirements.
    Longer histories may have a statistical advantage.
    """
    potentials = {}
    for node_id, history in cache.items():
        scores = [score for _, score in history]
        # Lexicographical sort (primary requirement)
        scores.sort(reverse=True)
        potentials[node_id] = scores
    return potentials


def select_best_neighbor(potentials: dict[int, list[float]]) -> int:
    """Select neighbor with highest potential using lexicographical comparison."""
    best_neighbor = -1
    best_vector = []

    for node_id, scores in potentials.items():
        if not scores:
            continue

        if scores > best_vector:
            best_vector = scores
            best_neighbor = node_id

    return best_neighbor


def build_hyperpixel_using_potential(
    predictor,
    input_batch,
    segments,
    adj_list: tuple[tuple[int, ...], ...],
    adj_masks: tuple[int, ...],
    target_class_idx: int,
    desired_length: int,
    seed_idx: int,
    num_simulations: int,
    used_segments: set | None = None,
    batch_size: int = 64,
    optimization_sign: int = 1,
):
    """Build a hyperpixel using Sequential Monte Carlo with Potential-based Selection.

    This algorithm grows a connected region on the segmentation graph by:
    1. Computing the expansion frontier (valid neighbors of current structure)
    2. For each frontier node, running Monte Carlo simulations of random expansions
    3. Selecting the frontier node with the best potential (lexicographically highest scores)
    4. Pruning: Discarding all histories except the winner's ("wavefunction collapse")
    5. Repeating until target length is reached

    The result is a connected component optimized for the target class prediction.

    Args:
        predictor: Model predictor for scoring hyperpixels
        input_batch: Preprocessed input image tensor [1, C, H, W]
        segments: Segmentation map [H, W] (pixel -> segment ID)
        adj_list: Adjacency list (tuple of tuples, legacy parameter for compatibility)
        adj_masks: Adjacency bitmasks (adj_masks[i] = neighbors of segment i)
        target_class_idx: Class to optimize for
        desired_length: Maximum hyperpixel size before prefix optimization
        seed_idx: Starting segment (typically highest scoring unprocessed segment)
        num_simulations: Monte Carlo samples per frontier node per iteration
        used_segments: Globally excluded segments (e.g., from other hyperpixels)
        batch_size: Batch size for model inference during evaluation
        optimization_sign: +1 to maximize score, -1 to minimize

    Returns:
        Dict with segments, mask, size, and statistics
    """
    # Initialize optional set
    if used_segments is None:
        used_segments = set()

    # Strict Type Separation: Convert set to mask ONCE at entry
    used_mask = 0
    for seg_id in used_segments:
        used_mask = add_node(used_mask, seg_id)

    # Initialize current hyperpixel structure
    hyperpixel_structure = [seed_idx]
    structure_mask = add_node(0, seed_idx)

    # Potential cache: Maps frontier_node_id -> [(sample_mask, score), ...]
    # This stores the Monte Carlo history for each frontier node
    potential_cache: dict[int, list[tuple[int, float]]] = {}

    # Statistics tracking
    total_steps = 0
    total_evaluations = 0
    total_samples = 0

    print("\n=== Sequential Monte Carlo Set Extension ===")
    print(f"Seed: {seed_idx}, Target: {desired_length}, Sims: {num_simulations}")

    step = 0

    # Main loop: Grow hyperpixel until target length
    while len(hyperpixel_structure) < desired_length:
        step += 1
        total_steps += 1
        print(
            f"\n--- Step {step}: |S| = {len(hyperpixel_structure)}/{desired_length} ---"
        )

        # Compute expansion frontier
        current_frontier_mask = get_frontier(structure_mask, adj_masks, used_mask)

        if not current_frontier_mask:
            print("Frontier empty. Stopping.")
            break

        frontier_list = mask_to_ids(current_frontier_mask)
        print(f"Frontier size: {len(frontier_list)}")

        # Phase 1: Sampling - Monte Carlo exploration from each frontier node
        step_start = time.time()
        num_evals, num_samps = sampling_phase(
            S_mask=structure_mask,
            neighbors=frontier_list,
            current_frontier_mask=current_frontier_mask,
            num_simulations=num_simulations,
            desired_length=desired_length,
            adj_masks=adj_masks,
            predictor=predictor,
            input_batch=input_batch,
            segments=segments,
            target_class_idx=target_class_idx,
            batch_size=batch_size,
            optimization_sign=optimization_sign,
            cache=potential_cache,
            used_mask=used_mask,
        )
        total_evaluations += num_evals
        total_samples += num_samps
        sampling_time = time.time() - step_start

        # Phase 2: Selection - Choose best frontier node by potential
        potentials = compute_potentials(potential_cache)
        winner = select_best_neighbor(potentials)

        if winner == -1:
            print("No valid winner found. Stopping.")
            break

        winner_stats = potentials[winner]
        max_potential = max(winner_stats) if winner_stats else 0
        print(
            f"Winner: {winner} (samples: {len(winner_stats)}, max: {max_potential:.4f})"
        )
        print(f"Timing: {sampling_time:.2f}s")

        # Commit: Add winner to hyperpixel structure
        hyperpixel_structure.append(winner)
        structure_mask = add_node(structure_mask, winner)

        # Phase 3: Pruning - Wavefunction collapse
        # Discard all histories except the winner's, then redistribute
        winner_history = potential_cache[winner]
        potential_cache = {}  # Aggressive pruning: discard all non-winner histories

        # Recompute frontier for the updated structure
        new_frontier_mask = get_frontier(structure_mask, adj_masks, used_mask)
        redistribute_history(winner_history, new_frontier_mask, potential_cache)

        recipient_count = len(potential_cache)
        print(
            f"Pruning: Kept {len(winner_history)} samples, redistributed to {recipient_count} neighbors"
        )

    print("\n=== Extension Complete ===")

    # Post-processing: Find optimal prefix (sometimes full length adds noise)
    final_hyperpixel, hyperpixel_score = select_best_prefix(
        full_structure=hyperpixel_structure,
        predictor=predictor,
        input_batch=input_batch,
        segments=segments,
        target_class_idx=target_class_idx,
        batch_size=batch_size,
        optimization_sign=optimization_sign,
        cache={},
    )

    # Count prefix evaluations
    prefix_evaluations = len(hyperpixel_structure)
    total_evaluations += prefix_evaluations

    # Create result dict with stats
    final_mask = 0
    for seg_id in final_hyperpixel:
        final_mask = add_node(final_mask, seg_id)

    return {
        "segments": final_hyperpixel,
        "mask": final_mask,
        "size": len(final_hyperpixel),
        "score": hyperpixel_score,  # Actual model delta score
        "stats": {
            "method": "potential",
            "num_simulations": num_simulations,
            "num_steps": total_steps,
            "total_evaluations": total_evaluations,
            "total_samples": total_samples,
            "prefix_evaluations": prefix_evaluations,
        },
    }


def sampling_phase(
    S_mask: int,
    neighbors: list[int],
    current_frontier_mask: int,
    num_simulations: int,
    desired_length: int,
    adj_masks: tuple[int, ...],
    predictor,
    input_batch,
    segments,
    target_class_idx: int,
    batch_size: int,
    optimization_sign: int,
    cache: dict[int, list[tuple[int, float]]],
    used_mask: int,
) -> tuple[int, int]:
    """Monte Carlo Sampling Phase: Explore expansions and populate potential cache.

    For each frontier node n:
        1. Create extended structure S ∪ {n}
        2. Run num_simulations random walk expansions from this extended structure
        3. Evaluate each unique expansion with the model
        4. Distribute results to cache: For each frontier node that appears in an
           expansion, record (expansion_mask, score) in that node's history

    This builds a statistical basis for comparing frontier nodes: nodes that
    consistently appear in high-scoring expansions will have better potentials.

    Args:
        S_mask: Current hyperpixel structure (bitmask)
        neighbors: Frontier nodes (list for iteration)
        current_frontier_mask: Same frontier as bitmask (for efficient computation)
        num_simulations: Monte Carlo samples per frontier node
        desired_length: Target expansion size for random walks
        adj_masks: Adjacency bitmasks for graph traversal
        predictor, input_batch, segments, target_class_idx: For model evaluation
        batch_size: Batch size for model inference
        optimization_sign: +1 or -1 for score interpretation
        cache: Potential cache to populate (modified in-place)
        used_mask: Global exclusion mask

    Returns:
        Tuple of (num_evaluations, num_samples)
    """
    evaluation_queue = []
    mask_to_neighbors_mask = {}  # Maps expansion_mask -> which frontier nodes it contains

    all_neighbors_mask = current_frontier_mask
    total_samples = 0

    # --- Sampling Loop: Generate candidate expansions ---
    for n in neighbors:
        # Start with S ∪ {n}
        extended_mask = add_node(S_mask, n)

        # Compute frontier for random walk:
        # - Start with current frontier
        # - Remove n (now part of structure)
        # - Add valid neighbors of n (expand exploration boundary)
        n_neighbors_mask = adj_masks[n]
        valid_n_neighbors = n_neighbors_mask & ~(used_mask | S_mask | add_node(0, n))

        base_frontier = remove_node(current_frontier_mask, n) | valid_n_neighbors

        for _ in range(num_simulations):
            total_samples += 1
            M = sample_connected_superset(
                base_mask=extended_mask,
                target_length=desired_length,
                adj_masks=adj_masks,
                base_frontier=base_frontier,
                used_mask=used_mask,
            )

            if M in mask_to_neighbors_mask:
                continue

            # Bucketization: Which frontier nodes appear in this expansion?
            hits = M & all_neighbors_mask
            evaluation_queue.append(M)
            mask_to_neighbors_mask[M] = hits

    if not evaluation_queue:
        return 0, total_samples

    # --- Batch Evaluation: Score all unique expansions ---
    print(f"  Evaluating {len(evaluation_queue)} unique samples...")
    segment_id_lists = [mask_to_ids(mask) for mask in evaluation_queue]
    scores = calculate_hyperpixel_deltas(
        predictor=predictor,
        input_batch=input_batch,
        segments=segments,
        hyperpixel_segment_ids_list=segment_id_lists,
        target_class_idx=target_class_idx,
        batch_size=batch_size,
    )

    # --- Distribution to Cache ---
    for mask, score in zip(evaluation_queue, scores):
        signed_score = score * optimization_sign
        hits = mask_to_neighbors_mask[mask]

        # Distribute to all neighbors in the hit set
        for neighbor_id in iter_bits(hits):
            if neighbor_id not in cache:
                cache[neighbor_id] = []
            cache[neighbor_id].append((mask, signed_score))

    return len(evaluation_queue), total_samples


def select_best_prefix(
    full_structure: list[int],
    predictor,
    input_batch,
    segments,
    target_class_idx: int,
    batch_size: int,
    optimization_sign: int,
    cache: dict[int, float],
) -> tuple[list[int], float]:
    """Find the optimal prefix of a hyperpixel structure.

    Sometimes the fixed desired_length forces the algorithm to add segments
    that dilute the signal after the main object region is covered. This
    post-processing step evaluates all prefixes [1:k] of the full structure
    and returns the one with the best score.

    Args:
        full_structure: Complete hyperpixel (list of segment IDs in build order)
        predictor, input_batch, segments, target_class_idx: For evaluation
        batch_size: Batch size for model inference
        optimization_sign: +1 to maximize, -1 to minimize
        cache: Simple cache (mask -> score) for reuse

    Returns:
        Tuple of (optimal_prefix, raw_score)
    """
    if not full_structure:
        return [], 0.0

    prefixes = []
    current_mask = 0
    prefix_masks = []

    for seg_id in full_structure:
        current_mask = add_node(current_mask, seg_id)
        prefix_masks.append(current_mask)
        prefixes.append(mask_to_ids(current_mask))

    missing_indices = []
    scores = [None] * len(prefix_masks)

    for i, mask in enumerate(prefix_masks):
        if mask in cache:
            scores[i] = cache[mask]
        else:
            missing_indices.append(i)

    if missing_indices:
        missing_id_lists = [prefixes[i] for i in missing_indices]

        computed_scores = calculate_hyperpixel_deltas(
            predictor=predictor,
            input_batch=input_batch,
            segments=segments,
            hyperpixel_segment_ids_list=missing_id_lists,
            target_class_idx=target_class_idx,
            batch_size=batch_size,
        )

        for i, score in zip(missing_indices, computed_scores):
            signed_score = score * optimization_sign
            scores[i] = signed_score
            cache[prefix_masks[i]] = signed_score

    best_idx = 0
    max_score = -float("inf")

    for i, score in enumerate(scores):
        if score > max_score:
            max_score = score
            best_idx = i

    optimized_length = best_idx + 1
    if optimized_length < len(full_structure):
        print(
            f"Optimization: Trimmed from {len(full_structure)} to {optimized_length} segments (Score: {max_score:.4f})"
        )
    else:
        print(
            f"Optimization: Kept full length {len(full_structure)} segments (Score: {max_score:.4f})"
        )

    # Return both the prefix and its score (convert signed score back to raw score)
    raw_score = max_score * optimization_sign
    return full_structure[: best_idx + 1], raw_score


def redistribute_history(
    H_winner: list[tuple[int, float]],
    new_frontier_mask: int,
    cache: dict[int, list[tuple[int, float]]],
):
    """Redistribute winner's Monte Carlo history to the new frontier.

    After adding the winning node to the structure, the frontier changes.
    The winner's history contains expansions that may include the NEW frontier
    nodes. This function distributes those historical samples to the appropriate
    frontier nodes' potential caches.

    This inheritance mechanism allows the algorithm to "learn" from past
    explorations: if an expansion that included the winner also included
    a new frontier node, that information is valuable for evaluating that
    frontier node.

    Args:
        H_winner: Winner's history [(expansion_mask, score), ...]
        new_frontier_mask: Frontier nodes after adding winner to structure
        cache: Potential cache to populate (modified in-place)
    """
    # Iterate through winner's historical expansions
    for M, v in H_winner:
        # Which new frontier nodes are in this historical expansion?
        hits = M & new_frontier_mask

        if not hits:
            continue

        # Distribute this historical (expansion, score) pair to all hit nodes
        for neighbor_id in iter_bits(hits):
            if neighbor_id not in cache:
                cache[neighbor_id] = []
            cache[neighbor_id].append((M, v))


def build_all_hyperpixels_potential(
    predictor,
    input_batch,
    segments,
    adj_list,
    adj_masks,
    target_class_idx,
    scores,
    max_hyperpixels=10,
    desired_length=30,
    num_simulations=50,
    batch_size=64,
):
    """Build multiple hyperpixels using the potential field method.

    Args:
        predictor: Model predictor
        input_batch: Preprocessed input tensor
        segments: Segmentation map
        adj_list: Adjacency list
        adj_masks: Adjacency bitmasks
        target_class_idx: Target class index
        scores: Individual segment scores
        max_hyperpixels: Maximum number of hyperpixels to build
        desired_length: Target segments per hyperpixel
        num_simulations: Number of MC simulations per frontier node
        batch_size: Batch size for evaluation

    Returns:
        List of hyperpixel dictionaries
    """
    hyperpixels = []
    processed_segments = set()

    for i in range(max_hyperpixels):
        # Find unprocessed segment with highest absolute score
        available_segments = [
            seg_id for seg_id in scores.keys() if seg_id not in processed_segments
        ]

        if not available_segments:
            break

        seed_idx = max(available_segments, key=lambda x: abs(scores[x]))
        seed_score = scores[seed_idx]
        optimization_sign = 1 if seed_score >= 0 else -1

        # Build hyperpixel using potential field
        result = build_hyperpixel_using_potential(
            predictor,
            input_batch,
            segments,
            adj_list,
            adj_masks,
            target_class_idx,
            desired_length,
            seed_idx,
            num_simulations,
            used_segments=processed_segments,
            batch_size=batch_size,
            optimization_sign=optimization_sign,
        )

        hyperpixel_segments = result["segments"]

        if hyperpixel_segments:
            hyperpixels.append(
                {
                    "segments": hyperpixel_segments,
                    "sign": optimization_sign,
                    "size": len(hyperpixel_segments),
                    "hyperpixel_score": result["score"],
                    "stats": result.get(
                        "stats", {}
                    ),  # Include potential method statistics
                }
            )
            processed_segments.update(hyperpixel_segments)

    return hyperpixels
