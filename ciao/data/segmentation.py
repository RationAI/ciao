import math
from typing import Literal

import numpy as np
import torch


def _hex_round_vectorized(
    q: np.ndarray, r: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized hex rounding for entire arrays of axial coordinates.

    Args:
        q: Array of fractional axial q coordinates
        r: Array of fractional axial r coordinates

    Returns:
        (q_rounded, r_rounded): Integer axial coordinates of nearest hex
    """
    x = q
    z = r
    y = -x - z

    rx = np.round(x)
    ry = np.round(y)
    rz = np.round(z)

    dx = np.abs(rx - x)
    dy = np.abs(ry - y)
    dz = np.abs(rz - z)

    # Vectorized conditional logic
    cond1 = (dx > dy) & (dx > dz)
    cond2 = dy > dz

    rx = np.where(cond1, -ry - rz, rx)
    ry = np.where(~cond1 & cond2, -rx - rz, ry)
    rz = np.where(~cond1 & ~cond2, -rx - ry, rz)

    return rx.astype(np.int32), rz.astype(np.int32)


def _build_square_adjacency_list(
    segments: np.ndarray, neighborhood: int = 8
) -> tuple[tuple[int, ...], ...]:
    """Build adjacency list from segment array (for square grids) using vectorized operations.

    Args:
        segments: 2D array mapping pixels to segment IDs
        neighborhood: 4 or 8 connectivity

    Returns:
        Adjacency list as tuple of tuples
    """
    num_segments = segments.max() + 1
    adjacency_sets: list[set[int]] = [set() for _ in range(num_segments)]

    # Vectorized horizontal adjacency
    left = segments[:, :-1].ravel()
    right = segments[:, 1:].ravel()
    mask_h = left != right
    edges_h = np.column_stack([left[mask_h], right[mask_h]])

    # Vectorized vertical adjacency
    top = segments[:-1, :].ravel()
    bottom = segments[1:, :].ravel()
    mask_v = top != bottom
    edges_v = np.column_stack([top[mask_v], bottom[mask_v]])

    # Collect all edges
    edge_arrays = [edges_h, edges_v]

    if neighborhood == 8:
        # Vectorized diagonal adjacency (down-right)
        top_left = segments[:-1, :-1].ravel()
        bottom_right = segments[1:, 1:].ravel()
        mask_dr = top_left != bottom_right
        edges_dr = np.column_stack([top_left[mask_dr], bottom_right[mask_dr]])

        # Vectorized diagonal adjacency (down-left)
        top_right = segments[:-1, 1:].ravel()
        bottom_left = segments[1:, :-1].ravel()
        mask_dl = top_right != bottom_left
        edges_dl = np.column_stack([top_right[mask_dl], bottom_left[mask_dl]])

        edge_arrays.extend([edges_dr, edges_dl])

    # Stack all edges together and populate adjacency sets in a single loop
    all_edges = np.vstack(edge_arrays)
    for seg1, seg2 in all_edges:
        adjacency_sets[seg1].add(seg2)
        adjacency_sets[seg2].add(seg1)

    # Convert to tuple of tuples
    return tuple(tuple(sorted(neighbors)) for neighbors in adjacency_sets)


def _build_fast_adjacency_list(
    hex_to_id: dict[tuple[int, int], int], max_id: int
) -> tuple[tuple[int, ...], ...]:
    """Create a static adjacency list optimized for fast reading.

    Args:
        hex_to_id: Dict mapping (q, r) -> int_id (0 to N-1)
        max_id: Total number of segments (N)

    Returns:
        adj_list: Tuple of Tuples.
                  adj_list[5] returns e.g. (4, 6, 12) - neighbors of segment 5.
    """
    # Initialize empty lists for each ID
    # Use list of lists for construction
    temp_adj: list[list[int]] = [[] for _ in range(max_id)]

    # Offsets for neighbors (axial coords)
    hex_neighbors = [(+1, 0), (+1, -1), (0, -1), (-1, 0), (-1, +1), (0, +1)]

    for (q, r), seg_id in hex_to_id.items():
        for dq, dr in hex_neighbors:
            neighbor_key = (q + dq, r + dr)

            # If neighbor exists (is within the image)
            if neighbor_key in hex_to_id:
                neighbor_id = hex_to_id[neighbor_key]
                temp_adj[seg_id].append(neighbor_id)

    # Convert to tuple of tuples for maximum read speed and memory efficiency
    # Sort neighbors (optional, but good for determinism)
    final_adj = tuple(tuple(sorted(neighbors)) for neighbors in temp_adj)

    return final_adj


def _build_adjacency_bitmasks(adj_list: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    """Convert adjacency list to a list of integers.

    adj_masks[5] will be an integer with bits set at positions of hex 5's neighbors.
    """
    adj_masks = []
    for neighbors in adj_list:
        mask = 0
        for n in neighbors:
            mask |= 1 << n
        adj_masks.append(mask)
    return tuple(adj_masks)


def _create_square_grid(
    input_tensor: torch.Tensor, square_size: int = 14, neighborhood: int = 8
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Create a grid of squares with adjacency list representing spatial relationships."""
    _channels, height, width = input_tensor.shape
    segments = np.zeros((height, width), dtype=np.int32)

    segment_id = 0

    # Create square grid
    for row in range(0, height, square_size):
        for col in range(0, width, square_size):
            # Define square boundaries
            row_end = min(row + square_size, height)
            col_end = min(col + square_size, width)

            # Assign segment ID to all pixels in this square
            segments[row:row_end, col:col_end] = segment_id
            segment_id += 1

    # Build adjacency list
    adjacency_list = _build_square_adjacency_list(segments, neighborhood=neighborhood)

    return segments, adjacency_list


def _create_hexagonal_grid(
    input_tensor: torch.Tensor, hex_radius: int = 14
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Create a grid of hexagons with adjacency list using vectorized operations.

    Uses axial coordinate system for precise hexagonal tiling (flat-top orientation).
    Each hexagon has exactly 6 neighbors (neighborhood parameter ignored).

    Args:
        input_tensor: Input image tensor [C, H, W]
        hex_radius: Hex size parameter (distance from center to flat edge, default: 14)

    Returns:
        segments: 2D array mapping pixels to segment IDs
        adjacency_list: Tuple of tuples representing segment relationships
    """
    _channels, height, width = input_tensor.shape

    # Create coordinate grids using meshgrid
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

    # Vectorized pixel to hex conversion
    sqrt3 = math.sqrt(3)
    q_float = (sqrt3 / 3 * x_coords - 1 / 3 * y_coords) / hex_radius
    r_float = (2 / 3 * y_coords) / hex_radius

    # Vectorized hex rounding
    q_int, r_int = _hex_round_vectorized(q_float, r_float)

    # Stack q and r to create unique keys
    qr_stacked = np.stack([q_int.ravel(), r_int.ravel()], axis=1)

    # Use np.unique to assign segment IDs efficiently (compute only once)
    unique_qr, segments_flat = np.unique(qr_stacked, axis=0, return_inverse=True)
    segments = segments_flat.reshape((height, width)).astype(np.int32)

    # Build hex_to_id mapping for adjacency construction
    hex_to_id = {(int(q), int(r)): idx for idx, (q, r) in enumerate(unique_qr)}

    # Build adjacency list using axial coordinate neighbors
    adjacency_list = _build_fast_adjacency_list(hex_to_id, len(hex_to_id))

    return segments, adjacency_list


def create_segmentation(
    input_tensor: torch.Tensor,
    segmentation_type: Literal["square", "hexagonal"] = "hexagonal",
    segment_size: int = 14,
    neighborhood: int = 8,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Create image segmentation with specified type.

    Args:
        input_tensor: Input image tensor [C, H, W]
        segmentation_type: "square" or "hexagonal"
        segment_size: Size parameter (square_size or hex_radius)
        neighborhood: Neighborhood connectivity for squares (4, or 8)

    Returns:
        segments: 2D array mapping pixels to segment IDs
        adj_masks: Tuple of integer bitmasks representing adjacency relationships
    """
    if segment_size <= 0:
        raise ValueError(
            f"segment_size must be positive, got {segment_size}. "
            "Non-positive values cause division by zero or invalid range operations."
        )

    if segmentation_type not in ("square", "hexagonal"):
        raise ValueError(
            f"Unknown segmentation_type: {segmentation_type}. Use 'square' or 'hexagonal'."
        )

    if segmentation_type == "square" and neighborhood not in (4, 8):
        raise ValueError(
            f"For square segmentation, neighborhood must be 4 or 8, got {neighborhood}."
        )

    if segmentation_type == "square":
        segments, adjacency_list = _create_square_grid(
            input_tensor, square_size=segment_size, neighborhood=neighborhood
        )
    else:
        segments, adjacency_list = _create_hexagonal_grid(
            input_tensor, hex_radius=segment_size
        )

    # Convert adjacency list to bitmasks
    adj_masks = _build_adjacency_bitmasks(adjacency_list)
    return segments, adj_masks
