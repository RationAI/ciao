import math

import numpy as np
import torch
from skimage.segmentation import slic

from ciao.algorithm.graph import ImageGraph
from ciao.data.constants import IMAGENET_MEAN, IMAGENET_STD
from ciao.typing import SegmentationFn


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


def _build_pixel_adjacency_list(segments: torch.Tensor) -> list[frozenset[int]]:
    """Build segment adjacency list from a 2D pixel label map.

    Works for any segmentation (regular grids or irregular blobs): two segments
    are neighbors iff at least one pair of their pixels shares an edge
    (4-connectivity).

    Args:
        segments: 2D tensor mapping pixels to segment IDs

    Returns:
        List of frozensets — adj[i] is the set of neighbors of segment i
    """
    num_segments = int(segments.max().item()) + 1
    adjacency_sets: list[set[int]] = [set() for _ in range(num_segments)]

    # Vectorized horizontal adjacency
    left = segments[:, :-1].flatten()
    right = segments[:, 1:].flatten()
    mask_h = left != right
    edges_h = torch.column_stack([left[mask_h], right[mask_h]])

    # Vectorized vertical adjacency
    top = segments[:-1, :].flatten()
    bottom = segments[1:, :].flatten()
    mask_v = top != bottom
    edges_v = torch.column_stack([top[mask_v], bottom[mask_v]])

    all_edges = torch.vstack([edges_h, edges_v])
    all_edges, _ = torch.sort(all_edges, dim=1)
    all_edges = torch.unique(all_edges, dim=0)
    for edge in all_edges:
        seg1, seg2 = edge[0].item(), edge[1].item()
        adjacency_sets[seg1].add(seg2)
        adjacency_sets[seg2].add(seg1)

    return [frozenset(neighbors) for neighbors in adjacency_sets]


def _build_hex_adjacency_list(
    hex_to_id: dict[tuple[int, int], int], max_id: int
) -> list[frozenset[int]]:
    """Create a static adjacency list optimized for fast reading.

    Args:
        hex_to_id: Dict mapping (q, r) -> int_id (0 to N-1)
        max_id: Total number of segments (N)

    Returns:
        adj_sets: List of frozensets.
                  adj_sets[5] returns e.g. frozenset({4, 6, 12}) - neighbors of segment 5.
    """
    temp_adj: list[set[int]] = [set() for _ in range(max_id)]

    hex_neighbors = [(+1, 0), (+1, -1), (0, -1), (-1, 0), (-1, +1), (0, +1)]

    for (q, r), seg_id in hex_to_id.items():
        for dq, dr in hex_neighbors:
            neighbor_key = (q + dq, r + dr)
            if neighbor_key in hex_to_id:
                temp_adj[seg_id].add(hex_to_id[neighbor_key])

    return [frozenset(neighbors) for neighbors in temp_adj]


def _create_square_grid(
    input_tensor: torch.Tensor, square_size: int = 14
) -> ImageGraph:
    """Create a grid of squares with frozenset adjacency."""
    _channels, height, width = input_tensor.shape
    segments = torch.zeros(
        (height, width), dtype=torch.int32, device=input_tensor.device
    )

    segment_id = 0

    for row in range(0, height, square_size):
        for col in range(0, width, square_size):
            row_end = min(row + square_size, height)
            col_end = min(col + square_size, width)
            segments[row:row_end, col:col_end] = segment_id
            segment_id += 1

    adj_list = _build_pixel_adjacency_list(segments)
    return ImageGraph(segments=segments, adj_list=adj_list)


def _create_hexagonal_grid(
    input_tensor: torch.Tensor, hex_radius: int = 14
) -> ImageGraph:
    """Create a grid of hexagons with adjacency list using vectorized operations.

    Uses axial coordinate system for precise hexagonal tiling (flat-top orientation).
    Each interior hexagon has 6 neighbors; boundary hexagons have fewer.

    Args:
        input_tensor: Input image tensor [C, H, W]
        hex_radius: Hex size parameter (distance from center to flat edge, default: 14)

    Returns:
        ImageGraph containing segments tensor and adjacency list
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
    segments_np = segments_flat.reshape((height, width)).astype(np.int32)
    segments = torch.tensor(segments_np, dtype=torch.int32, device=input_tensor.device)

    # Build hex_to_id mapping for adjacency construction
    hex_to_id = {(int(q), int(r)): idx for idx, (q, r) in enumerate(unique_qr)}

    # Build adjacency list using axial coordinate neighbors
    adj_list = _build_hex_adjacency_list(hex_to_id, len(hex_to_id))

    return ImageGraph(segments=segments, adj_list=adj_list)


def make_hexagonal_segmentation(hex_radius: int = 4) -> SegmentationFn:
    """Return a function that segments images into a hexagonal grid.

    Args:
        hex_radius: Distance from hexagon center to flat edge.

    Returns:
        SegmentationFn: A callable that generates a hexagonal ImageGraph.
    """
    if hex_radius <= 0:
        raise ValueError(f"hex_radius must be > 0, got {hex_radius}")

    def segmentation(image: torch.Tensor) -> ImageGraph:
        return _create_hexagonal_grid(image, hex_radius=hex_radius)

    return segmentation


def make_square_segmentation(square_size: int = 4) -> SegmentationFn:
    """Return a function that segments images into a square grid.

    Args:
        square_size: Size of each square segment block edge.

    Returns:
        SegmentationFn: A callable that generates a square bounding ImageGraph.
    """
    if square_size <= 0:
        raise ValueError(f"square_size must be > 0, got {square_size}")

    def segmentation(image: torch.Tensor) -> ImageGraph:
        return _create_square_grid(image, square_size=square_size)

    return segmentation


def _create_slic_segments(
    input_tensor: torch.Tensor,
    n_segments: int,
    compactness: float,
) -> ImageGraph:
    """Run SLIC on a (ImageNet-normalized) image tensor and build an ImageGraph."""
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    img = input_tensor.detach().cpu().float().numpy().transpose(1, 2, 0)
    img = np.clip(img * std + mean, 0.0, 1.0)

    labels = slic(
        img,
        n_segments=n_segments,
        compactness=compactness,
        start_label=0,
        enforce_connectivity=True,
        channel_axis=-1,
    )

    segments = torch.tensor(
        labels.astype(np.int32), dtype=torch.int32, device=input_tensor.device
    )
    adj_list = _build_pixel_adjacency_list(segments)
    return ImageGraph(segments=segments, adj_list=adj_list)


def make_slic_segmentation(
    n_segments: int = 200, compactness: float = 10.0
) -> SegmentationFn:
    """Return a function that segments images with SLIC superpixels.

    Args:
        n_segments: Approximate number of superpixels to produce.
        compactness: Balances color proximity and space proximity. Higher values
            give more weight to space proximity, yielding more compact, squarer
            superpixels.

    Returns:
        SegmentationFn: A callable that generates a SLIC ImageGraph.
    """
    if n_segments <= 0:
        raise ValueError(f"n_segments must be > 0, got {n_segments}")
    if compactness <= 0:
        raise ValueError(f"compactness must be > 0, got {compactness}")

    def segmentation(image: torch.Tensor) -> ImageGraph:
        return _create_slic_segments(
            image,
            n_segments=n_segments,
            compactness=compactness,
        )

    return segmentation
