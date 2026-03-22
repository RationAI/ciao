"""Graph utilities for segment manipulation using frozenset operations."""


def get_frontier(
    current_superpixels: frozenset[int],
    adj_superpixels: list[frozenset[int]],
    used_superpixels: frozenset[int],
) -> frozenset[int]:
    """Compute the expansion frontier (valid neighbors) for graph traversal.

    The frontier is the set of segments adjacent to the current structure
    that can be added in the next step.

    A segment is in the frontier if:
    - It is adjacent to at least one segment in the current superpixels
    - It is NOT already in the current superpixels
    - It is NOT in the used_superpixels (respects global exclusion constraints)

    Args:
        current_superpixels: Set of current superpixel IDs
        adj_superpixels: Tuple of neighbor frozensets (adj_superpixels[i] = neighbors of superpixel i)
        used_superpixels: Set of globally excluded superpixels

    Returns:
        Frozenset of valid frontier segments
    """
    neighbors: set[int] = set()
    for node_id in current_superpixels:
        neighbors |= adj_superpixels[node_id]

    # Remove segments already in the current superpixels and used segments
    frontier = frozenset(neighbors - current_superpixels - used_superpixels)
    return frontier
