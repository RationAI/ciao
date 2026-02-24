import math

import networkx as nx
import numpy as np


def hex_round(q, r):
    """Round axial coordinates to nearest hex.

    Args:
        q: Fractional axial q coordinate
        r: Fractional axial r coordinate

    Returns:
        (q, r): Integer axial coordinates of nearest hex
    """
    x = q
    z = r
    y = -x - z

    rx = round(x)
    ry = round(y)
    rz = round(z)

    dx = abs(rx - x)
    dy = abs(ry - y)
    dz = abs(rz - z)

    if dx > dy and dx > dz:
        rx = -ry - rz
    elif dy > dz:
        ry = -rx - rz
    else:
        rz = -rx - ry

    return int(rx), int(rz)  # back to axial (q = x, r = z)


def pixel_to_hex(px, py, size):
    """Convert pixel coordinate to axial hex coordinates.

    Args:
        px: Pixel x coordinate
        py: Pixel y coordinate
        size: Hex size (distance from center to flat edge for flat-top hexagons)

    Returns:
        (q, r): Axial coordinates of the hex containing this pixel
    """
    q = (math.sqrt(3) / 3 * px - 1 / 3 * py) / size
    r = (2 / 3 * py) / size
    return hex_round(q, r)


def build_hex_adjacency_graph(hex_to_id):
    """Build adjacency graph for hexagonal grid using axial coordinate neighbors.

    Hexagons have exactly 6 neighbors with well-defined axial coordinate offsets:
    [(+1,0), (+1,-1), (0,-1), (-1,0), (-1,+1), (0,+1)]

    Args:
        hex_to_id: Dictionary mapping (q, r) axial coords to segment IDs

    Returns:
        NetworkX graph with edges between adjacent hexagons
    """
    adj_graph = nx.Graph()
    adj_graph.add_nodes_from(hex_to_id.values())

    # Six neighbor offsets for flat-top hexagons in axial coordinates
    hex_neighbors = [
        (+1, 0),  # East
        (+1, -1),  # Northeast
        (0, -1),  # Northwest
        (-1, 0),  # West
        (-1, +1),  # Southwest
        (0, +1),  # Southeast
    ]

    for (q, r), seg_id in hex_to_id.items():
        for dq, dr in hex_neighbors:
            neighbor_key = (q + dq, r + dr)
            if neighbor_key in hex_to_id:
                neighbor_id = hex_to_id[neighbor_key]
                adj_graph.add_edge(seg_id, neighbor_id)

    return adj_graph


def build_adjacency_graph(segments, neighborhood=8):
    adj_graph = nx.Graph()
    segment_ids = np.unique(segments)
    adj_graph.add_nodes_from(segment_ids)

    height, width = segments.shape

    # Check horizontal adjacency
    for y in range(height):
        for x in range(width - 1):
            seg1, seg2 = segments[y, x], segments[y, x + 1]
            if seg1 != seg2:
                adj_graph.add_edge(seg1, seg2)

    # Check vertical adjacency
    for y in range(height - 1):
        for x in range(width):
            seg1, seg2 = segments[y, x], segments[y + 1, x]
            if seg1 != seg2:
                adj_graph.add_edge(seg1, seg2)

    if neighborhood == 8:
        # Add diagonal adjacency for 8-neighborhood
        for y in range(height - 1):
            for x in range(width - 1):
                center_seg = segments[y, x]
                # Check diagonal neighbors
                if segments[y + 1, x + 1] != center_seg:
                    adj_graph.add_edge(center_seg, segments[y + 1, x + 1])
                if x > 0 and segments[y + 1, x - 1] != center_seg:
                    adj_graph.add_edge(center_seg, segments[y + 1, x - 1])

    return adj_graph


def build_fast_adjacency_list(hex_to_id, max_id):
    """Vytvoří 'static adjacency list' optimalizovaný pro rychlé čtení.

    Args:
        hex_to_id: Dict mapující (q, r) -> int_id (0 až N-1)
        max_id: Celkový počet segmentů (N)

    Returns:
        adj_list: Tuple of Tuples.
                  adj_list[5] vrátí např. (4, 6, 12) - sousedy segmentu 5.
    """
    # Inicializujeme prázdné listy pro každé ID
    # Používáme list listů pro konstrukci
    temp_adj = [[] for _ in range(max_id)]

    # Offsets pro sousedy (axial coords)
    hex_neighbors = [(+1, 0), (+1, -1), (0, -1), (-1, 0), (-1, +1), (0, +1)]

    for (q, r), seg_id in hex_to_id.items():
        for dq, dr in hex_neighbors:
            neighbor_key = (q + dq, r + dr)

            # Pokud soused existuje (je uvnitř obrázku)
            if neighbor_key in hex_to_id:
                neighbor_id = hex_to_id[neighbor_key]
                temp_adj[seg_id].append(neighbor_id)

    # Konverze na tuple of tuples pro maximální rychlost čtení a paměťovou efektivitu
    # Seřadíme sousedy (volitelné, ale dobré pro determinismus)
    final_adj = tuple(tuple(sorted(neighbors)) for neighbors in temp_adj)

    return final_adj


# --- Upravená funkce create_hexagonal_grid ---


def create_hexagonal_grid_with_list(input_tensor, hex_radius=14):
    _channels, height, width = input_tensor.shape
    segments = np.zeros((height, width), dtype=np.int32)

    hex_to_id = {}
    next_id = 0

    # 1. Mapování pixelů na Hex ID
    # (Tohle je nejpomalejší část, ale běží jen jednou při initu)
    for y in range(height):
        for x in range(width):
            q, r = pixel_to_hex(x, y, hex_radius)
            key = (q, r)

            if key not in hex_to_id:
                hex_to_id[key] = next_id
                next_id += 1

            segments[y, x] = hex_to_id[key]

    # 2. Vytvoření Rychlého Grafu (žádný NetworkX)
    adjacency_list = build_fast_adjacency_list(hex_to_id, next_id)

    return segments, adjacency_list, next_id


def build_adjacency_bitmasks(adj_list):
    """Převede adjacency list na seznam integerů.
    
    adj_masks[5] bude integer, který má jedničky na pozicích sousedů hexu 5.
    """
    adj_masks = []
    for neighbors in adj_list:
        mask = 0
        for n in neighbors:
            mask |= 1 << n
        adj_masks.append(mask)
    return tuple(adj_masks)


def create_square_grid(input_tensor, square_size=14, neighborhood=8):
    """Create a grid of squares with graph structure representing spatial relationships."""
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

    # Build adjacency graph
    adjacency_graph = build_adjacency_graph(segments, neighborhood=neighborhood)

    return segments, adjacency_graph


def create_hexagonal_grid(input_tensor, hex_radius=14, neighborhood=6):
    """Create a grid of hexagons with graph structure representing spatial relationships.

    Uses axial coordinate system for precise hexagonal tiling (flat-top orientation).
    Each hexagon has exactly 6 neighbors (neighborhood parameter ignored).

    Args:
        input_tensor: Input image tensor [C, H, W]
        hex_radius: Hex size parameter (distance from center to flat edge, default: 14)
        neighborhood: Ignored for hexagons (always 6-connected)

    Returns:
        segments: 2D array mapping pixels to segment IDs
        adjacency_graph: NetworkX graph of segment relationships
    """
    _channels, height, width = input_tensor.shape
    segments = np.zeros((height, width), dtype=np.int32)

    # Map axial coordinates (q, r) to unique segment IDs
    hex_to_id = {}
    next_id = 0

    # Assign each pixel to its corresponding hex using axial coordinates
    for y in range(height):
        for x in range(width):
            q, r = pixel_to_hex(x, y, hex_radius)
            key = (q, r)

            if key not in hex_to_id:
                hex_to_id[key] = next_id
                next_id += 1

            segments[y, x] = hex_to_id[key]

    # Build adjacency graph using axial coordinate neighbors (always 6-connected)
    adjacency_graph = build_hex_adjacency_graph(hex_to_id)

    return segments, adjacency_graph


def create_segmentation(
    input_tensor, segmentation_type="hexagonal", segment_size=14, neighborhood=8
):
    """Create image segmentation with specified type.

    Args:
        input_tensor: Input image tensor [C, H, W]
        segmentation_type: "square" or "hexagonal"
        segment_size: Size parameter (square_size or hex_radius)
        neighborhood: Neighborhood connectivity (4, 6, or 8)

    Returns:
        segments: 2D array mapping pixels to segment IDs
        adjacency_graph: NetworkX graph of segment relationships
    """
    if segmentation_type == "square":
        return create_square_grid(
            input_tensor, square_size=segment_size, neighborhood=neighborhood
        )
    elif segmentation_type == "hexagonal":
        return create_hexagonal_grid(
            input_tensor, hex_radius=segment_size, neighborhood=neighborhood
        )
    else:
        raise ValueError(
            f"Unknown segmentation_type: {segmentation_type}. Use 'square' or 'hexagonal'."
        )
