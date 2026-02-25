import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F


class ModelPredictor:
    """Handles model predictions and class information."""

    def __init__(self, model: torch.nn.Module, class_names: list[str]) -> None:
        self.model = model
        self.class_names = class_names
        self.device = next(model.parameters()).device
        self.replacement_image = None

        # Pre-compute normalization constants
        self.imagenet_mean = (
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        )
        self.imagenet_std = (
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        )

    def get_predictions(self, input_batch: torch.Tensor) -> torch.Tensor:
        """Get model predictions (returns probabilities)."""
        with torch.no_grad():
            outputs = self.model(input_batch)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            return probabilities

    def predict_image(
        self, input_batch: torch.Tensor, top_k: int = 5
    ) -> list[tuple[int, str, float]]:
        """Get top-k predictions for an image."""
        probabilities = self.get_predictions(input_batch)
        top_probs, top_indices = torch.topk(probabilities[0], top_k)

        results = []
        for i in range(top_k):
            class_idx = int(top_indices[i].item())
            prob = float(top_probs[i].item())
            class_name = (
                self.class_names[class_idx]
                if class_idx < len(self.class_names)
                else f"class_{class_idx}"
            )
            results.append((class_idx, class_name, prob))
        return results

    def calculate_image_mean_color(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Calculate image mean color using pre-computed constants."""
        # Add batch dimension if needed
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        unnormalized = (input_tensor * self.imagenet_std) + self.imagenet_mean
        mean_color = unnormalized.mean(dim=(2, 3), keepdim=True)
        normalized_mean = (mean_color - self.imagenet_mean) / self.imagenet_std
        return normalized_mean.squeeze(0)  # Remove batch dimension

    def get_replacement_image(
        self,
        input_tensor: torch.Tensor,
        replacement: str = "mean_color",
        color: tuple[int, int, int] = (0, 0, 0),
    ) -> torch.Tensor:
        """Generate replacement image for masking operations.

        Args:
            input_tensor: Input tensor [3, H, W] (ImageNet normalized)
            replacement: Strategy - "mean_color", "interlacing", "blur", or "solid_color"
            color: For solid_color mode, RGB tuple (0-255). Defaults to black (0, 0, 0)

        Returns:
            replacement_image: torch tensor [3, H, W] on same device
        """
        # Ensure tensor is on correct device
        input_tensor = input_tensor.to(self.device)

        # Extract spatial dimensions from input tensor
        _, height, width = input_tensor.shape

        if replacement == "mean_color":
            # Fill entire image with mean color
            mean_color = self.calculate_image_mean_color(input_tensor)  # [3, 1, 1]
            replacement_image = mean_color.expand(-1, height, width)  # [3, H, W]

        elif replacement == "interlacing":
            # Create interlaced pattern: even columns flipped vertically, then even indices flipped horizontally
            replacement_image = input_tensor.clone()
            even_indices = torch.arange(0, height, 2)  # Even row indices

            # Step 1: Flip even columns vertically (upside down)
            replacement_image[:, :, even_indices] = torch.flip(
                replacement_image[:, :, even_indices], dims=[1]
            )

            # Step 2: Flip even indices horizontally (left-right)
            replacement_image[:, even_indices, :] = torch.flip(
                replacement_image[:, even_indices, :], dims=[2]
            )

        elif replacement == "blur":
            # Apply Gaussian blur using conv2d
            # Create 7x7 Gaussian kernel (sigma≈1.5 for noticeable but not extreme blur)
            kernel_size = 7
            sigma = 1.5

            # Generate 1D Gaussian kernel
            x = torch.arange(kernel_size, dtype=torch.float32, device=self.device)
            x = x - kernel_size // 2
            gaussian_1d = torch.exp(-(x**2) / (2 * sigma**2))
            gaussian_1d = gaussian_1d / gaussian_1d.sum()

            # Create 2D kernel by outer product
            gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
            gaussian_2d = gaussian_2d / gaussian_2d.sum()

            # Create convolution kernel for each channel
            kernel = gaussian_2d.expand(3, 1, kernel_size, kernel_size)

            # Apply blur with padding to maintain image size
            input_batch = input_tensor.unsqueeze(0)  # [1, 3, H, W]
            padding = kernel_size // 2

            replacement_image = F.conv2d(
                input_batch,
                kernel,
                padding=padding,
                groups=3,  # Apply same kernel to each channel independently
            ).squeeze(0)  # [3, H, W]

        elif replacement == "solid_color":
            # Fill with specified solid color (expects RGB values in 0-255 range)
            # Convert color to torch tensor (always assume 0-255 range)
            color_tensor = torch.tensor(color, dtype=torch.float32, device=self.device)

            # Convert from 0-255 range to 0-1 range
            color_tensor = color_tensor / 255.0

            # Apply ImageNet normalization - squeeze to remove batch dimension from constants
            color_tensor = color_tensor.view(3, 1, 1)  # [3, 1, 1]
            mean = self.imagenet_mean.squeeze(0)  # [3, 1, 1]
            std = self.imagenet_std.squeeze(0)  # [3, 1, 1]
            normalized_color = (color_tensor - mean) / std
            replacement_image = normalized_color.expand(-1, height, width)  # [3, H, W]

        else:
            raise ValueError(f"Unknown replacement strategy: {replacement}")

        return replacement_image

    def plot_image_mean_color(self, input_tensor: torch.Tensor) -> None:
        normalized_mean = self.calculate_image_mean_color(input_tensor).unsqueeze(0)
        plt.imshow(normalized_mean[0].permute(1, 2, 0))
        plt.show()

    def get_class_logit_batch(
        self, input_batch: torch.Tensor, target_class_idx: int
    ) -> torch.Tensor:
        """Get logits for a batch of images - optimized for batched inference (directly from model outputs)."""
        with torch.no_grad():
            outputs = self.model(input_batch)  # Get raw logits

            # experiment with logarithms
            # probabilities = self.get_predictions(input_batch)
            # result = torch.log(probabilities) - torch.log(1 - probabilities)

            return outputs[:, target_class_idx]


def create_surrogate_dataset(
    predictor: ModelPredictor,
    input_batch: torch.Tensor,
    segments: np.ndarray,
    graph: nx.Graph,
    target_class_idx: int,
    neighborhood_distance: int = 1,
    batch_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Create surrogate dataset for interpretability.

    Each row represents one masking operation:
    - Features (X): Binary indicator vector [num_segments] - 1 if segment was masked, 0 otherwise
    - Target (y): Delta score (original_logit - masked_logit)

    This dataset can be used for:
    - Computing segment importance scores
    - Training interpretable models (like LIME does)
    - Analyzing masking effects

    Args:
        predictor: ModelPredictor instance
        input_batch: Input tensor batch
        segments: Pixel-to-segment mapping array [H, W]
        graph: NetworkX graph of spatial relationships
        target_class_idx: Target class index
        neighborhood_distance: Distance for neighborhood masking
        batch_size: Batch size for processing segments

    Returns:
        X: Binary indicator matrix [num_samples, num_segments]
        y: Delta scores array [num_samples]
    """
    # Get original logit
    original_logit = predictor.get_class_logit_batch(input_batch, target_class_idx)[
        0
    ].item()
    print(f"Original logit: {original_logit}")
    print(
        f"Probability of class {target_class_idx}: "
        f"{predictor.get_predictions(input_batch)[0, target_class_idx].item()}"
    )

    segment_ids = list(graph.nodes())
    num_segments = len(segment_ids)

    # Pre-compute local groups (segment + neighbors within distance)
    local_groups = []
    for segment_id in segment_ids:
        # Get neighbors within specified distance using BFS
        neighbors = {segment_id}
        current_layer = {segment_id}

        for _ in range(neighborhood_distance):
            next_layer = set()
            for node in current_layer:
                next_layer.update(graph.neighbors(node))
            next_layer -= neighbors
            neighbors.update(next_layer)
            current_layer = next_layer

        local_groups.append(list(neighbors))

    # Calculate deltas for all local groups
    deltas = calculate_hyperpixel_deltas(
        predictor,
        input_batch,
        segments,
        local_groups,
        target_class_idx,
        batch_size=batch_size,
    )

    # Create surrogate dataset
    num_samples = len(local_groups)
    X = np.zeros((num_samples, num_segments), dtype=np.float32)
    y = np.array(deltas, dtype=np.float32)

    # Fill indicator matrix
    for i, masked_segments in enumerate(local_groups):
        for segment_id in masked_segments:
            X[i, segment_id] = 1.0

    print(f"Created surrogate dataset: X shape {X.shape}, y shape {y.shape}")
    print(f"Average delta: {y.mean():.4f}, std: {y.std():.4f}")

    return X, y


def calculate_scores_from_surrogate(X: np.ndarray, y: np.ndarray) -> dict[int, float]:  # noqa: N803
    """Calculate averaged segment importance scores from surrogate dataset.

    For each segment, averages all delta scores where that segment was masked.
    This provides an importance score representing how much the segment
    contributes to the prediction.

    Args:
        X: Binary indicator matrix [num_samples, num_segments]
        y: Delta scores array [num_samples]

    Returns:
        Dict mapping segment_id -> averaged score
    """
    num_segments = X.shape[1]
    scores = {}

    for segment_id in range(num_segments):
        # Find all samples where this segment was masked
        mask = X[:, segment_id] == 1.0

        segment_scores = y[mask]
        scores[segment_id] = float(segment_scores.mean())

    score_values = list(scores.values())
    print(f"Score range: [{min(score_values):.4f}, {max(score_values):.4f}]")

    return scores


def get_predicted_class(predictor: ModelPredictor, input_batch: torch.Tensor) -> int:
    """Get the predicted class index from model output."""
    predictions = predictor.predict_image(input_batch, top_k=1)
    return predictions[0][0]


def calculate_hyperpixel_deltas(
    predictor: ModelPredictor,
    input_batch: torch.Tensor,
    segments: np.ndarray,
    hyperpixel_segment_ids_list: list[list[int]],
    target_class_idx: int,
    batch_size: int = 64,
) -> list[float]:
    """Calculate masking deltas for hyperpixel candidates using batched inference.

    Handles internal batching to prevent memory overflow with large path counts.

    Args:
        predictor: ModelPredictor instance
        input_batch: Input tensor batch [1, 3, H, W]
        segments: Pixel-to-segment mapping array [H, W]
        hyperpixel_segment_ids_list: List of segment ID lists, e.g. [[1,2,3], [4,5,6]]
        target_class_idx: Target class index
        batch_size: Batch size

    Returns:
        List[float]: Delta scores for each candidate
    """
    if not hyperpixel_segment_ids_list:
        return []

    # Validate all segment lists are non-empty
    for i, segment_ids in enumerate(hyperpixel_segment_ids_list):
        if not segment_ids:
            raise ValueError(f"Empty segment list at index {i}")

    with torch.no_grad():  # Prevent gradient accumulation
        original_logit = predictor.get_class_logit_batch(input_batch, target_class_idx)[
            0
        ].item()

        # Get replacement image using the specified strategy
        assert predictor.replacement_image is not None
        replacement_image = predictor.replacement_image

        # Convert segments numpy array to GPU tensor once (outside loop)
        gpu_segments = torch.from_numpy(segments).to(predictor.device)

        # Process in batches to avoid memory overflow
        all_deltas = []
        num_masks = len(hyperpixel_segment_ids_list)

        for batch_start in range(0, num_masks, batch_size):
            batch_end = min(batch_start + batch_size, num_masks)
            current_batch_size = batch_end - batch_start

            batch_inputs = input_batch.repeat(current_batch_size, 1, 1, 1)

            for i, segment_ids in enumerate(
                hyperpixel_segment_ids_list[batch_start:batch_end]
            ):
                # Optimized: Use torch.isin for fast GPU-based mask creation
                target_ids = torch.tensor(
                    segment_ids, dtype=gpu_segments.dtype, device=predictor.device
                )
                combined_mask = torch.isin(gpu_segments, target_ids)

                # Apply mask with proper broadcasting
                batch_inputs[i] = torch.where(
                    combined_mask.unsqueeze(0),  # [1, H, W] broadcasts to [3, H, W]
                    replacement_image,  # [3, H, W]
                    batch_inputs[i],  # [3, H, W]
                )

            masked_logits = predictor.get_class_logit_batch(
                batch_inputs, target_class_idx
            )
            batch_deltas = [
                original_logit - masked_logit.item() for masked_logit in masked_logits
            ]
            all_deltas.extend(batch_deltas)

            # Memory cleanup
            del batch_inputs, masked_logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return all_deltas


def select_top_hyperpixels(
    hyperpixels: list[dict[str, object]], max_hyperpixels: int = 10
) -> list[dict[str, object]]:
    """Select top hyperpixels by their primary algorithm-specific score."""
    # Use hyperpixel_score
    return sorted(
        hyperpixels,
        key=lambda hp: abs(hp.get("hyperpixel_score", 0)),  # type: ignore[arg-type]
        reverse=True,
    )[:max_hyperpixels]
