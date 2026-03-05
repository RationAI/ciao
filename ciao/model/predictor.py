import torch


class ModelPredictor:
    """Handles model predictions and class information."""

    def __init__(self, model: torch.nn.Module, class_names: list[str]) -> None:
        self.model = model
        self.class_names = class_names
        self.device = next(model.parameters()).device

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

    def get_class_logit_batch(
        self, input_batch: torch.Tensor, target_class_idx: int
    ) -> torch.Tensor:
        """Get logits for a batch of images - optimized for batched inference (directly from model outputs)."""
        with torch.no_grad():
            outputs = self.model(input_batch)
            return outputs[:, target_class_idx]
