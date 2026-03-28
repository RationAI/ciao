import torch


class ModelPredictor:
    """Handles model predictions and class information for the CIAO explainer."""

    def __init__(self, model: torch.nn.Module, class_names: list[str]) -> None:
        self.model = model
        self.class_names = class_names

        # Ensure deterministic inference by disabling Dropout and freezing BatchNorm
        self.model.eval()

        # Robustly determine the device (fall back to CPU if model has no parameters)
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")

    def get_predictions(self, input_batch: torch.Tensor) -> torch.Tensor:
        """Get model predictions (returns probabilities)."""
        input_batch = input_batch.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_batch)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            return probabilities

    def get_predicted_class(self, input_batch: torch.Tensor) -> int:
        """Get the most likely class index for a single input."""
        if input_batch.shape[0] != 1:
            raise ValueError(
                f"get_predicted_class expects a single input (batch size 1), but got batch size {input_batch.shape[0]}"
            )

        probs = self.get_predictions(input_batch)
        return int(torch.argmax(probs[0]).item())

    def get_class_logit_batch(
        self, input_batch: torch.Tensor, target_class_idx: int
    ) -> torch.Tensor:
        """Get raw logits for a specific target class across a batch of images."""
        input_batch = input_batch.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_batch)
            return outputs[:, target_class_idx]
