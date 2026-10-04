from torchvision.models import ResNet50_Weights


def get_imagenet_classes() -> list[str]:
    """Retrieve ImageNet class names from torchvision."""
    return ResNet50_Weights.DEFAULT.meta["categories"]


def get_pcam_classes() -> list[str]:
    """Class names for PatchCamelyon (PCam) binary classification."""
    return ["non-tumor", "tumor"]
