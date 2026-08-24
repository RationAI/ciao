import timm
from torch import nn


def load_pcam_model() -> nn.Module:
    """Load resnet50.tiatoolbox-pcam from HuggingFace via timm."""
    return timm.create_model(
        model_name="hf-hub:1aurent/resnet50.tiatoolbox-pcam",
        pretrained=True,
    )
