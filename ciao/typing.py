from collections.abc import Callable
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from ciao.algorithm.context import SearchContext
    from ciao.algorithm.graph import ImageGraph
    from ciao.scoring.region import RegionResult


ReplacementFn = Callable[[torch.Tensor], torch.Tensor]
SegmentationFn = Callable[[torch.Tensor], "ImageGraph"]
ExplanationMethodFn = Callable[["SearchContext"], "RegionResult"]
