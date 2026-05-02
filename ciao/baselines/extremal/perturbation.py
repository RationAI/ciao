"""Extremal Perturbations optimization (preservation game).

Inspired by Fong, Patrick & Vedaldi 2019 ("Understanding Deep Networks via
Extremal Perturbations and Smooth Masks", ICCV) and the TorchRay reference
implementation. We solve the preservation variant: find a smooth mask of
fixed area that, when used to keep a fraction of the image and replace the
rest with the user-supplied replacement image, maximises the model's
log-probability for the target class.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchray.attribution.extremal_perturbation import MaskGenerator


@dataclass
class EPResult:
    """Output of the EP optimizer."""

    soft_mask: torch.Tensor  # [H, W], float in [0, 1]
    iterations: int
    final_loss: float
    final_target_logprob: float
    final_area: float  # fraction of pixels with mask > 0.5
    trajectory: list[dict[str, float]]


def extremal_perturbation(
    model: torch.nn.Module,
    input_batch: torch.Tensor,
    target_class_idx: int,
    replacement_image: torch.Tensor,
    *,
    area: float = 0.1,
    max_time: float = 60.0,
    max_iterations: int = 800,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    mask_step: int = 7,
    mask_sigma: float = 21.0,
    area_lambda: float = 300.0,
    area_lambda_growth: float = 1.0035,
    jitter: bool = True,
    trajectory_log_every: int = 10,
) -> EPResult:
    """Run extremal-perturbation preservation-game optimization.

    Args:
        model: Eval-mode classifier returning logits ``[B, num_classes]``.
        input_batch: Input image tensor ``[1, C, H, W]`` on the model's device.
        target_class_idx: Class index to preserve.
        replacement_image: Replacement tensor ``[C, H, W]`` matching ``input_batch``.
        area: Target fraction of pixels to keep (e.g. ``0.1``).
        max_time: Wall-clock budget in seconds. Optimization stops at the
            first iteration boundary after the budget is exceeded.
        max_iterations: Hard cap on optimization iterations.
        learning_rate: SGD learning rate.
        momentum: SGD momentum (also used as dampening, TorchRay-style).
        mask_step: Spatial downsampling factor for the parameter mask.
        mask_sigma: Kernel sigma (in image-space pixels) for TorchRay's
            ``MaskGenerator``. Also sets the mask margin (= sigma).
        area_lambda: Weight on the area-constraint penalty.
        area_lambda_growth: Per-iteration multiplicative growth of
            ``area_lambda`` (TorchRay-style annealing).
        jitter: If True, horizontally flip the perturbed input on alternating
            iterations. Breaks the zero-padding / corner-bias artifact that
            otherwise pushes the mask to image edges.
        trajectory_log_every: Append a trajectory point every N iterations.

    Returns:
        EPResult with the soft mask and optimization metadata.
    """
    if not 0.0 < area < 1.0:
        raise ValueError(f"area must be in (0, 1), got {area}")
    if max_time <= 0:
        raise ValueError(f"max_time must be > 0, got {max_time}")
    if max_iterations <= 0:
        raise ValueError(f"max_iterations must be > 0, got {max_iterations}")
    if mask_step <= 0:
        raise ValueError(f"mask_step must be > 0, got {mask_step}")

    device = input_batch.device
    if input_batch.dim() != 4 or input_batch.shape[0] != 1:
        raise ValueError(
            f"input_batch must have shape [1, C, H, W], got {tuple(input_batch.shape)}"
        )
    _, _, height, width = input_batch.shape
    repl = replacement_image.to(device=device, dtype=input_batch.dtype).unsqueeze(0)

    # Freeze the model — only the mask parameter receives gradients.
    saved_requires_grad = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)

    try:
        # TorchRay's MaskGenerator: unfold + per-position kernel weighting with
        # explicit margin/padding. Avoids the bilinear-upsample-plus-blur edge
        # artifacts (mask drifting to corners, big-square blobs) that a naive
        # parameterization produces.
        mask_generator = MaskGenerator(
            shape=(height, width), step=mask_step, sigma=mask_sigma
        ).to(device)
        h_lo, w_lo = mask_generator.shape_in
        pmask = torch.ones(
            (1, 1, h_lo, w_lo),
            device=device,
            dtype=input_batch.dtype,
        )
        pmask.requires_grad_(True)

        # SGD+momentum (TorchRay default). Adam's per-parameter adaptive
        # learning rates can amplify tiny initial asymmetries before the
        # target_log_prob signal kicks in.
        optimizer = torch.optim.SGD(
            [pmask], lr=learning_rate, momentum=momentum, dampening=momentum
        )

        trajectory: list[dict[str, float]] = []
        start = time.perf_counter()

        soft_mask: torch.Tensor | None = None
        final_loss = float("inf")
        final_target_logprob = float("-inf")
        iters_done = 0
        current_lambda = area_lambda

        for t in range(max_iterations):
            elapsed = time.perf_counter() - start
            if elapsed >= max_time:
                break

            # mask_cropped is [1, 1, H, W] (matches input). mask_with_margin
            # is larger by 2*sigma — the margin lets the mask drift off-image
            # without the area regularizer biasing it toward edges.
            mask_cropped, mask_with_margin = mask_generator.generate(pmask)

            # Preservation: keep mask*x, replace (1-mask) with replacement.
            x_pert = mask_cropped * input_batch + (1.0 - mask_cropped) * repl

            # Horizontal flip on alternating iterations breaks the
            # zero-padding / positional-bias exploit that otherwise pushes
            # the mask to a fixed image corner.
            if jitter and t % 2 == 0:
                x_pert = torch.flip(x_pert, dims=(3,))

            logits = model(x_pert)
            log_probs = F.log_softmax(logits, dim=1)
            target_log_prob = log_probs[0, target_class_idx]

            # Area regularization (TorchRay-style): sort flattened mask
            # ascending and compare against a reference vector that holds 0
            # for the bottom (1 - area) fraction and 1 for the top area
            # fraction. Squared error pushes the histogram toward {0, 1}
            # with the desired count. Use the with-margin mask so the area
            # constraint accounts for mask mass that drifted into the margin.
            sorted_mask, _ = torch.sort(mask_with_margin.flatten(), descending=False)
            n = sorted_mask.numel()
            n_keep = round(area * n)
            ref = torch.zeros_like(sorted_mask)
            if n_keep > 0:
                ref[-n_keep:] = 1.0
            area_loss = ((sorted_mask - ref) ** 2).mean()

            loss = -target_log_prob + current_lambda * area_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pmask.clamp_(0.0, 1.0)

            current_lambda *= area_lambda_growth
            iters_done = t + 1
            final_loss = float(loss.item())
            final_target_logprob = float(target_log_prob.item())
            soft_mask = mask_cropped.detach()

            if t % trajectory_log_every == 0:
                trajectory.append(
                    {
                        "evals": float(iters_done),
                        "best_score": final_target_logprob,
                        "time": float(elapsed),
                    }
                )

        if soft_mask is None:
            # Time budget elapsed before any iteration completed.
            soft_mask = torch.full(
                (1, 1, height, width), 0.5, device=device, dtype=input_batch.dtype
            )

        soft_mask_2d = soft_mask.squeeze(0).squeeze(0)
        final_area = float((soft_mask_2d > 0.5).float().mean().item())

        return EPResult(
            soft_mask=soft_mask_2d,
            iterations=iters_done,
            final_loss=final_loss,
            final_target_logprob=final_target_logprob,
            final_area=final_area,
            trajectory=trajectory,
        )
    finally:
        for p, flag in zip(model.parameters(), saved_requires_grad, strict=True):
            p.requires_grad_(flag)
