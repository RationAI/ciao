"""Extremal Perturbations optimization (preservation game).

Inspired by Fong, Patrick & Vedaldi 2019 ("Understanding Deep Networks via
Extremal Perturbations and Smooth Masks", ICCV) and the TorchRay reference
implementation. We solve the preservation variant: find a smooth mask of
fixed area that, when used to keep a fraction of the image and replace the
rest with the user-supplied replacement image, maximises the model's
log-probability for the target class.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


@dataclass
class EPResult:
    """Output of the EP optimizer."""

    soft_mask: torch.Tensor  # [H, W], float in [0, 1]
    iterations: int
    final_loss: float
    final_target_logprob: float
    final_area: float  # fraction of pixels with mask > 0.5
    trajectory: list[dict[str, float]]


def _sigma_to_kernel_size(sigma: float) -> int:
    """Pick an odd kernel size large enough to cover ~3 sigma on each side."""
    k = max(3, int(2 * round(3 * sigma) + 1))
    if k % 2 == 0:
        k += 1
    return k


def extremal_perturbation(
    model: torch.nn.Module,
    input_batch: torch.Tensor,
    target_class_idx: int,
    replacement_image: torch.Tensor,
    *,
    area: float = 0.1,
    max_time: float = 60.0,
    max_iterations: int = 800,
    learning_rate: float = 0.05,
    mask_step: int = 7,
    mask_sigma: float = 21.0,
    area_lambda: float = 300.0,
    area_lambda_growth: float = 1.0035,
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
        learning_rate: Adam learning rate.
        mask_step: Spatial downsampling factor for the parameter mask.
        mask_sigma: Sigma (in upsampled pixels) of Gaussian smoothing applied
            to the mask each iteration.
        area_lambda: Weight on the area-constraint penalty.
        area_lambda_growth: Per-iteration multiplicative growth of
            ``area_lambda`` (TorchRay-style annealing).
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
        h_lo = math.ceil(height / mask_step)
        w_lo = math.ceil(width / mask_step)
        # Initialize at 0.5 so sigmoid-free upsampling gives a roughly
        # half-everywhere mask.
        pmask = torch.full(
            (1, 1, h_lo, w_lo),
            0.5,
            device=device,
            dtype=input_batch.dtype,
            requires_grad=True,
        )

        optimizer = torch.optim.Adam([pmask], lr=learning_rate)

        kernel_size = _sigma_to_kernel_size(mask_sigma) if mask_sigma > 0 else 0

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

            # Upsample low-res param to full image size and smooth.
            mask_full = F.interpolate(
                pmask, size=(height, width), mode="bilinear", align_corners=False
            )
            if kernel_size > 0:
                mask_full = TF.gaussian_blur(
                    mask_full,
                    kernel_size=[kernel_size, kernel_size],
                    sigma=[mask_sigma, mask_sigma],
                )
            mask_full = mask_full.clamp(0.0, 1.0)

            # Preservation: keep mask*x, replace (1-mask) with replacement.
            x_pert = mask_full * input_batch + (1.0 - mask_full) * repl

            logits = model(x_pert)
            log_probs = F.log_softmax(logits, dim=1)
            target_log_prob = log_probs[0, target_class_idx]

            # Area regularization (TorchRay-style): sort flattened mask
            # ascending and compare against a reference vector that holds 0
            # for the bottom (1 - area) fraction and 1 for the top area
            # fraction. Squared error pushes the histogram toward {0, 1}
            # with the desired count.
            sorted_mask, _ = torch.sort(mask_full.flatten(), descending=False)
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
            soft_mask = mask_full.detach()

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
