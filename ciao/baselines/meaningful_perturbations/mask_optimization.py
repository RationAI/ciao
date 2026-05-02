"""Meaningful-Perturbations mask optimization (deletion game).

Implements the mask-learning approach from Fong & Vedaldi 2017
("Interpretable Explanations of Black Boxes by Meaningful Perturbations", ICCV).

The objective is: find the smallest mask m such that perturbing the image
inside the mask (replacing with the replacement image) causes the model to
lose confidence in the target class.

Loss = prob_target(perturbed) + λ_area * mean(mask) + λ_tv * TV_β(mask)

The mask is parameterized at low resolution (h//step x w//step) with a sigmoid
activation, bilinearly upsampled to full resolution, then blurred with a Gaussian
kernel -- matching the paper's upsampling + smoothing scheme from sec. 4 of the paper.

Convention: mask=1 means replace (delete), mask=0 means keep.  This is 1-m from
the paper's notation, where m=1 means keep.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class MPResult:
    """Output of the meaningful-perturbations optimizer."""

    soft_mask: torch.Tensor  # [H, W], float in [0, 1]; 1 = perturbed, 0 = kept
    iterations: int
    final_loss: float
    final_target_logprob: float
    final_area: float  # fraction of pixels with mask > 0.5
    trajectory: list[dict[str, float]] = field(default_factory=list)


def _make_gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    """Build a separable 2D Gaussian conv kernel [1, 1, K, K]."""
    k = max(3, int(6 * sigma + 1) | 1)  # force odd
    half = k // 2
    x = torch.arange(-half, half + 1, dtype=torch.float32, device=device)
    g = torch.exp(-x.pow(2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = g.unsqueeze(0) * g.unsqueeze(1)
    return kernel.view(1, 1, k, k)


def _total_variation(mask: torch.Tensor, beta: float = 3.0) -> torch.Tensor:
    """TV regularization of a [H, W] mask: mean |∇m|^β (eq. 4 in paper)."""
    dh = (mask[1:, :] - mask[:-1, :]).abs().pow(beta)
    dw = (mask[:, 1:] - mask[:, :-1]).abs().pow(beta)
    return dh.mean() + dw.mean()


def meaningful_perturbation(
    model: torch.nn.Module,
    input_batch: torch.Tensor,
    target_class_idx: int,
    replacement_image: torch.Tensor,
    *,
    area_lambda: float = 1e-4,
    area_lambda_growth: float = 1.0,
    tv_lambda: float = 1e-2,
    tv_beta: float = 3.0,
    max_time: float = 60.0,
    max_iterations: int = 300,
    learning_rate: float = 0.1,
    mask_step: int = 8,
    mask_sigma: float = 5.0,
    jitter: bool = True,
    jitter_tau: int = 4,
    trajectory_log_every: int = 10,
) -> MPResult:
    """Run the meaningful-perturbations deletion-game optimization.

    Args:
        model: Eval-mode classifier returning logits [B, num_classes].
        input_batch: Input image [1, C, H, W] on the model's device.
        target_class_idx: Class index to suppress.
        replacement_image: Replacement tensor [C, H, W] matching input_batch.
        area_lambda: Weight on mask area penalty λ1 (paper default 1e-4).
        area_lambda_growth: Per-iteration multiplicative growth for area_lambda (1.0 = fixed).
        tv_lambda: Weight on TV regularization λ2 (paper default 1e-2).
        tv_beta: Exponent for TV norm β (paper default 3.0).
        max_time: Wall-clock budget in seconds.
        max_iterations: Hard cap on optimizer iterations.
        learning_rate: Adam learning rate (paper default 0.1).
        mask_step: Downscale factor for low-resolution mask (paper uses 8 -> 28x28 for 224px).
        mask_sigma: Gaussian blur sigma applied after upsampling (paper default 5.0).
        jitter: Whether to apply random 2D shift each iteration.
        jitter_tau: Jitter range in pixels — shift drawn from [0, jitter_tau) (paper default 4).
        trajectory_log_every: Interval between trajectory log entries.

    Returns:
        MPResult with the learned deletion mask.
    """
    device = input_batch.device
    _, _c, h, w = input_batch.shape
    replacement_batch = replacement_image.unsqueeze(0)  # [1, C, H, W]

    mask_h = max(1, h // mask_step)
    mask_w = max(1, w // mask_step)

    # Sigmoid-parameterized low-res mask; init near 0 (delete nothing).
    # sigmoid(-4) ≈ 0.018, so the starting mask is nearly transparent.
    pmask_logits = torch.full(
        (1, 1, mask_h, mask_w),
        fill_value=-4.0,
        device=device,
        dtype=input_batch.dtype,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([pmask_logits], lr=learning_rate)

    # Precompute Gaussian kernel (static across iterations).
    gauss_kernel = _make_gaussian_kernel(mask_sigma, device)
    gauss_pad = gauss_kernel.shape[-1] // 2

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    trajectory: list[dict[str, float]] = []
    best_loss = float("inf")
    start_time = time.monotonic()

    for iteration in range(1, max_iterations + 1):
        if time.monotonic() - start_time > max_time:
            break

        optimizer.zero_grad()

        # Build full-resolution deletion mask: sigmoid → upsample → Gaussian blur → clamp.
        pmask = torch.sigmoid(pmask_logits)
        mask_up = F.interpolate(
            pmask, size=(h, w), mode="bilinear", align_corners=False
        )
        mask_smooth = F.conv2d(mask_up, gauss_kernel, padding=gauss_pad).clamp(0.0, 1.0)

        inp = input_batch
        repl = replacement_batch
        if jitter and jitter_tau > 0:
            dy = random.randint(0, jitter_tau - 1)
            dx = random.randint(0, jitter_tau - 1)
            inp = torch.roll(inp, shifts=(dy, dx), dims=(-2, -1))
            repl = torch.roll(repl, shifts=(dy, dx), dims=(-2, -1))

        # Deletion: mask=1 → replace with repl, mask=0 → keep original.
        perturbed = (1.0 - mask_smooth) * inp + mask_smooth * repl

        logits = model(perturbed)
        probs = F.softmax(logits, dim=1)
        target_prob = probs[0, target_class_idx]

        mask_2d = mask_smooth.squeeze()  # [H, W]
        loss = (
            target_prob
            + area_lambda * mask_2d.mean()
            + tv_lambda * _total_variation(mask_2d, beta=tv_beta)
        )

        loss.backward()
        optimizer.step()
        area_lambda *= area_lambda_growth

        loss_val = float(loss.item())
        if loss_val < best_loss:
            best_loss = loss_val

        if iteration % trajectory_log_every == 0:
            trajectory.append(
                {
                    "evals": float(iteration),
                    "best_score": float(-torch.log(target_prob + 1e-10).item()),
                    "time": time.monotonic() - start_time,
                }
            )

    with torch.no_grad():
        pmask = torch.sigmoid(pmask_logits)
        mask_up = F.interpolate(
            pmask, size=(h, w), mode="bilinear", align_corners=False
        )
        final_mask = (
            F.conv2d(mask_up, gauss_kernel, padding=gauss_pad).squeeze().clamp(0.0, 1.0)
        )

        final_perturbed = (1.0 - final_mask) * input_batch.squeeze(
            0
        ) + final_mask * replacement_image
        final_logits = model(final_perturbed.unsqueeze(0))
        final_log_probs = F.log_softmax(final_logits, dim=1)
        final_target_logprob = float(final_log_probs[0, target_class_idx].item())
        final_area = float((final_mask > 0.5).float().mean().item())

    return MPResult(
        soft_mask=final_mask.detach(),
        iterations=iteration,
        final_loss=best_loss,
        final_target_logprob=final_target_logprob,
        final_area=final_area,
        trajectory=trajectory,
    )
