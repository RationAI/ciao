"""Meaningful-Perturbations mask optimization (deletion game).

Implements the mask-learning approach from Fong & Vedaldi 2017
("Interpretable Explanations of Black Boxes by Meaningful Perturbations", ICCV).

The objective is: find the smallest mask m such that perturbing the image
inside the mask (replacing with the replacement image) causes the model to
lose confidence in the target class.

Loss = log_prob_target(perturbed) + λ_area * mean(m) + λ_tv * TV(m)

Minimizing this drives the model probability down (first term, note: we want
it low, so we minimize it directly) while keeping the mask small (second term)
and smooth (third term via total variation).

We borrow TorchRay's MaskGenerator for the smooth low-resolution parameterization.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torchray.attribution.extremal_perturbation import MaskGenerator


@dataclass
class MPResult:
    """Output of the meaningful-perturbations optimizer."""

    soft_mask: torch.Tensor  # [H, W], float in [0, 1]; 1 = perturbed, 0 = kept
    iterations: int
    final_loss: float
    final_target_logprob: float
    final_area: float  # fraction of pixels with mask > 0.5
    trajectory: list[dict[str, float]] = field(default_factory=list)


def _total_variation(mask: torch.Tensor) -> torch.Tensor:
    """Isotropic total variation of a [H, W] mask."""
    diff_h = (mask[1:, :] - mask[:-1, :]).pow(2)
    diff_w = (mask[:, 1:] - mask[:, :-1]).pow(2)
    return diff_h.mean() + diff_w.mean()


def meaningful_perturbation(
    model: torch.nn.Module,
    input_batch: torch.Tensor,
    target_class_idx: int,
    replacement_image: torch.Tensor,
    *,
    area_lambda: float = 8.0,
    tv_lambda: float = 1e-2,
    max_time: float = 60.0,
    max_iterations: int = 800,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    mask_step: int = 7,
    mask_sigma: float = 21.0,
    jitter: bool = True,
    trajectory_log_every: int = 10,
) -> MPResult:
    """Run the meaningful-perturbations deletion-game optimization.

    Args:
        model: Eval-mode classifier returning logits [B, num_classes].
        input_batch: Input image [1, C, H, W] on the model's device.
        target_class_idx: Class index to suppress.
        replacement_image: Replacement tensor [C, H, W] matching input_batch.
        area_lambda: Weight on the mask area penalty (encourages small mask).
        tv_lambda: Weight on total-variation smoothness penalty.
        max_time: Wall-clock budget in seconds.
        max_iterations: Hard cap on optimizer iterations.
        learning_rate: SGD learning rate.
        momentum: SGD momentum.
        mask_step: Low-resolution parameterization step (pixels).
        mask_sigma: MaskGenerator Gaussian kernel sigma.
        jitter: Horizontal-flip augmentation every other iteration.
        trajectory_log_every: Interval between trajectory log entries.

    Returns:
        MPResult with the learned deletion mask.
    """
    device = input_batch.device
    _, _c, h, w = input_batch.shape
    replacement_batch = replacement_image.unsqueeze(0)  # [1, C, H, W]

    mask_gen = MaskGenerator(
        shape=(h, w),
        step=mask_step,
        sigma=mask_sigma,
        clamp=True,
    ).to(device)
    h_lo, w_lo = mask_gen.shape_in
    # Separate low-res parameter tensor; MaskGenerator.generate() upsamples it.
    pmask = torch.full(
        (1, 1, h_lo, w_lo),
        0.5,
        device=device,
        dtype=input_batch.dtype,
        requires_grad=True,
    )
    optimizer = torch.optim.SGD(
        [pmask],
        lr=learning_rate,
        momentum=momentum,
        dampening=momentum,
    )

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

        # mask_cropped: [1, 1, H, W], deletion mask: 1 = replace, 0 = keep
        mask_cropped, _ = mask_gen.generate(pmask)

        inp = input_batch
        repl = replacement_batch
        if jitter and iteration % 2 == 0:
            inp = torch.flip(inp, dims=[-1])
            repl = torch.flip(repl, dims=[-1])
            perturbed = (1.0 - mask_cropped) * inp + mask_cropped * repl
            perturbed = torch.flip(perturbed, dims=[-1])
        else:
            perturbed = (1.0 - mask_cropped) * inp + mask_cropped * repl

        logits = model(perturbed)
        log_probs = F.log_softmax(logits, dim=1)
        target_logprob = log_probs[0, target_class_idx]

        mask_2d = mask_cropped.squeeze()  # [H, W]
        loss = (
            target_logprob
            + area_lambda * mask_2d.mean()
            + tv_lambda * _total_variation(mask_2d)
        )

        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        if loss_val < best_loss:
            best_loss = loss_val

        if iteration % trajectory_log_every == 0:
            trajectory.append(
                {
                    "evals": float(iteration),
                    "best_score": float(-target_logprob.item()),
                    "time": time.monotonic() - start_time,
                }
            )

    with torch.no_grad():
        final_mask, _ = mask_gen.generate(pmask)
        final_mask = final_mask.squeeze()  # [H, W]
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
