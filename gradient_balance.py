"""Gradient diagnostics and safe scaling for auxiliary world-model losses.

The helper measures the base-loss and auxiliary-loss gradients on the exact
parameter support touched by the auxiliary objective.  It returns a detached
scalar multiplier that caps the auxiliary gradient norm relative to the base
Dreamer gradient norm, without changing the base optimizer implementation.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch


def _squared_norm(grads):
    terms = [grad.detach().float().square().sum() for grad in grads if grad is not None]
    if not terms:
        return None
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total


def balance_auxiliary_loss(
    base_loss: torch.Tensor,
    auxiliary_loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    max_ratio: float = 0.10,
    max_norm: float = 25.0,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Scale an auxiliary loss using its gradient norm on shared parameters.

    Let ``g_b`` and ``g_a`` denote gradients of the base and auxiliary losses on
    parameters touched by the auxiliary objective.  The returned multiplier is

        min(1, max_ratio * ||g_b|| / (||g_a|| + eps),
               max_norm / (||g_a|| + eps)).

    Non-finite auxiliary gradients disable the auxiliary update for that batch.
    ``torch.autograd.grad`` is used only for diagnostics; the caller still runs
    the normal optimizer backward pass on ``base_loss + scaled_auxiliary_loss``.
    """

    if base_loss.ndim != 0 or auxiliary_loss.ndim != 0:
        raise ValueError((base_loss.shape, auxiliary_loss.shape))
    if max_ratio < 0.0 or max_norm < 0.0 or eps <= 0.0:
        raise ValueError((max_ratio, max_norm, eps))

    params = [parameter for parameter in parameters if parameter.requires_grad]
    zero = base_loss.detach() * 0.0
    default_metrics = {
        "oa_base_grad_norm_on_aux_support": zero,
        "oa_aux_grad_norm": zero,
        "oa_aux_to_base_grad_ratio": zero,
        "oa_aux_base_grad_cosine": zero,
        "oa_aux_grad_scale": torch.ones_like(zero),
        "oa_aux_grad_finite": torch.ones_like(zero),
    }

    if not auxiliary_loss.requires_grad or not params:
        return auxiliary_loss, default_metrics

    auxiliary_grads = torch.autograd.grad(
        auxiliary_loss,
        params,
        retain_graph=True,
        allow_unused=True,
        create_graph=False,
    )
    support = [
        (parameter, gradient)
        for parameter, gradient in zip(params, auxiliary_grads)
        if gradient is not None
    ]
    if not support:
        metrics = dict(default_metrics)
        metrics["oa_aux_grad_scale"] = zero
        return auxiliary_loss.detach() * 0.0, metrics

    support_params = [item[0] for item in support]
    aux_support_grads = [item[1] for item in support]
    base_support_grads = torch.autograd.grad(
        base_loss,
        support_params,
        retain_graph=True,
        allow_unused=True,
        create_graph=False,
    )

    aux_sq = _squared_norm(aux_support_grads)
    base_sq = _squared_norm(base_support_grads)
    if aux_sq is None:
        metrics = dict(default_metrics)
        metrics["oa_aux_grad_scale"] = zero
        return auxiliary_loss.detach() * 0.0, metrics
    if base_sq is None:
        base_sq = torch.zeros_like(aux_sq)

    aux_norm = torch.sqrt(aux_sq + eps)
    base_norm = torch.sqrt(base_sq + eps)
    finite = torch.isfinite(aux_norm) & torch.isfinite(base_norm)

    ratio_limit = max_ratio * base_norm
    if max_norm > 0.0:
        absolute_limit = torch.as_tensor(max_norm, device=aux_norm.device, dtype=aux_norm.dtype)
        allowed_norm = torch.minimum(ratio_limit, absolute_limit)
    else:
        allowed_norm = ratio_limit
    scale = (allowed_norm / (aux_norm + eps)).clamp(0.0, 1.0)

    dot = torch.zeros_like(aux_sq)
    base_overlap_sq = torch.zeros_like(aux_sq)
    aux_overlap_sq = torch.zeros_like(aux_sq)
    for base_grad, aux_grad in zip(base_support_grads, aux_support_grads):
        if base_grad is None:
            continue
        base_float = base_grad.detach().float()
        aux_float = aux_grad.detach().float()
        dot = dot + (base_float * aux_float).sum()
        base_overlap_sq = base_overlap_sq + base_float.square().sum()
        aux_overlap_sq = aux_overlap_sq + aux_float.square().sum()
    cosine = dot / (
        torch.sqrt(base_overlap_sq + eps) * torch.sqrt(aux_overlap_sq + eps) + eps
    )

    if not bool(finite.detach().cpu().item()):
        scale = torch.zeros_like(scale)
        scaled_auxiliary = auxiliary_loss.detach() * 0.0
    elif float(scale.detach().cpu().item()) <= 0.0:
        scaled_auxiliary = auxiliary_loss.detach() * 0.0
    else:
        scaled_auxiliary = auxiliary_loss * scale.detach()

    metrics = {
        "oa_base_grad_norm_on_aux_support": base_norm.detach(),
        "oa_aux_grad_norm": aux_norm.detach(),
        "oa_aux_to_base_grad_ratio": (aux_norm / (base_norm + eps)).detach(),
        "oa_aux_base_grad_cosine": cosine.detach(),
        "oa_aux_grad_scale": scale.detach(),
        "oa_aux_grad_finite": finite.to(aux_norm.dtype).detach(),
    }
    return scaled_auxiliary, metrics
