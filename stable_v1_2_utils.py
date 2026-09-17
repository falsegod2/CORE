import numpy as np
import torch


def gradient_guard_instant_gate(
    base_loss,
    aux_loss,
    params,
    conflict_low=-0.20,
    conflict_high=0.0,
    min_conflict_gate=0.25,
    max_norm_ratio=0.35,
    min_total_gate=0.10,
):
    """Measure auxiliary-vs-core gradient conflict on a small parameter proxy.

    Stable-V1.2 does not rewrite gradient vectors. It only returns a detached
    scalar attenuation gate for the Proto loss.
    """
    params = tuple(p for p in params if p.requires_grad)
    if not params:
        return 1.0, 1.0, 0.0

    base_grads = torch.autograd.grad(
        base_loss, params, retain_graph=True, allow_unused=True
    )
    aux_grads = torch.autograd.grad(
        aux_loss, params, retain_graph=True, allow_unused=True
    )

    dot = None
    base_sq = None
    aux_sq = None
    for gb, ga in zip(base_grads, aux_grads):
        if gb is None or ga is None:
            continue
        gb = gb.detach().float()
        ga = ga.detach().float()
        d = torch.sum(gb * ga)
        bs = torch.sum(gb * gb)
        aas = torch.sum(ga * ga)
        dot = d if dot is None else dot + d
        base_sq = bs if base_sq is None else base_sq + bs
        aux_sq = aas if aux_sq is None else aux_sq + aas

    if dot is None:
        return 1.0, 1.0, 0.0

    eps = 1.0e-12
    base_norm = torch.sqrt(torch.clamp(base_sq, min=eps))
    aux_norm = torch.sqrt(torch.clamp(aux_sq, min=eps))
    cosine = torch.clamp(dot / (base_norm * aux_norm + eps), -1.0, 1.0)
    norm_ratio = aux_norm / (base_norm + eps)

    cos_v = float(cosine.cpu())
    ratio_v = float(norm_ratio.cpu())

    low = float(conflict_low)
    high = float(conflict_high)
    if high <= low:
        raise ValueError(
            f"gradient_guard conflict_high must be > conflict_low, got {high} <= {low}"
        )

    if cos_v >= high:
        conflict_gate = 1.0
    elif cos_v <= low:
        conflict_gate = float(min_conflict_gate)
    else:
        alpha = (cos_v - low) / (high - low)
        conflict_gate = float(min_conflict_gate) + alpha * (
            1.0 - float(min_conflict_gate)
        )

    norm_gate = (
        1.0 if ratio_v <= 0.0 else min(1.0, float(max_norm_ratio) / ratio_v)
    )
    gate = max(float(min_total_gate), min(conflict_gate, norm_gate))
    return min(1.0, gate), cos_v, ratio_v


def episode_has_positive_reward(episode, positive_threshold=1.0e-6):
    reward = episode.get("reward", None)
    if reward is None:
        return False
    return bool(np.any(np.asarray(reward) > float(positive_threshold)))


def choose_replay_episode_index(
    episode_values,
    lengths,
    np_random,
    success_replay_ratio=0.0,
    success_positive_threshold=1.0e-6,
):
    """Choose one replay episode index using the Stable-V1.2 mixture."""
    lengths = np.asarray(lengths, dtype=np.float64)
    p_all = lengths / np.sum(lengths)
    ratio = float(np.clip(success_replay_ratio, 0.0, 1.0))
    if ratio <= 0.0:
        episode_index = int(np_random.choice(np.arange(len(episode_values)), p=p_all))
        return episode_index, False

    success_indices = [
        i
        for i, episode in enumerate(episode_values)
        if episode_has_positive_reward(episode, success_positive_threshold)
    ]
    draw_success = bool(success_indices) and np_random.rand() < ratio
    if draw_success:
        success_lengths = lengths[success_indices]
        p_success = success_lengths / np.sum(success_lengths)
        local_index = int(
            np_random.choice(np.arange(len(success_indices)), p=p_success)
        )
        return int(success_indices[local_index]), True
    episode_index = int(np_random.choice(np.arange(len(episode_values)), p=p_all))
    return episode_index, False
