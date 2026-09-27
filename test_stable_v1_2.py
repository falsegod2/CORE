import numpy as np
import torch

from stable_v1_2_utils import gradient_guard_instant_gate, choose_replay_episode_index


def test_gradient_guard_aligned_and_conflicting():
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    base = (p ** 2).sum()
    aligned = 0.01 * (p ** 2).sum()
    gate_a, cos_a, ratio_a = gradient_guard_instant_gate(
        base, aligned, [p], max_norm_ratio=0.35
    )
    assert cos_a > 0.99, (gate_a, cos_a, ratio_a)
    assert gate_a > 0.99, (gate_a, cos_a, ratio_a)

    p2 = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    base2 = (p2 ** 2).sum()
    conflicting = -0.01 * (p2 ** 2).sum()
    gate_c, cos_c, ratio_c = gradient_guard_instant_gate(
        base2, conflicting, [p2], conflict_low=-0.2, min_conflict_gate=0.25
    )
    assert cos_c < -0.99, (gate_c, cos_c, ratio_c)
    assert 0.10 <= gate_c <= 0.25 + 1e-6, (gate_c, cos_c, ratio_c)
    print("PASS: Proto gradient guard preserves aligned gradients and attenuates conflicting gradients")


def fake_episode(length, success=False):
    reward = np.zeros((length,), dtype=np.float32)
    if success:
        reward[-2] = 1.0
    return {
        "reward": reward,
        "image": np.zeros((length, 2, 2, 3), dtype=np.uint8),
        "heatmap": np.zeros((length, 2, 2), dtype=np.uint8),
        "action": np.zeros((length, 3), dtype=np.float32),
        "is_first": np.zeros((length,), dtype=bool),
        "is_terminal": np.zeros((length,), dtype=bool),
    }


def test_success_replay_bias():
    episodes = [
        fake_episode(40, False),
        fake_episode(40, False),
        fake_episode(40, False),
        fake_episode(40, True),
    ]
    lengths = [40, 40, 40, 40]
    rng0 = np.random.RandomState(7)
    rng1 = np.random.RandomState(7)
    base_pri = []
    biased_pri = []
    for _ in range(2000):
        _, pri0 = choose_replay_episode_index(
            episodes, lengths, rng0, success_replay_ratio=0.0
        )
        _, pri1 = choose_replay_episode_index(
            episodes, lengths, rng1, success_replay_ratio=0.50
        )
        base_pri.append(float(pri0))
        biased_pri.append(float(pri1))
    assert np.mean(base_pri) == 0.0
    assert 0.45 < np.mean(biased_pri) < 0.55, np.mean(biased_pri)
    print("PASS: success-aware replay adds only the requested mixture probability")


if __name__ == "__main__":
    test_gradient_guard_aligned_and_conflicting()
    test_success_replay_bias()
    print("PASS: Stable V1.2 checks completed")
