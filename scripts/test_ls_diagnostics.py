#!/usr/bin/env python3
"""Dependency-light invariance test for detached LS-Imagine diagnostics."""
import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from ls_diagnostics import LSImagineDiagnostics


class DummyDynamics(nn.Module):
    def __init__(self, actions=4, stoch=4, discrete=4, deter=8):
        super().__init__()
        self._discrete = discrete
        self._stoch = stoch
        self._actions = actions
        self._deter = deter
        self.lin = nn.Linear(deter + stoch * discrete + actions, deter)
        self.stat = nn.Linear(deter, stoch * discrete)

    def img_step(self, prev, action, sample=False):
        probs = torch.softmax(prev["logit"], -1).reshape(prev["deter"].shape[0], -1)
        deter = torch.tanh(self.lin(torch.cat([prev["deter"], probs, action], -1)))
        logit = self.stat(deter).reshape(deter.shape[0], self._stoch, self._discrete)
        stoch = F.one_hot(logit.argmax(-1), self._discrete).float()
        return {"deter": deter, "logit": logit, "stoch": stoch}

    def get_feat(self, state):
        stoch = state["stoch"].reshape(state["stoch"].shape[:-2] + (-1,))
        return torch.cat([stoch, state["deter"]], -1)


class _Dist:
    def __init__(self, value):
        self.value = value

    def mode(self):
        return self.value


class DummyRewardHead(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.linear = nn.Linear(feat_dim, 1)

    def forward(self, x):
        return _Dist(self.linear(x))


def main():
    batch, time, actions, stoch, discrete, deter = 3, 24, 4, 4, 4, 8
    dynamics = DummyDynamics(actions, stoch, discrete, deter)
    feat_dim = stoch * discrete + deter
    reward_head = DummyRewardHead(feat_dim)

    ids = torch.arange(batch * time).reshape(batch, time) % actions
    action = F.one_hot(ids, actions).float()
    is_first = torch.zeros(batch, time, dtype=torch.bool)
    is_first[:, 0] = True
    logit = torch.randn(batch, time, stoch, discrete)
    posterior = {
        "logit": logit,
        "stoch": F.one_hot(logit.argmax(-1), discrete).float(),
        "deter": torch.randn(batch, time, deter),
    }
    reward = torch.zeros(batch, time)
    reward[:, 7] = 1.0
    data = {
        "action": action,
        "is_first": is_first,
        "reward": reward,
        "heatmap": torch.randint(0, 256, (batch, time, 8, 8, 1), dtype=torch.uint8),
        "score": torch.linspace(0, 1, batch * time).reshape(batch, time, 1),
    }

    diag = LSImagineDiagnostics(feat_dim, projection_dim=32, task_probe=True)
    rng_before = torch.random.get_rng_state().clone()
    metrics = diag(dynamics, reward_head, posterior, data)
    rng_after = torch.random.get_rng_state().clone()
    assert torch.equal(rng_before, rng_after), "diagnostics changed global RNG"

    required = (
        "ls_multistep_cosine_h15",
        "ls_cf_margin_shuffle_h15",
        "ls_return_corr_h4",
        "ls_return_mae_h8",
        "ls_task_probe_acc",
        "ls_reanchor_gain_h15",
    )
    for key in required:
        assert key in metrics, key
    print("PASS: detached LS-Imagine diagnostics")
    print("PASS: global torch RNG unchanged")
    for key in required:
        print(f"{key}: {float(metrics[key]):.6f}")


if __name__ == "__main__":
    main()
