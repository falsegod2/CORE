#!/usr/bin/env python3
"""CPU smoke test for Dreamer + Generic Proto on full RSSM latent."""

import torch
from torch import nn
import torch.nn.functional as F

from full_latent_prototype import FullLatentPrototypeUtility


class FakeDiscreteDreamerRSSM(nn.Module):
    def __init__(self, deter=12, stoch=3, discrete=4, actions=5):
        super().__init__()
        self._deter = deter
        self._stoch = stoch
        self._discrete = discrete
        self.action_to_deter = nn.Linear(actions, deter, bias=False)
        self.deter_to_deter = nn.Linear(deter, deter, bias=False)
        self.deter_to_logits = nn.Linear(deter, stoch * discrete, bias=False)

    def img_step(self, prev_state, prev_action, sample=True):
        deter = torch.tanh(
            self.deter_to_deter(prev_state["deter"])
            + self.action_to_deter(prev_action)
        )
        logit = self.deter_to_logits(deter).reshape(
            deter.shape[0], self._stoch, self._discrete
        )
        probs = torch.softmax(logit, dim=-1)
        # Deterministic soft state is sufficient for this smoke test; the
        # production Dreamer RSSM preserves its own native sampling behavior.
        stoch = probs
        return {"deter": deter, "logit": logit, "stoch": stoch}


def main():
    torch.manual_seed(7)
    batch, time, actions = 4, 24, 5
    deter, stoch, discrete = 12, 3, 4
    feat_dim = deter + stoch * discrete

    dynamics = FakeDiscreteDreamerRSSM(deter, stoch, discrete, actions)
    posterior = {
        "deter": torch.randn(batch, time, deter, requires_grad=True),
        "logit": torch.randn(batch, time, stoch, discrete, requires_grad=True),
        "stoch": torch.softmax(torch.randn(batch, time, stoch, discrete), dim=-1),
    }

    action_idx = torch.randint(0, actions, (batch, time))
    action = F.one_hot(action_idx, actions).float()
    is_first = torch.zeros(batch, time)
    is_first[:, 0] = 1.0
    rewards = torch.zeros(batch, time)
    rewards[0, 18] = 1.0
    rewards[2, 21] = 1.0

    # Dense, task-conditioned teacher signals with non-flat distributions.
    task_score = torch.linspace(0.02, 0.98, time)[None].repeat(batch, 1)
    task_score = (task_score + 0.03 * torch.randn_like(task_score)).clamp(0, 1)
    heatmap = torch.rand(batch, time, 8, 8, 1)
    heatmap = heatmap * (0.2 + 0.8 * task_score[..., None, None, None])
    intrinsic = torch.zeros(batch, time)
    intrinsic[:, 5] = 0.2
    intrinsic[:, 12] = 0.5

    proto = FullLatentPrototypeUtility(
        feat_dim=feat_dim,
        horizons=(1, 2, 4, 8, 15),
        horizon_weights=(1.0, 1.0, 0.75, 0.5, 0.25),
        projection_dim=32,
        starts_per_sequence=3,
        projection_seed=314159,
        diagnostics_enabled=True,
    )

    loss, metrics = proto(
        dynamics=dynamics,
        posterior=posterior,
        actions=action,
        is_first=is_first,
        rewards=rewards,
        task_scores=task_score,
        heatmaps=heatmap,
        intrinsic=intrinsic,
    )

    assert torch.isfinite(loss), loss
    assert float(metrics["proto_known_fraction"]) > 0.5
    assert float(metrics["proto_initialized"]) >= 3.0
    assert "proto_acc_h15" in metrics
    assert "proto_cf_margin_h15" in metrics
    assert "proto_response_gap_h15" in metrics
    assert all(not key.startswith("s_multistep_loss") for key in metrics)

    loss.backward()
    grad = dynamics.action_to_deter.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0

    print("PASS: full-latent Generic Proto loss is finite and trains native Dreamer dynamics.")
    print("proto_loss=", float(loss.detach()))
    print("known_fraction=", float(metrics["proto_known_fraction"]))
    print("initialized=", float(metrics["proto_initialized"]))
    print("proto_acc_h15=", float(metrics["proto_acc_h15"]))
    print("cf_margin_h15=", float(metrics["proto_cf_margin_h15"]))


if __name__ == "__main__":
    main()
