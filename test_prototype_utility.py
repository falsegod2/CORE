import copy

import torch
from torch import nn

from s_multistep_consistency import SOnlyMultiStepRSSMConsistency
from s_prototype_utility import build_progress_labels


class DummyDynamics(nn.Module):
    def __init__(self, action_dim=3, state_dim=3):
        super().__init__()
        self._discrete = False
        self.scale = nn.Parameter(torch.tensor(0.25))
        self.action_dim = action_dim
        self.state_dim = state_dim

    def img_step_s(self, prev_state, prev_action, sample=True):
        deter = prev_state["deter_s"] + self.scale * prev_action[..., : self.state_dim]
        noise = 0.01 * torch.randn_like(deter) if sample else torch.zeros_like(deter)
        mean = deter + noise
        return {
            "stoch_s": mean,
            "deter_s": deter,
            "s_mean": mean,
            "s_std": torch.ones_like(mean) * 0.1,
        }


def make_batch(B=3, T=18, A=3, D=3):
    torch.manual_seed(19)
    action_idx = torch.randint(0, A, (B, T))
    actions = torch.nn.functional.one_hot(action_idx, A).float()
    is_first = torch.zeros(B, T)
    is_first[:, 0] = 1.0

    deter = torch.zeros(B, T, D)
    mean = torch.zeros(B, T, D)
    for t in range(1, T):
        deter[:, t] = deter[:, t - 1] + 0.30 * actions[:, t, :D]
        mean[:, t] = deter[:, t]

    post = {
        "stoch_s": mean.clone(),
        "deter_s": deter.clone(),
        "s_mean": mean.clone(),
        "s_std": torch.ones_like(mean) * 0.1,
    }

    rewards = torch.zeros(B, T)
    rewards[0, 16] = 1.0
    rewards[1, 13] = 1.0
    # row 2 intentionally has no success -> Unknown labels
    return post, actions, is_first, rewards


def build(proto_enabled):
    return SOnlyMultiStepRSSMConsistency(
        feat_dim=6,
        horizons=[1, 2, 4],
        horizon_weights=[1.0, 1.0, 0.5],
        projection_dim=16,
        starts_per_sequence=3,
        projection_seed=314159,
        diagnostics_enabled=False,
        outcome_enabled=False,
        prototype_enabled=proto_enabled,
        prototype_temperature=0.10,
        prototype_ema=0.95,
        prototype_ready_steps=2,
        prototype_near_steps=6,
        prototype_progress_steps=None,
        prototype_positive_threshold=1.0e-6,
    )


post, actions, is_first, rewards = make_batch()

# 1) Label propagation must produce ordinal stages while leaving no-success
# trajectories unknown.
labels, dist = build_progress_labels(
    rewards,
    is_first=is_first,
    ready_steps=2,
    near_steps=6,
)
assert labels[0, 16].item() == 3
assert labels[0, 15].item() == 2
assert labels[0, 12].item() == 1
assert labels[0, 0].item() == 0
assert torch.all(labels[2] == -1)

# 2) Enabling prototype supervision must not change S5A stochastic RNG path.
mod_off = build(False)
mod_on = build(True)
dyn_off = DummyDynamics()
dyn_on = copy.deepcopy(dyn_off)

torch.manual_seed(98765)
rng0 = torch.get_rng_state().clone()
s5a_off, out_off, proto_off, _ = mod_off(
    dyn_off, post, actions, is_first, rewards=rewards
)
rng_after_off = torch.get_rng_state().clone()

torch.set_rng_state(rng0)
s5a_on, out_on, proto_on, metrics = mod_on(
    dyn_on, post, actions, is_first, rewards=rewards
)
rng_after_on = torch.get_rng_state().clone()

assert torch.equal(rng_after_off, rng_after_on), "Prototype path changed RNG consumption"
assert torch.allclose(s5a_off, s5a_on, atol=0.0, rtol=0.0)
assert float(out_off) == 0.0 and float(out_on) == 0.0
assert float(proto_off) == 0.0
assert proto_on.requires_grad and torch.isfinite(proto_on)

# 3) Prototype loss must reach the existing S transition.
(0.05 * s5a_on + 0.01 * proto_on).backward()
assert dyn_on.scale.grad is not None
assert torch.isfinite(dyn_on.scale.grad)

# 4) Support bank should initialize multiple task stages.
assert mod_on.prototype_bank is not None
assert int(mod_on.prototype_bank.initialized.sum().item()) >= 2

required = [
    "s_proto_loss",
    "s_proto_initialized",
    "s_proto_support_progress",
    "s_proto_support_near",
    "s_proto_support_ready",
    "s_proto_support_success",
    "s_proto_known_fraction",
    "s_proto_unknown_fraction",
    "s_proto_acc_h1",
    "s_proto_acc_h4",
    "s_proto_num_h4",
]
for key in required:
    assert key in metrics, key
    assert not metrics[key].requires_grad, key

print("PASS: ordinal prototype labels are generated from rare real success")
print("PASS: zero-reward/no-success trajectory stays Unknown, not negative")
print("PASS: Prototype Utility leaves the S5A RNG/sampling path unchanged")
print("PASS: Prototype loss backpropagates through predicted future S")
print("PASS: Prototype metrics are present")
print("prototype loss:", float(proto_on.detach()))
print("initialized:", int(mod_on.prototype_bank.initialized.sum()))
print("supports:", mod_on.prototype_bank.support_count.tolist())
