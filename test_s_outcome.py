import copy
import torch
from torch import nn

from s_multistep_consistency import SOnlyMultiStepRSSMConsistency


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


def make_batch(B=3, T=9, A=3, D=3):
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
    # Sparse environment reward, aligned with transition into state t.
    rewards = torch.zeros(B, T)
    rewards[0, 4] = 1.0
    rewards[1, 6] = 2.0
    rewards[2, 8] = 1.0
    return post, actions, is_first, rewards


def build(outcome_enabled):
    return SOnlyMultiStepRSSMConsistency(
        feat_dim=6,
        horizons=[1, 2, 4],
        horizon_weights=[1.0, 1.0, 0.5],
        projection_dim=16,
        starts_per_sequence=2,
        projection_seed=314159,
        diagnostics_enabled=True,
        counterfactual_diagnostics=True,
        direct_vs_composed_diagnostics=True,
        direct_vs_composed_horizon=4,
        direct_vs_composed_midpoint=2,
        outcome_enabled=outcome_enabled,
        outcome_hidden_dim=16,
        outcome_init_seed=161803,
        outcome_discount=0.997,
        outcome_positive_weight=1.0,
    )


post, actions, is_first, rewards = make_batch()

# 1) Constructing the S-Outcome head must not perturb global CPU RNG.
torch.manual_seed(2026)
rng_before = torch.get_rng_state().clone()
mod_out = build(True)
rng_after = torch.get_rng_state().clone()
assert torch.equal(rng_before, rng_after), "S-Outcome construction perturbed global RNG"

# 2) Existing S5A stochastic rollout must remain identical when S-Outcome is enabled.
mod_base = build(False)
dyn_base = DummyDynamics()
dyn_out = copy.deepcopy(dyn_base)

torch.manual_seed(98765)
state0 = torch.get_rng_state().clone()
loss_base, out_base, _ = mod_base(dyn_base, post, actions, is_first, rewards=rewards)
rng_base = torch.get_rng_state().clone()

torch.set_rng_state(state0)
loss_s5a, loss_out, metrics = mod_out(dyn_out, post, actions, is_first, rewards=rewards)
rng_out = torch.get_rng_state().clone()

assert torch.equal(rng_base, rng_out), "S-Outcome changed S5A sampling RNG consumption"
assert torch.allclose(loss_base, loss_s5a, atol=0.0, rtol=0.0), (loss_base, loss_s5a)
assert float(out_base) == 0.0
assert loss_out.requires_grad and float(loss_out.detach()) >= 0.0

# 3) Outcome loss must train both its head and the existing S transition.
total = 0.05 * loss_s5a + 0.01 * loss_out
total.backward()
assert dyn_out.scale.grad is not None and torch.isfinite(dyn_out.scale.grad)
outcome_grads = [p.grad for p in mod_out.outcome_head.parameters()]
assert all(g is not None and torch.isfinite(g).all() for g in outcome_grads)

# 4) Required metrics must exist and remain detached.
required = [
    "s_outcome_loss",
    "s_outcome_loss_h1",
    "s_outcome_loss_h4",
    "s_outcome_target_return_mean_h4",
    "s_outcome_target_return_std_h4",
    "s_outcome_target_nonzero_frac_h4",
    "s_outcome_pred_return_mean_h4",
    "s_outcome_mae_h4",
    "s_outcome_corr_h4",
    "s_cf_margin_shuffle_h4",
    "s_dvc_reanchor_gain_h4",
]
for key in required:
    assert key in metrics, key
    assert not metrics[key].requires_grad, key

print("PASS: S-Outcome initialization preserves global RNG")
print("PASS: S5A loss/sampling path unchanged by enabling S-Outcome")
print("PASS: S-Outcome gradients reach both outcome head and S dynamics")
print("PASS: S-Outcome + existing diagnostics metrics are present")
for key in required[:9]:
    print(key, float(metrics[key]))
