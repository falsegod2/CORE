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
        if sample:
            # Mimic stochastic RSSM sampling so RNG-invariance is actually tested.
            noise = 0.01 * torch.randn_like(deter)
        else:
            noise = torch.zeros_like(deter)
        mean = deter + noise
        stoch = mean
        std = torch.ones_like(mean) * 0.1
        return {
            "stoch_s": stoch,
            "deter_s": deter,
            "s_mean": mean,
            "s_std": std,
        }


def make_batch(B=2, T=8, A=3, D=3):
    torch.manual_seed(7)
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
    return post, actions, is_first


def build(enabled):
    return SOnlyMultiStepRSSMConsistency(
        feat_dim=6,
        horizons=[1, 2, 4],
        horizon_weights=[1.0, 1.0, 0.5],
        projection_dim=16,
        starts_per_sequence=2,
        projection_seed=314159,
        diagnostics_enabled=enabled,
        counterfactual_diagnostics=True,
        direct_vs_composed_diagnostics=True,
        direct_vs_composed_horizon=4,
        direct_vs_composed_midpoint=2,
    )


post, actions, is_first = make_batch()
dyn_off = DummyDynamics()
dyn_on = copy.deepcopy(dyn_off)
mod_off = build(False)
mod_on = build(True)

# Use the exact same initial RNG state so the stochastic *training* rollout is identical.
torch.manual_seed(12345)
rng0 = torch.get_rng_state().clone()
loss_off, outcome_off, proto_off, metrics_off = mod_off(dyn_off, post, actions, is_first)
loss_off.backward()
rng_after_off = torch.get_rng_state().clone()
grad_off = dyn_off.scale.grad.detach().clone()

# Reset RNG to the exact pre-forward state and run diagnostics-enabled variant.
torch.set_rng_state(rng0)
loss_on, outcome_on, proto_on, metrics_on = mod_on(dyn_on, post, actions, is_first)
loss_on.backward()
rng_after_on = torch.get_rng_state().clone()
grad_on = dyn_on.scale.grad.detach().clone()

assert torch.equal(rng_after_off, rng_after_on), "Diagnostics changed global torch RNG state"
assert torch.allclose(loss_off, loss_on, atol=0.0, rtol=0.0), (loss_off, loss_on)
assert float(outcome_off) == 0.0 and float(outcome_on) == 0.0
assert float(proto_off) == 0.0 and float(proto_on) == 0.0
assert torch.allclose(grad_off, grad_on, atol=0.0, rtol=0.0), (grad_off, grad_on)

required = [
    "s_cf_margin_shuffle_h1",
    "s_cf_margin_shuffle_h4",
    "s_cf_response_gap_zero_h4",
    "s_dvc_reanchored_consistency_cos_h4",
    "s_dvc_reanchor_gain_h4",
]
for key in required:
    assert key in metrics_on, key
    assert not metrics_on[key].requires_grad, key

print("PASS: training loss unchanged:", float(loss_on.detach()))
print("PASS: training gradient unchanged:", float(grad_on))
print("PASS: global RNG state unchanged by diagnostics")
for key in required:
    print(key, float(metrics_on[key]))
