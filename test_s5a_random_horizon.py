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
            noise = 0.01 * torch.randn_like(deter)
        else:
            noise = torch.zeros_like(deter)
        mean = deter + noise
        return {
            "stoch_s": mean,
            "deter_s": deter,
            "s_mean": mean,
            "s_std": torch.ones_like(mean) * 0.1,
        }


def make_batch(B=2, T=20, A=3, D=3):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(7)
    action_idx = torch.randint(0, A, (B, T), generator=gen)
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


def build(random_enabled):
    return SOnlyMultiStepRSSMConsistency(
        feat_dim=6,
        horizons=[1, 2, 4, 8, 15],
        horizon_weights=[1.0, 1.0, 0.75, 0.5, 0.25],
        projection_dim=16,
        starts_per_sequence=2,
        projection_seed=314159,
        random_horizon_enabled=random_enabled,
        random_horizon_count=3,
        random_horizon_seed=271828,
        random_horizon_unbiased_reweight=True,
        diagnostics_enabled=True,
        counterfactual_diagnostics=True,
        direct_vs_composed_diagnostics=True,
        direct_vs_composed_horizon=15,
        direct_vs_composed_midpoint=7,
    )


post, actions, is_first = make_batch()
dyn_fixed = DummyDynamics()
dyn_random = copy.deepcopy(dyn_fixed)
mod_fixed = build(False)
mod_random = build(True)

# RandomHorizon selection itself must consume no global RNG.  Since both versions
# execute the same stochastic rollout to h=15, post-forward RNG states must match.
torch.manual_seed(12345)
rng0 = torch.get_rng_state().clone()
loss_fixed, metrics_fixed = mod_fixed(dyn_fixed, post, actions, is_first)
rng_after_fixed = torch.get_rng_state().clone()

torch.set_rng_state(rng0)
loss_random, metrics_random = mod_random(dyn_random, post, actions, is_first)
rng_after_random = torch.get_rng_state().clone()
assert torch.equal(rng_after_fixed, rng_after_random), "RandomHorizon changed global torch RNG"

# Every legacy horizon metric remains present and exactly comparable because the
# stochastic rollout and metric computation are unchanged.
for h in [1, 2, 4, 8, 15]:
    for stem in [
        "s_multistep_loss_h",
        "s_multistep_cosine_h",
        "s_multistep_valid_h",
        "s_multistep_target_raw_std_h",
    ]:
        key = f"{stem}{h}"
        assert key in metrics_random, key
        assert torch.allclose(metrics_fixed[key], metrics_random[key], atol=0.0, rtol=0.0), key
    assert f"s_random_horizon_selected_h{h}" in metrics_random

# Backward-compatible aggregate remains the full five-horizon metric.
assert torch.allclose(metrics_fixed["s_multistep_loss"], metrics_random["s_multistep_loss"], atol=0.0, rtol=0.0)
assert torch.allclose(loss_random.detach(), metrics_random["s_multistep_train_loss"], atol=0.0, rtol=0.0)

# Verify the exact unbiased reweighted training estimator for this update.
weights = {1: 1.0, 2: 1.0, 4: 0.75, 8: 0.5, 15: 0.25}
p = float(metrics_random["s_random_horizon_inclusion_prob"])
manual = 0.0
for h, w in weights.items():
    selected = float(metrics_random[f"s_random_horizon_selected_h{h}"])
    manual += selected / p * w * float(metrics_random[f"s_multistep_loss_h{h}"])
manual /= sum(weights.values())
assert abs(float(loss_random.detach()) - manual) < 2e-6, (loss_random, manual)
assert int(metrics_random["s_random_horizon_selected_count"].item()) == 3

# All previous Counterfactual + DVC metrics remain present.
required_diag = [
    "s_cf_margin_shuffle_h1",
    "s_cf_margin_shuffle_h15",
    "s_cf_response_gap_shuffle_h15",
    "s_cf_response_gap_zero_h15",
    "s_dvc_reanchored_consistency_cos_h15",
    "s_dvc_reanchored_gap_h15",
    "s_dvc_reanchor_gain_h15",
]
for key in required_diag:
    assert key in metrics_random, key
    assert not metrics_random[key].requires_grad, key

# One full C(5,3)=10 schedule cycle must cover every horizon exactly six times.
selector = build(True)
coverage = torch.zeros(5)
seen = set()
for _ in range(10):
    mask, prob, _ = selector._random_horizon_selection(torch.device("cpu"))
    coverage += mask.cpu()
    seen.add(tuple(int(x) for x in mask.tolist()))
assert len(seen) == 10, len(seen)
assert torch.equal(coverage, torch.full((5,), 6.0)), coverage
assert abs(float(prob) - 0.6) < 1e-6

# Persistent selector counter makes checkpoint resume continue exactly.
resume_src = build(True)
for _ in range(4):
    resume_src._random_horizon_selection(torch.device("cpu"))
state = {k: v.detach().clone() for k, v in resume_src.state_dict().items()}
mask_expected, _, step_expected = resume_src._random_horizon_selection(torch.device("cpu"))
resume_dst = build(True)
resume_dst.load_state_dict(state)
mask_loaded, _, step_loaded = resume_dst._random_horizon_selection(torch.device("cpu"))
assert torch.equal(mask_expected, mask_loaded)
assert torch.equal(step_expected, step_loaded)

print("PASS: RandomHorizon preserves global torch RNG state")
print("PASS: all legacy S5A metrics are preserved")
print("PASS: Counterfactual + DVC diagnostics are preserved")
print("PASS: 3/5 balanced schedule covers each horizon 6/10 updates")
print("PASS: checkpoint resume preserves selector position")
print("fixed full loss:", float(loss_fixed.detach()))
print("random train loss:", float(loss_random.detach()))
print("full diagnostic loss:", float(metrics_random["s_multistep_loss"]))
print("selected:", {h: int(metrics_random[f"s_random_horizon_selected_h{h}"].item()) for h in weights})
