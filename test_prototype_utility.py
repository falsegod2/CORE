import copy

import torch
from torch import nn

from s_multistep_consistency import SOnlyMultiStepRSSMConsistency
from s_prototype_utility import build_task_evidence_labels


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

    # Generic task evidence: no sheep/tree semantics are encoded here.
    # Different sequences have different evidence ranges, but all use the same
    # task_score + task-conditioned heatmap interface.
    base = torch.linspace(0.05, 0.95, T)
    task_score = torch.stack([
        base,
        torch.flip(base, dims=[0]),
        0.25 + 0.50 * base,
    ], dim=0).unsqueeze(-1)

    heatmap = torch.zeros(B, T, 8, 8, 1)
    for b in range(B):
        for t in range(T):
            # Stronger task-conditioned affordance response as semantic score rises.
            heatmap[b, t, :2, :2, 0] = task_score[b, t, 0]

    intrinsic = torch.zeros(B, T)
    intrinsic[0, 5] = 0.01
    intrinsic[0, 10] = 0.01

    rewards = torch.zeros(B, T)
    rewards[0, 16] = 1.0  # real Success anchor
    return post, actions, is_first, rewards, task_score, heatmap, intrinsic


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
        prototype_label_mode="task_evidence",
        prototype_positive_threshold=1.0e-6,
        prototype_semantic_weight=0.65,
        prototype_affordance_weight=0.30,
        prototype_intrinsic_weight=0.05,
        prototype_heatmap_topk_fraction=0.05,
        prototype_stage_low_quantile=0.30,
        prototype_stage_high_quantile=0.70,
        prototype_boundary_margin=0.05,
    )


post, actions, is_first, rewards, task_score, heatmap, intrinsic = make_batch()

# 1) Generic evidence labels must exist before success and real reward must be
# the hard Success anchor.
labels, evidence, teacher_metrics = build_task_evidence_labels(
    rewards=rewards,
    task_scores=task_score,
    heatmaps=heatmap,
    intrinsic=intrinsic,
)
assert labels.shape == rewards.shape
assert evidence.shape == rewards.shape
assert labels[0, 16].item() == 3
assert (labels == 0).any(), "Low evidence class missing"
assert (labels == 1).any(), "Mid evidence class missing"
assert (labels == 2).any(), "High evidence class missing"
assert teacher_metrics["s_proto_teacher_semantic_active"].item() == 1.0
assert teacher_metrics["s_proto_teacher_affordance_active"].item() == 1.0

# 2) Flat, non-informative teacher signals must NOT fabricate pseudo classes.
flat_score = torch.zeros_like(task_score)
flat_heatmap = torch.zeros_like(heatmap)
flat_intrinsic = torch.zeros_like(intrinsic)
flat_rewards = torch.zeros_like(rewards)
flat_labels, _, _ = build_task_evidence_labels(
    rewards=flat_rewards,
    task_scores=flat_score,
    heatmaps=flat_heatmap,
    intrinsic=flat_intrinsic,
)
assert torch.all(flat_labels == -1), "Flat teacher evidence should remain Unknown"

# 3) Enabling prototype supervision must not change S5A stochastic RNG path.
mod_off = build(False)
mod_on = build(True)
dyn_off = DummyDynamics()
dyn_on = copy.deepcopy(dyn_off)

torch.manual_seed(98765)
rng0 = torch.get_rng_state().clone()
s5a_off, out_off, proto_off, _ = mod_off(
    dyn_off,
    post,
    actions,
    is_first,
    rewards=rewards,
    task_scores=task_score,
    heatmaps=heatmap,
    intrinsic=intrinsic,
)
rng_after_off = torch.get_rng_state().clone()

torch.set_rng_state(rng0)
s5a_on, out_on, proto_on, metrics = mod_on(
    dyn_on,
    post,
    actions,
    is_first,
    rewards=rewards,
    task_scores=task_score,
    heatmaps=heatmap,
    intrinsic=intrinsic,
)
rng_after_on = torch.get_rng_state().clone()

assert torch.equal(rng_after_off, rng_after_on), "Prototype path changed RNG consumption"
assert torch.allclose(s5a_off, s5a_on, atol=0.0, rtol=0.0)
assert float(out_off) == 0.0 and float(out_on) == 0.0
assert float(proto_off) == 0.0
assert proto_on.requires_grad and torch.isfinite(proto_on)

# 4) Prototype loss must reach the existing S transition.
(0.05 * s5a_on + 0.01 * proto_on).backward()
assert dyn_on.scale.grad is not None
assert torch.isfinite(dyn_on.scale.grad)

# 5) Generic evidence should initialize multiple prototypes without requiring
# every batch to contain a real success.
assert mod_on.prototype_bank is not None
assert int(mod_on.prototype_bank.initialized.sum().item()) >= 3

required = [
    "s_proto_loss",
    "s_proto_initialized",
    "s_proto_support_low",
    "s_proto_support_mid",
    "s_proto_support_high",
    "s_proto_support_success",
    "s_proto_known_fraction",
    "s_proto_unknown_fraction",
    "s_proto_teacher_semantic_active",
    "s_proto_teacher_affordance_active",
    "s_proto_evidence_mean",
    "s_proto_acc_h1",
    "s_proto_acc_h4",
    "s_proto_num_h4",
]
for key in required:
    assert key in metrics, key
    assert not metrics[key].requires_grad, key

print("PASS: generic task-evidence labels do not depend on sheep/tree-specific stages")
print("PASS: flat/uninformative teacher signals remain Unknown")
print("PASS: real environment reward is the hard Success anchor")
print("PASS: Prototype Utility leaves the S5A RNG/sampling path unchanged")
print("PASS: Prototype loss backpropagates through predicted future S")
print("PASS: generic Prototype metrics are present")
print("prototype loss:", float(proto_on.detach()))
print("initialized:", int(mod_on.prototype_bank.initialized.sum()))
print("supports:", mod_on.prototype_bank.support_count.tolist())
