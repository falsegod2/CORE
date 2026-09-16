import torch
from torch import nn

from s_prototype_utility import EMATaskEvidenceTeacher, PrototypeUtilityBank
from s_multistep_consistency import SOnlyMultiStepRSSMConsistency


# 1) EMA teacher should expose both batch and EMA boundaries and keep real success hard anchored.
teacher = EMATaskEvidenceTeacher(quantile_ema=0.9, boundary_margin=0.10)
B, T = 2, 16
rewards = torch.zeros(B, T)
rewards[0, -1] = 1.0
score1 = torch.linspace(0.0, 1.0, T).repeat(B, 1)
heat1 = score1[:, :, None, None, None].repeat(1, 1, 4, 4, 1)
labels1, _, m1 = teacher(rewards, task_scores=score1, heatmaps=heat1)
assert labels1[0, -1].item() == 3
assert m1["s_proto_teacher_ema_enabled"].item() == 1.0
assert "s_proto_stage_batch_low_threshold" in m1
assert "s_proto_stage_low_threshold" in m1

# Shift the batch scale; EMA boundary should not jump exactly to the new batch boundary.
score2 = score1 + 10.0
heat2 = heat1 + 10.0
_, _, m2 = teacher(rewards, task_scores=score2, heatmaps=heat2)
assert not torch.allclose(
    m2["s_proto_teacher_semantic_ema_qlo"], m2["s_proto_teacher_semantic_qlo"]
)
print("PASS: EMA task-evidence teacher keeps smoothed quantile boundaries")


# 2) Success prototype must stay inactive until minimum support is reached.
bank = PrototypeUtilityBank(feature_dim=4, success_min_support=3)
feat = torch.eye(4)
labels = torch.tensor([0, 1, 2, 3])
bank.update(feat, labels)
assert bank.initialized[3].item()
assert not bank.active_mask()[3].item()
bank.update(feat[3:4].repeat(2, 1), torch.tensor([3, 3]))
assert bank.active_mask()[3].item()
assert bank.metrics()["s_proto_success_ready"].item() == 1.0
print("PASS: Success prototype cold-start protection works")


class DummyDynamics(nn.Module):
    def __init__(self, action_dim=3, state_dim=3):
        super().__init__()
        self._discrete = False
        self.scale = nn.Parameter(torch.tensor(0.25))
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


def make_batch(with_positive=False, B=4, T=12, A=3, D=3):
    torch.manual_seed(7)
    idx = torch.randint(0, A, (B, T))
    actions = torch.nn.functional.one_hot(idx, A).float()
    first = torch.zeros(B, T)
    first[:, 0] = 1.0
    deter = torch.zeros(B, T, D)
    for t in range(1, T):
        deter[:, t] = deter[:, t - 1] + 0.2 * actions[:, t, :D]
    post = {
        "stoch_s": deter.clone(),
        "deter_s": deter.clone(),
        "s_mean": deter.clone(),
        "s_std": torch.ones_like(deter) * 0.1,
    }
    rewards = torch.zeros(B, T)
    if with_positive:
        rewards[:, 6] = 1.0
    return post, actions, first, rewards


# 3) Outcome rollout gate should be zero/low without evidence and rise after positives.
mod = SOnlyMultiStepRSSMConsistency(
    feat_dim=6,
    horizons=[1, 2, 4],
    horizon_weights=[1.0, 1.0, 0.5],
    projection_dim=16,
    starts_per_sequence=2,
    diagnostics_enabled=False,
    outcome_enabled=True,
    outcome_hidden_dim=16,
    outcome_calibration_enabled=True,
    outcome_calibration_scale=0.5,
    outcome_gate_enabled=True,
    outcome_gate_ema=0.5,
    outcome_gate_low=0.01,
    outcome_gate_high=0.10,
    outcome_gate_horizon=4,
    outcome_gate_min=0.0,
)
dyn = DummyDynamics()
post, actions, first, rewards = make_batch(False)
_, out0, _, met0 = mod(dyn, post, actions, first, rewards=rewards)
assert met0["s_outcome_gate"].item() == 0.0
post, actions, first, rewards = make_batch(True)
met1 = None
out1 = None
for _ in range(8):
    _, out1, _, met1 = mod(dyn, post, actions, first, rewards=rewards)
assert met1 is not None and met1["s_outcome_gate"].item() > 0.0
assert "s_outcome_rollout_loss_pre_gate" in met1
assert "s_outcome_rollout_loss_post_gate" in met1
assert torch.isfinite(out1)
print("PASS: Outcome evidence gate opens as positive-return evidence accumulates")

print("PASS: Stable V1 checks completed")
