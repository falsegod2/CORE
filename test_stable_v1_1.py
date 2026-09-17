import torch
from s_multistep_consistency import SOnlyMultiStepRSSMConsistency

mod = SOnlyMultiStepRSSMConsistency(
    feat_dim=6,
    horizons=[1, 2, 4, 8, 15],
    horizon_weights=[1.0, 1.0, 0.75, 0.5, 0.25],
    projection_dim=8,
    starts_per_sequence=1,
    diagnostics_enabled=False,
    prototype_enabled=True,
    prototype_ema=0.98,
    prototype_success_min_support=3,
    prototype_reliability_gate_enabled=True,
    prototype_reliability_ema=0.90,
    prototype_reliability_low=0.30,
    prototype_reliability_high=0.60,
    prototype_reliability_min=0.0,
    prototype_reliability_semantic_weight=0.65,
    prototype_reliability_affordance_weight=0.30,
    prototype_success_reliability_min_gate=0.10,
    prototype_mask_pseudo_bank_when_gate_zero=True,
)

# Accumulate enough hard Success support first. Reliability should still keep
# Success inactive while the teacher discrimination is weak.
features = torch.randn(3, 8)
labels = torch.tensor([3, 3, 3])
mod.prototype_bank.update(features, labels)
assert mod.prototype_bank.support_count[3].item() >= 3

weak = {
    's_proto_teacher_semantic_span': torch.tensor(0.30),
    's_proto_teacher_affordance_span': torch.tensor(0.20),
}
last = None
for _ in range(100):
    last = mod._prototype_reliability_metrics(weak, torch.device('cpu'))
assert last is not None
assert last['s_proto_reliability_score_batch'].item() < 0.30
assert last['s_proto_reliability_gate'].item() == 0.0
assert not mod.prototype_bank.active_mask()[3].item()
print('PASS: weak teacher keeps Proto gate closed even with enough Success support')

# Strong discrimination should raise the zero-start EMA and eventually open the
# Proto gate. The same reliability condition then allows Success into logits.
strong = {
    's_proto_teacher_semantic_span': torch.tensor(0.90),
    's_proto_teacher_affordance_span': torch.tensor(0.80),
}
for _ in range(80):
    last = mod._prototype_reliability_metrics(strong, torch.device('cpu'))
assert last['s_proto_reliability_gate'].item() > 0.10
assert mod.prototype_bank.active_mask()[3].item()
print('PASS: strong teacher gradually opens Proto gate and Success prototype')

# Gate is bounded and monotonic with the reliability score mapping.
assert 0.0 <= last['s_proto_reliability_gate'].item() <= 1.0
assert last['s_proto_reliability_score_ema'].item() > 0.30
print('PASS: Stable V1.1 Proto reliability gate checks completed')

# End-to-end check: weak evidence must zero the actual Proto loss and protect the
# pseudo prototype bank; strong evidence must open both without changing S5A.
class DummyDynamics(torch.nn.Module):
    def __init__(self, state_dim=3):
        super().__init__()
        self._discrete = False
        self.state_dim = state_dim
        self.scale = torch.nn.Parameter(torch.tensor(0.2))

    def img_step_s(self, prev_state, prev_action, sample=True):
        deter = prev_state['deter_s'] + self.scale * prev_action[..., : self.state_dim]
        mean = deter
        return {
            'stoch_s': mean,
            'deter_s': deter,
            's_mean': mean,
            's_std': torch.ones_like(mean) * 0.1,
        }


def make_replay(B=4, T=12, A=3, D=3):
    torch.manual_seed(123)
    idx = torch.randint(0, A, (B, T))
    actions = torch.nn.functional.one_hot(idx, A).float()
    first = torch.zeros(B, T)
    first[:, 0] = 1.0
    deter = torch.zeros(B, T, D)
    for t in range(1, T):
        deter[:, t] = deter[:, t - 1] + 0.2 * actions[:, t, :D]
    posterior = {
        'stoch_s': deter.clone(),
        'deter_s': deter.clone(),
        's_mean': deter.clone(),
        's_std': torch.ones_like(deter) * 0.1,
    }
    rewards = torch.zeros(B, T)
    return posterior, actions, first, rewards

mod2 = SOnlyMultiStepRSSMConsistency(
    feat_dim=6,
    horizons=[1, 2, 4],
    horizon_weights=[1.0, 1.0, 0.5],
    projection_dim=16,
    starts_per_sequence=2,
    diagnostics_enabled=False,
    prototype_enabled=True,
    prototype_ema=0.98,
    prototype_success_min_support=3,
    prototype_quantile_mode='ema',
    prototype_quantile_ema=0.9,
    prototype_boundary_margin=0.10,
    prototype_reliability_gate_enabled=True,
    prototype_reliability_ema=0.5,
    prototype_reliability_low=0.30,
    prototype_reliability_high=0.60,
    prototype_reliability_semantic_weight=0.65,
    prototype_reliability_affordance_weight=0.30,
    prototype_success_reliability_min_gate=0.10,
    prototype_mask_pseudo_bank_when_gate_zero=True,
)
dyn = DummyDynamics()
posterior, actions, first, rewards = make_replay()
B, T = rewards.shape
weak_score = torch.linspace(0.0, 0.15, T).repeat(B, 1)
weak_heat = weak_score[:, :, None, None, None].repeat(1, 1, 4, 4, 1)
_, _, proto_weak, met_weak = mod2(
    dyn, posterior, actions, first, rewards=rewards,
    task_scores=weak_score, heatmaps=weak_heat,
)
assert met_weak['s_proto_reliability_gate'].item() == 0.0
assert met_weak['s_proto_pseudo_bank_update_enabled'].item() == 0.0
assert proto_weak.item() == 0.0
assert mod2.prototype_bank.support_count[:3].sum().item() == 0
print('PASS: gate=0 blocks weak pseudo-bank updates and Proto gradient loss')

strong_score = torch.linspace(0.0, 1.0, T).repeat(B, 1)
strong_heat = strong_score[:, :, None, None, None].repeat(1, 1, 4, 4, 1)
proto_strong = None
met_strong = None
for _ in range(12):
    _, _, proto_strong, met_strong = mod2(
        dyn, posterior, actions, first, rewards=rewards,
        task_scores=strong_score, heatmaps=strong_heat,
    )
    if mod2.prototype_bank.support_count[:3].sum().item() > 0:
        break
assert met_strong is not None and proto_strong is not None
assert met_strong['s_proto_reliability_gate'].item() > 0.0
assert met_strong['s_proto_pseudo_bank_update_enabled'].item() == 1.0
assert mod2.prototype_bank.support_count[:3].sum().item() > 0
assert met_strong['s_proto_loss_post_gate'].item() <= met_strong['s_proto_loss_pre_gate'].item() + 1e-6
assert proto_strong.item() >= 0.0
print('PASS: reliable teacher opens bank updates and scales actual Proto loss')
