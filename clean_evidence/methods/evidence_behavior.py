"""Native behavior learning with broadcast read-only evidence snapshots.

The sampled actions, joint transitions, feature/target timing, actor and critic
losses are unchanged. Only the native scan's repeated materialization of fixed
memory is avoided: each time index views the same immutable start evidence.
"""
import torch

from ..vendor.dreamerv3_torch.models import ImagBehavior
from ..vendor.dreamerv3_torch import tools


class EvidenceImagBehavior(ImagBehavior):
    def _train(self, start, objective):
        result = super()._train(start, objective)
        # Inspect the very same imagined states after the native update. World
        # model heads are unchanged by actor/critic optimization, and no new
        # trajectories or RNG draws are introduced by this diagnostic.
        _, states, _, _, metrics = result
        with torch.no_grad():
            feature = self._world_model.dynamics.get_feat(states)
            task = self._world_model.heads['task_reward'](feature).mode()
            metrics.update(tools.tensorstats(task, 'imag_task_reward'))
            if 'clip_reward' in self._world_model.heads:
                clip = self._world_model.heads['clip_reward'](feature).mode()
                metrics.update(tools.tensorstats(clip, 'imag_clip_reward'))
                metrics.update(tools.tensorstats(
                    self._world_model.options['clip_scale'] * clip,
                    'imag_scaled_clip_reward'))
        return result

    def _imagine(self, start, policy, horizon):
        dynamics = self._world_model.dynamics
        start = {key: value.reshape([-1] + list(value.shape[2:]))
                 for key, value in start.items()}
        if start['deter'].shape[0] == 0:
            feature = dynamics.get_feat(start)
            return (feature.new_empty((horizon, 0, feature.shape[-1])),
                    {key: value.new_empty((horizon,) + value.shape)
                     for key, value in start.items()},
                    feature.new_empty((horizon, 0, self._config.num_actions)))
        readonly = ('mem_tokens', 'mem_valid', 'mem_age')
        traces = {key: [] for key in start if key not in readonly}
        features, actions = [], []
        state = start
        for _ in range(horizon):
            for key in traces:
                traces[key].append(state[key])
            feature = dynamics.get_feat(state)
            action = policy(feature.detach()).sample()
            features.append(feature)
            actions.append(action)
            state = dynamics.img_step(state, action)
        states = {key: torch.stack(values, 0) for key, values in traces.items()}
        for key in readonly:
            states[key] = start[key].unsqueeze(0).expand((horizon,) + start[key].shape)
        return torch.stack(features, 0), states, torch.stack(actions, 0)
