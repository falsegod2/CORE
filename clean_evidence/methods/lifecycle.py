"""Replay lifecycle and exact learner checkpoint state; no environment or runner."""
import copy
import random
import numpy as np


class EvidenceLifecycle:

    def begin_episode(self, obs):
        self.replay.begin(obs)

    def record(self, previous, action, current, extras):
        self.replay.add(action, current)

    def finish_episode(self):
        self.replay.finish()

    def ready(self):
        return self.replay.ready()

    def state_dict(self):
        t = self.torch
        optimizers = (self.wm._model_opt, self.behavior._actor_opt, self.behavior._value_opt)
        state = {
            'world_model': self.wm.state_dict(),
            # ImagBehavior registers its reference to the world model as a child
            # module. Store that model once, rather than doubling a large snapshot.
            'behavior': {k: v for k, v in self.behavior.state_dict().items()
                         if not k.startswith('_world_model.')},
            'optimizers': [{'optimizer': opt._opt.state_dict(), 'scaler': opt._scaler.state_dict()}
                           for opt in optimizers],
            'behavior_updates': getattr(self.behavior, '_updates', 0), 'updates': self.updates,
            'torch_rng': t.get_rng_state(),
            'cuda_rng': t.cuda.get_rng_state_all() if t.cuda.is_available() else None,
            'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
        }

        def cpu(value):
            if isinstance(value, t.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {k: cpu(v) for k, v in value.items()}
            if isinstance(value, tuple):
                return tuple(cpu(v) for v in value)
            if isinstance(value, list):
                return [cpu(v) for v in value]
            return copy.deepcopy(value)

        state = cpu(state)
        state['replay'] = self.replay.state_dict()
        if hasattr(self, 'diagnostic_probes'):
            state['diagnostic_probes'] = self.diagnostic_probes.state_dict()
        return state

    def load_state_dict(self, state):
        self.wm.load_state_dict(state['world_model'])
        result = self.behavior.load_state_dict(state['behavior'], strict=False)
        missing = [key for key in result.missing_keys if not key.startswith('_world_model.')]
        if missing or result.unexpected_keys:
            raise ValueError(f'Behavior checkpoint mismatch: missing={missing}, unexpected={result.unexpected_keys}')
        optimizers = (self.wm._model_opt, self.behavior._actor_opt, self.behavior._value_opt)
        for opt, saved in zip(optimizers, state['optimizers']):
            # On the same device PyTorch may retain references to state tensors;
            # copying prevents a later update from mutating the saved snapshot.
            opt._opt.load_state_dict(copy.deepcopy(saved['optimizer']))
            opt._scaler.load_state_dict(copy.deepcopy(saved['scaler']))
        self.behavior._updates = int(state['behavior_updates'])
        self.updates = int(state['updates'])
        self.replay.load_state_dict(state['replay'])
        if hasattr(self, 'diagnostic_probes') and 'diagnostic_probes' in state:
            self.diagnostic_probes.load_state_dict(state['diagnostic_probes'])
        self.torch.set_rng_state(state['torch_rng'])
        if state['cuda_rng'] is not None and self.torch.cuda.is_available():
            if len(state['cuda_rng']) != self.torch.cuda.device_count():
                raise ValueError('Saved CUDA RNG device count differs; exact learner recovery is unavailable')
            self.torch.cuda.set_rng_state_all(state['cuda_rng'])
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])

    def should_finish(self, obs, training=True):
        return bool(obs['done'])

    def flush_metrics(self):
        return {}

    def close(self):
        pass
