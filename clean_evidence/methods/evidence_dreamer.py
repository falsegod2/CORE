"""Read-only task-evidence world model on the supplied Clean LS environment.

This is a new named method, not an official Dreamer implementation. The native
actor--critic is reused; its imagination calls the same joint prior as recorded
action FC/RC training. EvidenceWorldModel is built directly, including every
trainable head before its model optimizer is created.
"""
from __future__ import annotations

import contextlib
import copy
from types import SimpleNamespace

import numpy as np

from ..model_utils import ROOT, merge, read_yaml
from ..rng import isolated_rng, seed_for
from .lifecycle import EvidenceLifecycle
from .evidence_replay import EvidenceReplay


class EvidenceDreamer(EvidenceLifecycle):
    needs_heatmap = True
    needs_jump_labels = False

    def __init__(self, config, run_dir):
        import torch
        from ..vendor.dreamerv3_torch import tools
        from .evidence_behavior import EvidenceImagBehavior
        from .evidence_model import EvidenceWorldModel

        self.torch, self.tools = torch, tools
        self.c, self.run_dir = config, run_dir
        self.needs_heatmap = bool(config.get('needs_heatmap', True))
        dreamer = config['dreamer']
        self.options = copy.deepcopy(config.get('evidence', {}))
        # Branch allocation uses its own initialization stream derived from the
        # experiment seed, rather than the model constructor's default seed=0.
        self.options['seed'] = seed_for(config['seed'], 'evidence_dynamics')
        if dreamer.get('loss_plugins') or dreamer.get('train_overrides'):
            raise ValueError('EvidenceDreamer uses native dreamer.overrides and explicit FC/RC; '
                             'loss_plugins and JAX train_overrides are unsupported')
        raw = read_yaml(ROOT / 'vendor/dreamerv3_torch/configs.yaml')['defaults']
        native = merge(raw, {
            'actor': {'dist': 'onehot', 'std': 'none'},
            'imag_gradient': 'reinforce', 'expl_behavior': 'greedy',
            'encoder': {'cnn_keys': '^image$', 'mlp_keys': '$^'},
            'decoder': {'cnn_keys': '^image$', 'mlp_keys': '$^'},
        })
        overrides = dreamer.get('overrides', {})
        unknown = set(overrides) - set(raw)
        if unknown:
            raise ValueError(f'Unknown PyTorch Dreamer config fields: {sorted(unknown)}')
        native = merge(native, overrides)
        for section in ('encoder', 'decoder'):
            if native[section]['cnn_keys'] not in ('image', '^image$') or native[section]['mlp_keys'] != '$^':
                raise ValueError('EvidenceDreamer encoder/decoder use RGB only; teacher heatmap '
                                 'is a training target, never an observation input')
        if native['actor']['dist'] != 'onehot':
            raise ValueError('EvidenceDreamer requires the shared categorical onehot actor')
        if native['expl_behavior'] != 'greedy' or native.get('eval_state_mean', False):
            raise ValueError('EvidenceDreamer uses sampled posterior and task imagination; '
                             'separate exploration and eval_state_mean are unsupported')
        if int(native['imag_horizon']) < 3:
            raise ValueError('EvidenceDreamer imag_horizon must be at least 3')
        if config['device'] == 'cpu' and native['precision'] != 32:
            raise ValueError('CPU verification requires precision=32')
        native.update(
            device=config['device'], compile=False, video_pred_log=False,
            logdir=str(run_dir), num_actions=config['num_actions'],
            batch_size=config['batch_size'], batch_length=config['batch_length'],
            seed=config['seed'],
        )
        self.native = SimpleNamespace(**native)
        space = SimpleNamespace(spaces={
            'image': SimpleNamespace(shape=(64, 64, 3)),
            'is_first': SimpleNamespace(shape=()),
            'is_last': SimpleNamespace(shape=()),
            'is_terminal': SimpleNamespace(shape=()),
        })
        torch.manual_seed(config['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['seed'])
        self.wm = EvidenceWorldModel(
            space, SimpleNamespace(n=config['num_actions']), 0,
            self.native, self.options,
        ).to(config['device'])
        # Behavior uses the same native objectives and joint prior, broadcasting
        # fixed evidence snapshots instead of copying them for each time step.
        self.behavior = EvidenceImagBehavior(self.native, self.wm).to(config['device'])
        self.wm.requires_grad_(False)
        self.behavior.requires_grad_(False)
        self.replay = EvidenceReplay(config, self.options)
        self.updates = 0
        from ..diagnostics import FixedReplayProbes
        self.diagnostic_probes = FixedReplayProbes(config)

    def diagnostics(self, step):
        return self.diagnostic_probes.evaluate(self, step)

    def act(self, obs, carry, training=True, random_action=None):
        torch = self.torch
        # Teacher labels and rewards cannot enter policy state or the RGB writer.
        data = {key: np.asarray(obs[key])[None]
                for key in ('image', 'is_first', 'is_terminal')}
        with torch.no_grad():
            latent, previous = carry if carry is not None else (None, None)
            data = self.wm.preprocess(data)
            latent, _ = self.wm.observe_step(
                latent, previous, self.wm.encoder(data),
                data['is_first'], data['image'],
            )
            if random_action is not None:
                if not 0 <= int(random_action) < self.c['num_actions']:
                    raise ValueError('random_action outside discrete action space')
                action = torch.nn.functional.one_hot(
                    torch.tensor([int(random_action)], device=self.c['device']),
                    self.c['num_actions'],
                ).float()
            else:
                policy = self.behavior.actor(self.wm.dynamics.get_feat(latent))
                action = policy.sample() if training or self.c['protocol']['eval_action'] == 'sample' else policy.mode()
            return int(action.argmax(-1).item()), (
                {key: value.detach() for key, value in latent.items()}, action.detach(),
            ), {}

    def update(self):
        batch = self.replay.sample()
        post, context, metrics = self.wm._train(batch)
        # Native flatten would otherwise start imagination from burn-in, padding
        # and ended-episode rows, inflating actor budget and leaking invalid
        # starts. Clean success/time-limit rows may be last but nonterminal.
        valid = self.torch.as_tensor(batch['loss_mask'] & batch['sequence_valid']
                                    & ~batch['is_terminal'] & ~batch['is_last'],
                                    device=self.c['device'])
        if bool(valid.any()):
            starts = {key: value[valid][None].detach() for key, value in post.items()}
            reward = lambda feat, state, action: self.wm.policy_reward(
                self.wm.dynamics.get_feat(state))
            metrics.update(self.behavior._train(starts, reward)[-1])
            metrics['actor_valid_starts'] = int(valid.sum().item())
            metrics['actor_skipped'] = 0
        else:
            metrics['actor_valid_starts'] = 0
            metrics['actor_skipped'] = 1
        self.updates += 1
        return metrics

    @contextlib.contextmanager
    def evaluation(self, seed):
        # Restore modes as well as all global random streams. Memory is episode
        # carry only; this context never reads or writes the training replay.
        modules = list(self.behavior.modules())
        modes = [(module, module.training) for module in modules]
        with isolated_rng(seed):
            self.behavior.eval()
            try:
                yield
            finally:
                for module, training in modes:
                    module.training = training

    def parameter_report(self):
        unique = {id(parameter): parameter for module in (self.wm, self.behavior)
                  for parameter in module.parameters()}
        options = getattr(self.wm, 'options', self.options)
        if not isinstance(options, dict):
            options = self.options
        report = {
            'identity': 'Candidate read-only real task evidence + joint S/Z prior + FC/RC + native imagined actor-critic',
            'ablation': self.c.get('ablation', 'full'),
            'active_components': {'memory': bool(options['memory_enabled']),
                                  'memory_read': options['read_mode'],
                                  'semantic_teacher': self.needs_heatmap,
                                  'fc': options['fc_weight'] > 0,
                                  'rc': options['rc_weight'] > 0},
            'validation_status': 'implementation candidate; control benefit not established',
            'backend': 'pytorch',
            'dynamics': 'joint_action_conditioned_task_and_context',
            'observation': ('RGB only; frozen teacher heatmaps are training targets' if self.needs_heatmap
                            else 'RGB only; semantic teacher supervision disabled'),
            'learner_reward': 'task_reward + clip_scale * Clean MineCLIP record-high intrinsic',
            'clip_scale': options.get('clip_scale', 1.0),
            'policy_reward_route': 'same task reward head checked by RC + clip reward head',
            'rc_target': 'finite recorded task rewards; no critic bootstrap and no MineCLIP',
            'shared_initialization': False,
            'parameters_including_slow_value': sum(parameter.numel() for parameter in unique.values()),
            'world_model_parameters': sum(parameter.numel() for parameter in self.wm.parameters()),
            'dynamics_parameters': sum(parameter.numel() for parameter in self.wm.dynamics.parameters()),
            'latent_feature_size': self.native.dyn_deter + self.native.dyn_stoch * self.native.dyn_discrete,
            'memory_update': ('RGB observations only; fixed per imagination start' if options['memory_enabled']
                              else 'disabled; no memory writes or reads'),
            'burnin': self.replay.burnin,
            'learning_length': self.c['batch_length'],
            'sample_length': self.replay.burnin + self.c['batch_length'],
            'memory_observations': options.get('memory_observations', 16),
            'memory_tokens': 4 * options.get('memory_observations', 16),
            'memory_dim': options.get('memory_dim', 256),
            'attention_heads': options.get('attention_heads', 4),
            'reward_head_routes': {
                'task_reward': 'posterior real task labels; same head in RC and actor imagination',
                'clip_reward': 'posterior Clean MineCLIP record-high intrinsic labels; actor imagination',
            },
            'native_loss_scales': dict(self.wm._scales),
            'options': copy.deepcopy(options),
        }
        for name in ('s', 'z', 'writer'):
            module = getattr(self.wm.dynamics, name, None)
            if module is None:
                module = getattr(self.wm, name, None)
            if module is not None and hasattr(module, 'parameters'):
                report[f'{name}_parameters'] = sum(parameter.numel() for parameter in module.parameters())
        return report
