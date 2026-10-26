"""Named component and combination controls for the Clean Evidence experiment.

The environment, MineCLIP reward, model size recipe and interaction budget are
shared. ``no_semantic`` deliberately removes all teacher supervision together;
it is not a replacement reward or a different observation protocol.
"""
from __future__ import annotations

import copy


ABLATIONS = {
    'full': {
        'description': 'Full Evidence: RGB memory, per-step reads, semantic targets, FC and RC.',
        'evidence': {},
    },
    'no_memory': {
        'description': 'Disable RGB memory writes and reads; retain semantic targets, FC and RC.',
        'evidence': {'memory_enabled': False, 'read_mode': 'none'},
    },
    'no_memory_no_rc': {
        'description': 'Disable RGB memory and RC; retain FC, semantic targets and both supervised reward heads.',
        'evidence': {'memory_enabled': False, 'read_mode': 'none', 'rc_weight': 0.0},
    },
    'no_memory_no_fc_rc': {
        'description': 'Disable RGB memory, FC and RC; retain local/posterior semantic targets and both supervised reward heads.',
        'evidence': {'memory_enabled': False, 'read_mode': 'none',
                     'fc_weight': 0.0, 'rc_weight': 0.0},
    },
    'memory_start_only': {
        'description': 'Read the fixed RGB memory only at the first imagined transition; retain real-observation updates.',
        'evidence': {'read_mode': 'start_only'},
    },
    'no_fc': {
        'description': 'Disable the complete future-consistency objective (categorical, hidden and heatmap terms).',
        'evidence': {'fc_weight': 0.0},
    },
    'no_rc': {
        'description': 'Disable recorded task-reward consistency; retain the supervised task-reward head.',
        'evidence': {'rc_weight': 0.0},
    },
    'no_fc_rc': {
        'description': 'Disable both recorded-prefix FC and RC objectives; retain RGB memory and semantic targets.',
        'evidence': {'fc_weight': 0.0, 'rc_weight': 0.0},
    },
    'no_semantic': {
        'description': 'Remove local, posterior and future heatmap supervision and the U-Net teacher; retain RGB memory, categorical/hidden FC and RC.',
        'evidence': {'local_heat_weight': 0.0, 'posterior_heat_weight': 0.0, 'fc_heat_weight': 0.0},
    },
}


def requires_teacher(options):
    return bool(options['local_heat_weight'] > 0 or
                options['posterior_heat_weight'] > 0 or
                (options['fc_weight'] > 0 and options['fc_heat_weight'] > 0))


def apply_ablation(config, name):
    if name not in ABLATIONS:
        raise ValueError(f'Unknown Evidence ablation: {name}; choose from {", ".join(ABLATIONS)}')
    config['ablation'] = name
    config['evidence'].update(copy.deepcopy(ABLATIONS[name]['evidence']))
    config['needs_heatmap'] = requires_teacher(config['evidence'])


def validate_ablation(config):
    name = config.get('ablation', 'full')
    if name not in ABLATIONS:
        raise ValueError(f'Unknown Evidence ablation: {name}')
    for key, value in ABLATIONS[name]['evidence'].items():
        if config['evidence'][key] != value:
            raise ValueError(f'Ablation {name} requires evidence.{key}={value!r}; do not override its defining switch')
    if config['needs_heatmap'] != requires_teacher(config['evidence']):
        raise ValueError('needs_heatmap must match the enabled semantic losses')


def descriptions():
    return [dict(name=name, **copy.deepcopy(recipe)) for name, recipe in ABLATIONS.items()]
