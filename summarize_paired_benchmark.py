#!/usr/bin/env python3
"""Read matched-seed metrics and print a common-checkpoint comparison as JSON.

Only flat eval_success rows count as completed evaluations. LS runtime events
repeat a summary under result; they are deliberately ignored. Multiple flat
evaluations at the same step use the last record and report their duplication.
No files are written and no training or simulator code is imported.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics

from run_paired_benchmark import ROOT, METHODS


def evaluation_rows(path, invalidated=None):
    rows = {}
    duplicates = []
    with Path(path).open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'{path}:{line_number}: invalid/incomplete JSON; copy the log after a completed line') from error
            if row.get('event') in ('resume', 'resume_boundary'):
                restored_step = row.get('step')
                if (isinstance(restored_step, bool) or not isinstance(restored_step, (int, float)) or
                        not math.isfinite(restored_step) or restored_step < 0 or int(restored_step) != restored_step):
                    raise ValueError(f'{path}:{line_number}: invalid resume step')
                stale = sorted(step for step in rows if step > restored_step)
                for step in stale:
                    del rows[step]
                if stale and invalidated is not None:
                    invalidated.append(dict(resumed_at_step=int(restored_step), discarded_steps=stale))
            if 'eval_success' not in row:
                continue
            step, rate, episodes = row.get('step'), row['eval_success'], row.get('eval_episodes')
            if (isinstance(step, bool) or not isinstance(step, (float, int)) or
                    not math.isfinite(step) or step < 0 or int(step) != step):
                raise ValueError(f'{path}:{line_number}: invalid evaluation step')
            if (isinstance(rate, bool) or not isinstance(rate, (float, int)) or
                    not math.isfinite(rate) or not 0 <= rate <= 1):
                raise ValueError(f'{path}:{line_number}: invalid evaluation success')
            if (isinstance(episodes, bool) or not isinstance(episodes, (float, int)) or
                    not math.isfinite(episodes) or episodes < 1 or int(episodes) != episodes):
                raise ValueError(f'{path}:{line_number}: complete eval_episodes is required')
            step, episodes = int(step), int(episodes)
            successes = round(rate * episodes)
            if not math.isclose(rate * episodes, successes, abs_tol=1e-5):
                raise ValueError(f'{path}:{line_number}: success is inconsistent with episode count')
            if step in rows:
                duplicates.append(step)
            rows[step] = dict(successes=successes, episodes=episodes, rate=float(rate))
    return rows, duplicates


def summarize(root, methods, seeds, stage='formal', task='minedojo_harvest_log_in_plains',
              step=None, last_evals=10, reference='ls_imagine'):
    root = Path(root).expanduser().resolve()
    loaded, missing, duplicate_records, invalidated_records = {}, [], {}, {}
    for seed in seeds:
        for method in methods:
            path = root / method / stage / task / f'seed_{seed}' / 'metrics.jsonl'
            if not path.is_file():
                missing.append(dict(method=method, seed=seed, reason='metrics.jsonl missing'))
                continue
            invalidated = []
            records, duplicates = evaluation_rows(path, invalidated=invalidated)
            if invalidated:
                invalidated_records[f'{method}/seed_{seed}'] = invalidated
            if not records:
                missing.append(dict(method=method, seed=seed, reason='no completed flat evaluation'))
                continue
            loaded[seed, method] = records
            if duplicates:
                duplicate_records[f'{method}/seed_{seed}'] = duplicates
    paired_seeds = [seed for seed in seeds if all((seed, method) in loaded for method in methods)]
    if not paired_seeds:
        raise ValueError('No seed has completed evaluations for every selected method')
    common = set.intersection(*(set(loaded[seed, method]) for seed in paired_seeds for method in methods))
    if not common:
        raise ValueError('Selected runs have no common completed evaluation checkpoint')
    chosen_step = max(common) if step is None else step
    if chosen_step not in common:
        raise ValueError(f'--step {chosen_step} is not a completed evaluation shared by all paired runs; '
                         f'latest common checkpoint is {max(common)}')
    window = sorted(s for s in common if s <= chosen_step)[-last_evals:]
    if len(window) != last_evals:
        raise ValueError(f'Need {last_evals} common evaluations by {chosen_step}; only {len(window)} available')
    per_seed = []
    for seed in paired_seeds:
        for method in methods:
            selected = [loaded[seed, method][s] for s in window]
            episodes = sum(row['episodes'] for row in selected)
            successes = sum(row['successes'] for row in selected)
            per_seed.append(dict(method=method, seed=seed, successes=successes, episodes=episodes,
                                 success_rate=successes / episodes,
                                 endpoint_success_rate=loaded[seed, method][chosen_step]['rate']))
    def stats(values):
        return dict(n_seeds=len(values), mean=statistics.mean(values),
                    sample_std=statistics.stdev(values) if len(values) > 1 else None)
    aggregates = {method:stats([r['success_rate'] for r in per_seed if r['method'] == method])
                  for method in methods}
    differences = {}
    for method in methods:
        if method == reference:
            continue
        paired = []
        for seed in paired_seeds:
            by_method = {r['method']:r['success_rate'] for r in per_seed if r['seed'] == seed}
            paired.append(dict(seed=seed, percentage_points=100 * (by_method[method] - by_method[reference])))
        differences[method] = dict(reference=reference, per_seed=paired,
                                  **stats([r['percentage_points'] for r in paired]))
    return dict(experiment_root=str(root), paired_seeds=paired_seeds,
                excluded_seeds=[seed for seed in seeds if seed not in paired_seeds],
                missing_runs=missing, step=chosen_step, evaluation_steps=window,
                per_seed=per_seed, across_seed_success=aggregates,
                paired_difference_percentage_points=differences,
                duplicate_flat_eval_steps=duplicate_records,
                evaluations_invalidated_by_resume=invalidated_records,
                notes=['Window aggregates different checkpoints, not repeated tests of one frozen policy.',
                       'Across-seed statistics weight each seed equally; sample_std is null for one seed.',
                       'Matching experiment seeds does not guarantee identical Minecraft world sequences.',
                       'Nested LS evaluation summaries are ignored; duplicate flat steps use the last record.',
                       'On resume, evaluations after the restored checkpoint are discarded until rerun.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--experiment-root', type=Path, default=ROOT / 'logdir' / 'paired_20261025')
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=['ls_imagine', 'no_memory'])
    parser.add_argument('--reference', choices=METHODS, default='ls_imagine')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--stage', choices=('formal', 'smoke'), default='formal')
    parser.add_argument('--task', default='minedojo_harvest_log_in_plains')
    parser.add_argument('--step', type=int, help='Exact common evaluation checkpoint; default: latest common')
    parser.add_argument('--last-evals', type=int, default=10)
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.methods)) != len(args.methods):
        parser.error('Seeds and methods must be unique')
    if args.reference not in args.methods:
        parser.error('--reference must be included in --methods')
    if args.last_evals < 1 or any(s < 0 for s in args.seeds):
        parser.error('--last-evals must be positive and seeds must be nonnegative')
    result = summarize(args.experiment_root, args.methods, args.seeds, args.stage, args.task,
                       args.step, args.last_evals, args.reference)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
