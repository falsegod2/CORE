import json

import pytest

from summarize_paired_benchmark import evaluation_rows, summarize


def write_run(root, method, seed, rates, nested=True):
    path = root / method / 'formal/minedojo_harvest_log_in_plains' / f'seed_{seed}' / 'metrics.jsonl'
    path.parent.mkdir(parents=True)
    rows = []
    for step, rate in rates:
        rows.append(dict(step=step, eval_success=rate, eval_episodes=10))
        if nested:
            rows.append(dict(event='evaluation', step=step,
                             result=dict(eval_success=rate, eval_episodes=10)))
    path.write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    return path


def test_flat_and_nested_ls_evaluation_is_counted_once(tmp_path):
    path = write_run(tmp_path, 'ls_imagine', 0, [(2500, .2), (22500, .3)])
    rows, duplicates = evaluation_rows(path)
    assert len(rows) == 2 and duplicates == []
    assert rows[2500] == dict(successes=2, episodes=10, rate=.2)


def test_summary_uses_common_checkpoints_and_seed_paired_differences(tmp_path):
    write_run(tmp_path, 'ls_imagine', 0, [(2500, .2), (22500, .4), (42500, 1.)])
    write_run(tmp_path, 'no_memory', 0, [(2500, .4), (22500, .6)])
    write_run(tmp_path, 'ls_imagine', 1, [(2500, .6), (22500, .8)])
    write_run(tmp_path, 'no_memory', 1, [(2500, .4), (22500, .6), (42500, 0.)])
    write_run(tmp_path, 'ls_imagine', 2, [(2500, 1.), (22500, 1.)])
    out = summarize(tmp_path, ['ls_imagine', 'no_memory'], [0, 1, 2], last_evals=2)
    assert out['step'] == 22500 and out['evaluation_steps'] == [2500, 22500]
    assert out['paired_seeds'] == [0, 1] and out['excluded_seeds'] == [2]
    assert all(row['episodes'] == 20 for row in out['per_seed'])
    assert out['across_seed_success']['ls_imagine']['mean'] == pytest.approx(.5)
    assert out['across_seed_success']['no_memory']['mean'] == pytest.approx(.5)
    differences = out['paired_difference_percentage_points']['no_memory']
    assert differences['mean'] == pytest.approx(0.)
    assert [r['percentage_points'] for r in differences['per_seed']] == pytest.approx([20., -20.])


def test_explicit_unshared_checkpoint_and_short_windows_are_rejected(tmp_path):
    write_run(tmp_path, 'ls_imagine', 0, [(2500, .2), (22500, .3)])
    write_run(tmp_path, 'no_memory', 0, [(2500, .2)])
    with pytest.raises(ValueError, match='not a completed evaluation shared'):
        summarize(tmp_path, ['ls_imagine', 'no_memory'], [0], step=22500, last_evals=1)
    with pytest.raises(ValueError, match='only 1 available'):
        summarize(tmp_path, ['ls_imagine', 'no_memory'], [0], last_evals=2)
    out = summarize(tmp_path, ['ls_imagine', 'no_memory'], [0], last_evals=1)
    assert out['across_seed_success']['no_memory']['sample_std'] is None


def test_flat_duplicates_use_last_record_and_report_it(tmp_path):
    path = write_run(tmp_path, 'ls_imagine', 0, [(2500, .2), (2500, .4)])
    rows, duplicates = evaluation_rows(path)
    assert duplicates == [2500] and rows[2500]['successes'] == 4


def test_invalid_log_tail_and_missing_episode_count_are_not_silently_ignored(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    path.write_text('{"step":2500,')
    with pytest.raises(ValueError, match='invalid/incomplete JSON'):
        evaluation_rows(path)
    path.write_text('{"step":2500,"eval_success":0.2}\n')
    with pytest.raises(ValueError, match='eval_episodes is required'):
        evaluation_rows(path)


@pytest.mark.parametrize('event', ['resume', 'resume_boundary'])
def test_resume_discards_stale_evaluations_above_restored_checkpoint(tmp_path, event):
    path = write_run(tmp_path, 'ls_imagine', 0, [(2500, .2), (22500, 1.)])
    with path.open('a') as stream:
        stream.write(json.dumps(dict(event=event, step=20000)) + '\n')
        stream.write(json.dumps(dict(event='train', step=22000)) + '\n')
    write_run(tmp_path, 'no_memory', 0, [(2500, .3), (22500, .4)])
    out = summarize(tmp_path, ['ls_imagine', 'no_memory'], [0], last_evals=1)
    assert out['step'] == 2500
    assert out['evaluations_invalidated_by_resume']['ls_imagine/seed_0'] == [
        dict(resumed_at_step=20000, discarded_steps=[22500])]
    # A fresh evaluation at the same step becomes eligible again.
    with path.open('a') as stream:
        stream.write(json.dumps(dict(step=22500, eval_success=.5, eval_episodes=10)) + '\n')
    out = summarize(tmp_path, ['ls_imagine', 'no_memory'], [0], last_evals=1)
    assert out['step'] == 22500
    ls = next(row for row in out['per_seed'] if row['method'] == 'ls_imagine')
    assert ls['successes'] == 5
