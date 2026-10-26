from evidence_tests.test_environment import clean_runtime, chain
from clean_ls.configuration import build_config
from clean_ls.environment import make_environment


def test_managed_ls_train_and_eval_keep_shaping_and_absolute_teacher_paths(clean_runtime):
    state = clean_runtime
    config = build_config(assets_root=state.assets, logdir=state.run_dir,
                          overrides=['device=cpu', 'env_device=cpu'])
    environments = []
    try:
        for mode in ('train', 'eval'):
            env = make_environment(config, mode, state.run_dir)
            environments.append(env)
            assert 'ConcentrationWrapper' in chain(env)
            assert 'HeatmapOnlyWrapper' not in chain(env)
            for _ in range(6):
                obs = env.reset()
                assert obs['intrinsic'] > 0
            assert env.action_space.shape == (12,)
        assert len(state.constructs) == 2
        assert all(sim.fast_resets == 5 for sim in state.sims)
        assert all(c['device'] == 'cpu' for c in state.clips)
        assert all(c['ckpt'] == str(state.assets / 'weights/mineclip_attn.pth') for c in state.clips)
        assert all(c['device'] == 'cpu' for c in state.teachers)
        assert all(c['unet_checkpoint_dir'].startswith(str(state.assets)) for c in state.teachers)
        assert state.tasks.CUTOM_TASK_SPECS == state.original_specs
    finally:
        for env in environments:
            env.close()
