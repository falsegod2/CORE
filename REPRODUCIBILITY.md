# LS-Imagine statistical reproducibility profile

This code keeps the original LS-Imagine components enabled: RGB + affordance
encoder input, zoomed observations, jump transitions, interval/accumulated
reward heads, and mixed long-short imagination.

## Run

```bash
bash ./scripts/train_repro.sh harvest_log_in_plains 0 0
```

The launcher creates a fresh timestamped run. To resume, call `expr.py` with the
exact existing timestamp directory as `--logdir`.

## Changes

- fixed MineDojo simulator `seed` and `world_seed` for train and evaluation;
- hard reset replaces persistent semi-fast-reset worlds;
- deterministic PyTorch/cuDNN/cuBLAS path; TF32 and `torch.compile` disabled;
- evaluation uses posterior mode, fixed evaluation seeds/fresh affordance thresholds, and restores Python/NumPy/Torch RNG afterward;
- deterministic prefill and separately seeded replay samplers;
- separate train/evaluation `ScoreStorage` instances;
- MineCLIP and affordance branch-driving values quantized before comparisons;
- affordance maps lightly quantized before zoom selection and encoder use;
- U-Net dummy labels no longer consume NumPy RNG;
- long/short jump Bernoulli choices use a dedicated checkpointed generator;
- checkpoint saves Python, NumPy, Torch CPU/CUDA and jump-generator RNG states.

## Limits

The target is statistically closer training curves, not bitwise identity across
different GPU architectures or software stacks. Use the same Python, PyTorch,
CUDA/cuDNN, MineDojo, Java, MineCLIP/U-Net checkpoints and assets on both
machines. The reproducible profile changes reset behavior and lightly quantizes
semantic values, so all formal baselines should be rerun under the same profile.
