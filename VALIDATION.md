# Validation

Completed in the build environment:

- Full Python syntax compilation.
- YAML parsing and required-key checks.
- Static audit for removed S/Z, inverse-dynamics, heatmap, and long-term paths.
- Synthetic single-RSSM sequence test covering posterior, prior, feature shape, KL loss, and backward pass.
- Synthetic world-model optimization step covering RGB reconstruction, task reward, MineCLIP reward, termination prediction, and KL losses.
- Synthetic imagined actor-critic update covering categorical actions, lambda returns, return normalization, actor loss, and critic loss.

Not completed in the build environment:

- MineDojo end-to-end environment startup, because MineDojo/MineCLIP runtime dependencies are not available here.
