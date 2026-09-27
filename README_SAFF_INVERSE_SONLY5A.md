# ISO3-SAff-Inverse + S-only 5A

This experiment is a strict extension of the previously tested ISO3-SAff backbone.

## Kept unchanged
- Dual S/Z RSSM: S is action-conditioned; Z is action-free.
- RGB-only visual encoder.
- Global RGB + heatmap decoder.
- S-only affordance auxiliary supervision (`affordance_s_scale: 1.0`).
- Repaired S inverse dynamics (`inverse_loss_scale: 1.0`, target action is `data["action"][:,1:]`).
- MineCLIP/intrinsic reward path.
- No long branch.

## Disabled
- Z adversarial/GRL loss: `z_action_adv_scale: 0.0`.

## Added: S-only 5A
For horizons k in {1,2,4,8,15}, start from replay posterior S_t and roll the **existing S prior transition** forward with replay actions. Compare the predicted S prior to future posterior S using the same frozen random projection + cosine objective as Experiment 5A.

- horizons: `[1, 2, 4, 8, 15]`
- horizon weights: `[1.0, 1.0, 0.75, 0.5, 0.25]`
- projection dim: `512`
- starts per sequence: `4`
- loss scale: `0.05`

The auxiliary path calls `RSSM.img_step_s()`, which reuses `_img_in_s`, `_cell_s`, `_img_out_s`, and `_stat_s_img`. It does not evaluate Z, so the S-only 5A loss has no gradient path through Z.

## Important experimental-control choice
This **strict** package intentionally does **not** add the previously missing
`self._cell_s/z.apply(tools.weight_init)` lines. The old ISO3-SAff result was obtained without them. Keeping initialization unchanged isolates the effect of adding S-only 5A.

For the final paper-quality rerun, apply the separately provided InitFix patch to both the baseline and the new method and rerun from step 0.

## Expected metrics
- `loss_inverse`, `action_acc_s`
- `loss_affordance_s`
- `kl_s`, `kl_z`
- `s_multistep_loss`
- `s_multistep_loss_h1/h2/h4/h8/h15`
- `s_multistep_cosine_h1/h2/h4/h8/h15`
- `s_multistep_valid_h*`
- `s_multistep_target_raw_std_h*`

## Run
Use the same command as ISO3-SAff, e.g.

```bash
sh ./scripts/train.sh harvest_log_in_plains
```
