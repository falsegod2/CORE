# ISO3-SAff + S-only 5A (No Inverse)

Purpose: strict ablation against `ISO3-SAff + Inverse + S-only 5A`.

Training changes relative to the previous strict version:

- `inverse_loss_scale: 1.0 -> 0.0`
- Keep `affordance_s_scale: 1.0`
- Keep `z_action_adv_scale: 0.0`
- Keep S-only 5A unchanged: horizons `[1,2,4,8,15]`, weights `[1,1,0.75,0.5,0.25]`, scale `0.05`
- Keep the same dual S/Z RSSM and the same global RGB+heatmap decoder supervision.
- Do **not** apply the separate GRU InitFix in this ablation; that would introduce a second experimental variable.

## Why the inverse head is still instantiated

The inverse head remains in `networks.py`, but because `inverse_loss_scale=0.0`, it contributes zero gradient to the model objective. Keeping the head instantiated preserves the parameter-construction/RNG path of the previous strict baseline as closely as possible. Its `action_acc_s` log is therefore only a random/frozen-head diagnostic in this experiment and must not be interpreted as learned inverse accuracy.

## Core objective

`L = L_Dreamer + L_global_aff + 1.0 L_S-aff + 0.05 L_S-multistep`

with no inverse supervision and no Z adversarial loss.

## Run

```bash
python -m py_compile networks.py models.py s_multistep_consistency.py expr.py
python audit_saff_sonly5a_noinverse.py
bash ./scripts/train.sh harvest_log_in_plains
```

Diagnostics variant: see README_S5A_DIAGNOSTICS.md
