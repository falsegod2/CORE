CORE2-WMP-MineCLIPActor

Goal:
- Keep CORE2-WMP and long-term imagination.
- Disable the original LS-Imagine intrinsic path.
- Add a clean MineCLIP reward path to the actor reward.

Main actor reward:
  actor_reward = env_reward_prediction
               + mineclip_reward_scale * mineclip_reward_prediction
               + WMP_reward

Removed/disabled:
- obs['intrinsic'] from ClipWrapper is no longer used for MineCLIP progress.
- affordance/Gaussian intrinsic from ConcentrationWrapper is not written into obs['intrinsic'].
- LS_ImagineWrapper writes intrinsic and intrinsic_on_zoomed as 0.0.
- intrinsic_head remains for compatibility but loss_scale is 0 and use_original_intrinsic is False.

Added clean fields:
- obs['mineclip_score']
- obs['mineclip_reward']
- obs['mineclip_score_on_zoomed']
- obs['mineclip_reward_on_zoomed']

Added model head:
- heads['mineclip_reward'] trained from data['mineclip_reward'].

Long-term accumulated_reward:
- Original intrinsic is excluded.
- Clean mineclip_reward is included in interval targets.

Expected logs:
- mineclip_reward_mean / std / min / max
- mineclip_reward_loss
- mineclip_reward_scale
- wm_progress_reward_* and wmp_* still appear.
