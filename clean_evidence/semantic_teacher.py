"""Current-view semantic targets without Clean LS reward or zoom generation."""
from __future__ import annotations

import numpy as np


def make_heatmap_only_wrapper(env, teacher, prompts):
    from gym import Wrapper

    if not prompts:
        raise ValueError('Evidence semantic supervision requires a nonempty task prompt')

    class HeatmapOnlyWrapper(Wrapper):
        def _label(self, observation):
            # The Clean U-Net preprocessing creates an unused random dummy
            # target. Do not perturb later environment randomness for it.
            random_state = np.random.get_state()
            try:
                masks = np.asarray(teacher._generate_mask(observation, list(prompts)))
            finally:
                np.random.set_state(random_state)
            if masks.ndim != 3 or not np.isfinite(masks).all():
                raise ValueError('Task U-Net must return finite prompt-by-height-by-width heatmaps')
            result = dict(observation)
            # The unchanged Clean LSImagineWrapper resizes and converts this
            # probability map to its ordinary 64 x 64 uint8 observation field.
            result['heatmap'] = np.clip(masks.max(axis=0), 0.0, 1.0).astype(np.float32)[..., None]
            return result

        def reset(self, **kwargs):
            return self._label(self.env.reset(**kwargs))

        def step(self, action):
            observation, reward, done, info = self.env.step(action)
            return self._label(observation), reward, done, info

    return HeatmapOnlyWrapper(env)
