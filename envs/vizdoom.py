"""ViZDoom -> DreamerV3 pixel-only environment adapter.

The rest of this repository uses the legacy Gym 4-tuple API and legacy
``gym.spaces`` objects. Current ViZDoom releases expose Gymnasium. This module
keeps that incompatibility local: it owns the Gymnasium environment and
exports the same old-style interface as the existing MineDojo wrapper.

Only RGB pixels are consumed by Dreamer. Diagnostic-only ``heatmap`` and
``task_score`` fields may be stored in replay, but encoder/decoder regexes remain
``^image$`` and no diagnostic enters a training loss. ViZDoom game variables,
labels, depth, and automap buffers are intentionally excluded.
"""

import math

import cv2
import gym
import numpy as np


# Current ViZDoom uses -v1 IDs. -v0 is kept as a compatibility fallback for
# older 1.2/early-1.3 installations.
_SCENARIOS = {
    "basic": {
        "env_ids": ("VizdoomBasic-v1", "VizdoomBasic-v0"),
        "timeout_tics": 300,
        "success_threshold": 50.0,
        "teacher": "none",
    },
    "my_way_home": {
        "env_ids": ("VizdoomMyWayHome-v1", "VizdoomMyWayHome-v0"),
        "timeout_tics": 2100,
        "success_threshold": 0.5,
        "teacher": "green_goal",
    },
    "defend_center": {
        "env_ids": ("VizdoomDefendCenter-v1", "VizdoomDefendCenter-v0"),
        "timeout_tics": 2100,
        "success_threshold": 0.5,
        "teacher": "none",
    },
    "deadly_corridor": {
        "env_ids": ("VizdoomDeadlyCorridor-v1", "VizdoomCorridor-v0"),
        "timeout_tics": 2100,
        "success_threshold": 500.0,
        "teacher": "green_goal",
    },
    "predict_position": {
        "env_ids": ("VizdoomPredictPosition-v1", "VizdoomPredictPosition-v0"),
        "timeout_tics": 300,
        "success_threshold": 0.5,
        "teacher": "moving_target",
    },
}

_ALIASES = {
    "mywayhome": "my_way_home",
    "my-way-home": "my_way_home",
    "defendcenter": "defend_center",
    "defend-center": "defend_center",
    "deadlycorridor": "deadly_corridor",
    "deadly-corridor": "deadly_corridor",
    "corridor": "deadly_corridor",
    "predictposition": "predict_position",
    "predict-position": "predict_position",
}


def _normalize_task(task):
    task = str(task).strip().lower()
    task = _ALIASES.get(task, task)
    if task not in _SCENARIOS:
        options = ", ".join(sorted(_SCENARIOS))
        raise KeyError(f"Unknown ViZDoom task '{task}'. Available: {options}")
    return task


def available_tasks():
    return tuple(sorted(_SCENARIOS))


def episode_max_steps(task, frame_skip=1):
    """Return the scenario timeout in *agent decisions*, not Doom tics."""
    task = _normalize_task(task)
    frame_skip = max(1, int(frame_skip))
    return int(math.ceil(_SCENARIOS[task]["timeout_tics"] / frame_skip))


def default_teacher_mode(task):
    task = _normalize_task(task)
    return _SCENARIOS[task]["teacher"]


class VizDoomPixelEnv:
    """Pixel-only ViZDoom environment using the repository's legacy API.

    Parameters
    ----------
    task:
        One of :func:`available_tasks`.
    seed:
        Seed used on the first Gymnasium reset.
    image_size:
        Square RGB size given to Dreamer (default 64).
    frame_skip:
        Number of Doom tics advanced per agent decision.
    reward_mode:
        ``native`` keeps the scenario reward. ``sparse_success`` emits +1 only
        on the first success event in an episode and 0 otherwise. The latter is
        useful for explicitly sparse-reward ablations.
    success_threshold:
        Optional override for the task-specific single-step reward threshold
        used only for logging ``success`` / ``first_success_step``.
    """

    def __init__(
        self,
        task,
        seed=0,
        image_size=64,
        frame_skip=4,
        reward_mode="native",
        success_threshold=None,
        teacher_mode="auto",
        render_mode=None,
    ):
        task = _normalize_task(task)
        self.task = task
        self._seed = int(seed)
        self._seed_pending = True
        self._image_size = int(image_size)
        self._frame_skip = max(1, int(frame_skip))
        self._reward_mode = str(reward_mode).lower()
        if self._reward_mode not in ("native", "sparse_success"):
            raise ValueError(
                "vizdoom_reward_mode must be 'native' or 'sparse_success', "
                f"got {reward_mode!r}"
            )
        if teacher_mode is None or str(teacher_mode).lower() == "auto":
            teacher_mode = default_teacher_mode(task)
        self._teacher_mode = str(teacher_mode).strip().lower()
        if self._teacher_mode not in ("green_goal", "moving_target", "none"):
            raise ValueError(
                "vizdoom_diagnostic_teacher_mode must be 'auto', 'green_goal', "
                "'moving_target', or 'none'"
            )
        default_threshold = _SCENARIOS[task]["success_threshold"]
        if success_threshold is None or float(success_threshold) < 0:
            success_threshold = default_threshold
        self._success_threshold = float(success_threshold)
        self._max_steps = episode_max_steps(task, self._frame_skip)
        self._episode_step = 0
        self._success = False
        self._first_success_step = self._max_steps
        self._teacher_prev_image = None

        try:
            import gymnasium as gymnasium
            # Import is intentionally retained even if unused: ViZDoom performs
            # Gymnasium environment registration as an import side effect.
            from vizdoom import gymnasium_wrapper  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "ViZDoom support requires `pip install vizdoom` and Gymnasium. "
                "See docs/vizdoom.md."
            ) from exc

        errors = []
        self._env = None
        self._env_id = None
        for env_id in _SCENARIOS[task]["env_ids"]:
            try:
                self._env = gymnasium.make(
                    env_id,
                    frame_skip=self._frame_skip,
                    max_buttons_pressed=1,
                    render_mode=render_mode,
                )
                self._env_id = env_id
                break
            except Exception as exc:  # version fallback; report all failures
                errors.append(f"{env_id}: {type(exc).__name__}: {exc}")
        if self._env is None:
            details = "\n".join(errors)
            raise RuntimeError(
                f"Could not create ViZDoom scenario '{task}'. Tried:\n{details}"
            )

        # Dreamer expects legacy gym.spaces, so do not expose Gymnasium spaces.
        n = getattr(self._env.action_space, "n", None)
        if n is None:
            raise TypeError(
                f"{self._env_id} did not expose a Discrete action space. "
                "This adapter intentionally uses max_buttons_pressed=1 so all "
                "algorithms share a compact categorical action space."
            )
        self.action_space = gym.spaces.Discrete(int(n))
        self.action_space.discrete = True
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    0, 255,
                    shape=(self._image_size, self._image_size, 3),
                    dtype=np.uint8,
                ),
                # Diagnostic-only RGB teacher outputs. Encoder/decoder regexes
                # remain '^image$', so Dreamer never consumes these as inputs.
                "heatmap": gym.spaces.Box(
                    0, 255,
                    shape=(self._image_size, self._image_size, 1),
                    dtype=np.uint8,
                ),
                "task_score": gym.spaces.Box(
                    0.0, 1.0, shape=(1,), dtype=np.float32
                ),
                "is_first": gym.spaces.Box(0, 1, shape=(), dtype=np.uint8),
                "is_last": gym.spaces.Box(0, 1, shape=(), dtype=np.uint8),
                "is_terminal": gym.spaces.Box(0, 1, shape=(), dtype=np.uint8),
                # The current clean Dreamer code still defines this head. For
                # ViZDoom it is a zero target and its loss/reward scales are 0.
                "mineclip_reward": gym.spaces.Box(
                    -np.inf, np.inf, shape=(), dtype=np.float32
                ),
            }
        )

    @property
    def env_id(self):
        return self._env_id

    @property
    def max_episode_steps(self):
        return self._max_steps

    def reset(self):
        if self._seed_pending:
            raw_obs, _info = self._env.reset(seed=self._seed)
            self._seed_pending = False
        else:
            raw_obs, _info = self._env.reset()
        self._episode_step = 0
        self._success = False
        self._first_success_step = self._max_steps
        self._teacher_prev_image = None
        return self._convert_obs(raw_obs, is_first=True, is_last=False, is_terminal=False)

    def step(self, action):
        raw_obs, native_reward, terminated, truncated, info = self._env.step(int(action))
        self._episode_step += 1
        native_reward = float(native_reward)

        success_now = native_reward >= self._success_threshold
        if success_now and not self._success:
            self._success = True
            self._first_success_step = self._episode_step

        if self._reward_mode == "native":
            reward = native_reward
        else:  # sparse_success
            reward = 1.0 if success_now and self._first_success_step == self._episode_step else 0.0

        done = bool(terminated or truncated)
        # A time-limit truncation is not an environment terminal for Dreamer.
        is_terminal = bool(terminated)
        obs = self._convert_obs(
            raw_obs,
            is_first=False,
            is_last=done,
            is_terminal=is_terminal,
        )

        info = dict(info)
        info["real_done"] = done
        info["is_terminal"] = is_terminal
        info["success"] = bool(success_now)
        info["first_success_step"] = int(self._first_success_step)
        info["native_reward"] = native_reward
        info["vizdoom_diagnostic_teacher_mode"] = self._teacher_mode
        return obs, np.float32(reward), done, info

    def _convert_obs(self, raw_obs, is_first, is_last, is_terminal):
        if not isinstance(raw_obs, dict) or "screen" not in raw_obs:
            raise KeyError(
                "ViZDoom Gymnasium observation must contain the 'screen' RGB buffer"
            )
        image = np.asarray(raw_obs["screen"])
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        if image.shape[-1] > 3:
            image = image[..., :3]
        image = cv2.resize(
            image.astype(np.uint8),
            (self._image_size, self._image_size),
            interpolation=cv2.INTER_AREA,
        )
        heatmap, task_score = self._teacher(image)
        return {
            "image": image,
            "heatmap": heatmap[..., None],
            "task_score": np.asarray([task_score], dtype=np.float32),
            "is_first": np.asarray(is_first, dtype=np.uint8),
            "is_last": np.asarray(is_last, dtype=np.uint8),
            "is_terminal": np.asarray(is_terminal, dtype=np.uint8),
            "mineclip_reward": np.asarray(0.0, dtype=np.float32),
        }

    @staticmethod
    def _top_score(response, fraction=0.02):
        flat = np.asarray(response, dtype=np.float32).reshape(-1)
        k = max(1, int(math.ceil(float(fraction) * flat.size)))
        top = np.partition(flat, flat.size - k)[-k:]
        return float(np.clip(np.mean(top), 0.0, 1.0))

    def _green_goal_teacher(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hue = hsv[..., 0]
        sat = hsv[..., 1] / 255.0
        val = hsv[..., 2] / 255.0
        hue_dist = np.abs(hue - 60.0)
        hue_dist = np.minimum(hue_dist, 180.0 - hue_dist)
        hue_score = np.exp(-0.5 * (hue_dist / 16.0) ** 2)
        response = hue_score * np.power(sat, 1.5) * np.power(val, 0.75)
        response = np.clip(response, 0.0, 1.0)
        response = cv2.GaussianBlur(response, (0, 0), sigmaX=1.0)
        return np.clip(response, 0.0, 1.0)

    def _moving_target_teacher(self, image):
        current = image.astype(np.float32) / 255.0
        previous_u8 = self._teacher_prev_image
        if previous_u8 is None:
            return np.zeros((self._image_size, self._image_size), np.float32)
        previous = previous_u8.astype(np.float32) / 255.0
        cur_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY).astype(np.float32)
        prev_gray = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY).astype(np.float32)
        try:
            (dx, dy), _ = cv2.phaseCorrelate(prev_gray, cur_gray)
            dx = float(np.clip(dx, -self._image_size / 3, self._image_size / 3))
            dy = float(np.clip(dy, -self._image_size / 3, self._image_size / 3))
        except Exception:
            dx = dy = 0.0
        candidates = []
        for sign in (1.0, -1.0):
            matrix = np.asarray(
                [[1.0, 0.0, sign * dx], [0.0, 1.0, sign * dy]],
                dtype=np.float32,
            )
            aligned = cv2.warpAffine(
                previous, matrix, (self._image_size, self._image_size),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101,
            )
            residual = np.mean(np.abs(current - aligned), axis=-1)
            candidates.append((float(np.median(residual)), residual))
        residual = min(candidates, key=lambda x: x[0])[1]
        lo = float(np.quantile(residual, 0.70))
        hi = float(np.quantile(residual, 0.995))
        denom = max(hi - lo, 1.0e-4)
        response = np.clip((residual - lo) / denom, 0.0, 1.0)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        sat = hsv[..., 1] / 255.0
        response *= (0.35 + 0.65 * np.sqrt(np.clip(sat, 0.0, 1.0)))
        response = cv2.GaussianBlur(response, (0, 0), sigmaX=1.0)
        response[response < 0.08] = 0.0
        return np.clip(response, 0.0, 1.0)

    def _teacher(self, image):
        if self._teacher_mode == "none":
            response = np.zeros((self._image_size, self._image_size), np.float32)
        elif self._teacher_mode == "green_goal":
            response = self._green_goal_teacher(image)
        elif self._teacher_mode == "moving_target":
            response = self._moving_target_teacher(image)
        else:
            raise RuntimeError(self._teacher_mode)
        task_score = self._top_score(response, fraction=0.02)
        heatmap = np.round(np.clip(response, 0.0, 1.0) * 255.0).astype(np.uint8)
        self._teacher_prev_image = image.copy()
        return heatmap, np.float32(task_score)

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()


def make_env(task, **kwargs):
    return VizDoomPixelEnv(task, **kwargs)
