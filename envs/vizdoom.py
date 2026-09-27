"""ViZDoom adapter for LS-Imagine.

This module bridges the current Gymnasium ViZDoom API to the legacy Gym-style
interface used by this repository.  It preserves LS-Imagine's environment-
dependent inputs instead of zeroing them out:

* RGB observation (``image``)
* task-affordance heatmap (``heatmap``)
* a mild RGB-only zoomed pseudo-observation for the long-term branch
* ``jump`` / ``is_zoomed`` labels
* score pairs used by ``ScoreStorage`` to recover jumping steps and accumulated
  reward targets

For MyWayHome and DeadlyCorridor the task-specific teacher is an RGB-only
"green goal" detector.  It never reads depth, labels, automap, game variables,
or agent coordinates.  Other scenarios are supported for interface testing but
use teacher_mode='none' by default; they should not be presented as a full
LS-Imagine evaluation until a task-equivalent semantic teacher is supplied.
"""

import math

import cv2
import gym
import numpy as np


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
        "teacher": "none",
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
        raise KeyError(f"Unknown ViZDoom task {task!r}. Available: {options}")
    return task


def available_tasks():
    return tuple(sorted(_SCENARIOS))


def episode_max_steps(task, frame_skip=1):
    task = _normalize_task(task)
    frame_skip = max(1, int(frame_skip))
    return int(math.ceil(_SCENARIOS[task]["timeout_tics"] / frame_skip))


def default_teacher_mode(task):
    return _SCENARIOS[_normalize_task(task)]["teacher"]


class VizDoomLSImagineEnv:
    """Legacy-Gym interface with LS-Imagine auxiliary observations.

    Parameters
    ----------
    intrinsic_mode:
        ``score_improvement`` reproduces the *role* of LS-Imagine's semantic
        intrinsic signal using the environment-compatible RGB teacher.
        ``none`` emits zero intrinsic reward while leaving affordance/jump/zoom
        machinery active.  The latter is used by the no-intrinsic ablation.
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
        intrinsic_mode="score_improvement",
        intrinsic_scale=0.01,
        zoom_factor=1.35,
        jump_min_gain=0.015,
        render_mode=None,
    ):
        task = _normalize_task(task)
        self.task = task
        self._seed = int(seed)
        self._seed_pending = True
        self._image_size = int(image_size)
        self._frame_skip = max(1, int(frame_skip))
        self._max_steps = episode_max_steps(task, self._frame_skip)

        self._reward_mode = str(reward_mode).strip().lower()
        if self._reward_mode not in ("native", "sparse_success"):
            raise ValueError(
                "vizdoom_reward_mode must be 'native' or 'sparse_success', "
                f"got {reward_mode!r}"
            )

        if teacher_mode is None or str(teacher_mode).strip().lower() == "auto":
            teacher_mode = default_teacher_mode(task)
        self._teacher_mode = str(teacher_mode).strip().lower()
        if self._teacher_mode not in ("green_goal", "none"):
            raise ValueError(
                "vizdoom_teacher_mode must be auto|green_goal|none, got "
                f"{teacher_mode!r}"
            )

        self._intrinsic_mode = str(intrinsic_mode).strip().lower()
        if self._intrinsic_mode not in ("score_improvement", "none"):
            raise ValueError(
                "vizdoom_intrinsic_mode must be score_improvement|none, got "
                f"{intrinsic_mode!r}"
            )
        self._intrinsic_scale = float(intrinsic_scale)
        self._zoom_factor = max(1.0, float(zoom_factor))
        self._jump_min_gain = max(0.0, float(jump_min_gain))

        default_threshold = _SCENARIOS[task]["success_threshold"]
        if success_threshold is None or float(success_threshold) < 0:
            success_threshold = default_threshold
        self._success_threshold = float(success_threshold)

        self._episode_step = 0
        self._success = False
        self._first_success_step = self._max_steps
        self._best_score = 0.0
        self._best_zoom_score = 0.0

        try:
            import gymnasium as gymnasium
            # Registration side effect.
            from vizdoom import gymnasium_wrapper  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "ViZDoom support requires `pip install \"vizdoom>=1.3.0,<2.0\"`."
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
            except Exception as exc:  # allow v1/v0 compatibility fallback
                errors.append(f"{env_id}: {type(exc).__name__}: {exc}")
        if self._env is None:
            raise RuntimeError(
                f"Could not create ViZDoom scenario {task!r}. Tried:\n"
                + "\n".join(errors)
            )

        n = getattr(self._env.action_space, "n", None)
        if n is None:
            raise TypeError(
                f"{self._env_id} did not expose a Discrete action space. "
                "LS-Imagine uses max_buttons_pressed=1 so that the real-action "
                "branch remains categorical; the extra jump action is still "
                "reserved internally by its RSSM."
            )
        self.action_space = gym.spaces.Discrete(int(n))
        self.action_space.discrete = True

        # These shapes intentionally mirror LSImagineWrapper.  The replay item
        # for heatmap is HxW; WorldModel.preprocess appends the singleton
        # channel, exactly as in the MineDojo path.
        s = self._image_size
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, shape=(s, s, 3), dtype=np.uint8),
                "heatmap": gym.spaces.Box(0, 255, shape=(s, s, 1), dtype=np.uint8),
                "jump": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                "is_zoomed": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                "is_calculated": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                "is_first": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                "is_last": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                "is_terminal": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
                "reward_on_zoomed": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                "intrinsic": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                "intrinsic_on_zoomed": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                "score": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                "score_on_zoomed": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                "jumping_steps": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                "accumulated_reward": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            }
        )

    @property
    def env_id(self):
        return self._env_id

    @property
    def max_episode_steps(self):
        return self._max_steps

    @property
    def teacher_mode(self):
        return self._teacher_mode

    @property
    def intrinsic_mode(self):
        return self._intrinsic_mode

    def reset(self):
        if self._seed_pending:
            raw_obs, _info = self._env.reset(seed=self._seed)
            self._seed_pending = False
        else:
            raw_obs, _info = self._env.reset()
        self._episode_step = 0
        self._success = False
        self._first_success_step = self._max_steps
        self._best_score = 0.0
        self._best_zoom_score = 0.0
        return self._convert_obs(
            raw_obs,
            native_reward=0.0,
            is_first=True,
            is_last=False,
            is_terminal=False,
        )

    def step(self, action):
        raw_obs, native_reward, terminated, truncated, info = self._env.step(int(action))
        self._episode_step += 1
        native_reward = float(native_reward)

        success_now = native_reward >= self._success_threshold
        first_success_event = success_now and not self._success
        if first_success_event:
            self._success = True
            self._first_success_step = self._episode_step

        if self._reward_mode == "native":
            reward = native_reward
        else:
            reward = 1.0 if first_success_event else 0.0

        done = bool(terminated or truncated)
        is_terminal = bool(terminated)
        obs = self._convert_obs(
            raw_obs,
            native_reward=native_reward,
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
        info["vizdoom_teacher_mode"] = self._teacher_mode
        info["vizdoom_intrinsic_mode"] = self._intrinsic_mode
        return obs, np.float32(reward), done, info

    def _convert_obs(self, raw_obs, native_reward, is_first, is_last, is_terminal):
        image = self._screen(raw_obs)
        heatmap, score, target = self._teacher(image)

        zoomed_image = None
        heatmap_on_zoomed = None
        zoom_score = 0.0
        is_zoomed = False
        jump = False

        if target is not None:
            cx, cy = target
            zoomed_image = self._zoom_crop(image, cx, cy)
            heatmap_on_zoomed, zoom_score, _ = self._teacher(zoomed_image)
            is_zoomed = True
            jump = bool(zoom_score > score + self._jump_min_gain)

        intrinsic = 0.0
        intrinsic_on_zoomed = 0.0
        if self._intrinsic_mode == "score_improvement":
            if score > self._best_score:
                intrinsic = self._intrinsic_scale * score
                self._best_score = score
            if is_zoomed and zoom_score > self._best_zoom_score:
                intrinsic_on_zoomed = self._intrinsic_scale * zoom_score
                self._best_zoom_score = zoom_score

        return {
            "image": image,
            "heatmap": heatmap,
            "jump": np.asarray(jump, dtype=np.uint8),
            "is_zoomed": np.asarray(is_zoomed, dtype=np.uint8),
            "is_calculated": np.asarray(False, dtype=np.uint8),
            "is_first": np.asarray(is_first, dtype=np.uint8),
            "is_last": np.asarray(is_last, dtype=np.uint8),
            "is_terminal": np.asarray(is_terminal, dtype=np.uint8),
            "reward_on_zoomed": np.asarray(native_reward, dtype=np.float32),
            "intrinsic": np.asarray(intrinsic, dtype=np.float32),
            "intrinsic_on_zoomed": np.asarray(intrinsic_on_zoomed, dtype=np.float32),
            # Raw [0,1] semantic score is used only by ScoreStorage pairing.
            "score": np.asarray(score, dtype=np.float32),
            "score_on_zoomed": np.asarray(zoom_score, dtype=np.float32),
            "jumping_steps": np.asarray(float(self._max_steps), dtype=np.float32),
            "accumulated_reward": np.asarray(0.0, dtype=np.float32),
            "zoomed_image": zoomed_image,
            "heatmap_on_zoomed": heatmap_on_zoomed,
        }

    def _screen(self, raw_obs):
        if not isinstance(raw_obs, dict) or "screen" not in raw_obs:
            raise KeyError("ViZDoom Gymnasium observation must contain 'screen'.")
        image = np.asarray(raw_obs["screen"])
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        if image.shape[-1] > 3:
            image = image[..., :3]
        return cv2.resize(
            image.astype(np.uint8),
            (self._image_size, self._image_size),
            interpolation=cv2.INTER_AREA,
        )

    def _teacher(self, image):
        if self._teacher_mode == "none":
            return (
                np.zeros((self._image_size, self._image_size), dtype=np.uint8),
                0.0,
                None,
            )

        # Same RGB-only green-goal semantics used for the Stable V1.2 ViZDoom
        # adapter.  Hue/saturation/value produce a continuous affordance target.
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
        response = np.clip(response, 0.0, 1.0)

        flat = response.reshape(-1)
        k = max(1, int(math.ceil(0.02 * flat.size)))
        top = np.partition(flat, flat.size - k)[-k:]
        score = float(np.clip(np.mean(top), 0.0, 1.0))
        heatmap = np.round(response * 255.0).astype(np.uint8)

        # Zoom only compact, high-confidence green components.  This suppresses
        # broad green-ish wall/lighting regions while still reacting to the vest.
        binary = (response >= 0.35).astype(np.uint8)
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
        best = None
        best_quality = -1.0
        max_area = int(0.12 * self._image_size * self._image_size)
        for idx in range(1, num):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < 2 or area > max_area:
                continue
            component = response[labels == idx]
            quality = float(component.mean()) * math.sqrt(float(area))
            if quality > best_quality:
                best_quality = quality
                best = centroids[idx]

        target = None if best is None else (float(best[0]), float(best[1]))
        return heatmap, score, target

    def _zoom_crop(self, image, cx, cy):
        # Mild zoom: strong enough to define a long-term pseudo-state, but not
        # so strong that ScoreStorage can never find a later real observation
        # whose semantic score surpasses score_on_zoomed.
        s = self._image_size
        crop = max(8, min(s, int(round(s / self._zoom_factor))))
        half = crop // 2
        x0 = int(round(cx)) - half
        y0 = int(round(cy)) - half
        x0 = min(max(0, x0), s - crop)
        y0 = min(max(0, y0), s - crop)
        patch = image[y0 : y0 + crop, x0 : x0 + crop]
        return cv2.resize(patch, (s, s), interpolation=cv2.INTER_LINEAR)

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()


def make_env(task, **kwargs):
    return VizDoomLSImagineEnv(task, **kwargs)
