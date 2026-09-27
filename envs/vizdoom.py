"""ViZDoom adapter for ISO3 / Stable V1.2.

This module keeps ViZDoom-specific logic outside the world model.  The agent
still receives RGB only through the encoder.  ``heatmap`` and ``task_score``
are training-only teacher targets, mirroring the MineDojo affordance pathway.

The repository itself uses the legacy Gym 4-tuple API while ViZDoom >=1.3
exposes Gymnasium.  This adapter owns the Gymnasium environment and exports the
same API expected by ``expr.py`` and ``tools.simulate``.
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
        # ViZDoom 1.3 release/docs have used both names across revisions.
        "env_ids": (
            "VizdoomDeadlyCorridor-v1",
            "VizdoomCorridor-v1",
            "VizdoomCorridor-v0",
        ),
        "timeout_tics": 2100,
        "success_threshold": 500.0,
        "teacher": "green_goal",
    },
    "predict_position": {
        "env_ids": ("VizdoomPredictPosition-v1", "VizdoomPredictPosition-v0"),
        "timeout_tics": 300,
        "success_threshold": 0.5,
        # RGB-only temporal teacher: isolate the moving target after compensating
        # the dominant camera shift between consecutive frames.
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
        raise KeyError(f"Unknown ViZDoom task {task!r}. Available: {options}")
    return task


def available_tasks():
    return tuple(sorted(_SCENARIOS))


def episode_max_steps(task, frame_skip=1):
    task = _normalize_task(task)
    frame_skip = max(1, int(frame_skip))
    return int(math.ceil(_SCENARIOS[task]["timeout_tics"] / frame_skip))


def default_teacher_mode(task):
    task = _normalize_task(task)
    return _SCENARIOS[task]["teacher"]


class VizDoomStableEnv:
    """Pixel-only ViZDoom environment with Stable V1.2 auxiliary teachers.

    The *policy/world-model encoder* receives only ``image``.  ``heatmap`` and
    ``task_score`` are replay-only auxiliary supervision.  For MyWayHome and
    DeadlyCorridor, ``green_goal`` is a simple task-specific visual teacher.
    PredictPosition uses ``moving_target``, a temporal RGB-motion teacher with
    camera-shift compensation.  Neither mode uses depth, labels, automap, game
    variables, or simulator state.
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
        self._reward_mode = str(reward_mode).strip().lower()
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
                "vizdoom_teacher_mode must be 'auto', 'green_goal', "
                "'moving_target', or 'none', "
                f"got {teacher_mode!r}"
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
            from vizdoom import gymnasium_wrapper  # noqa: F401  (registration)
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
            except Exception as exc:
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
                "This adapter uses max_buttons_pressed=1 for a compact "
                "categorical action space shared by all algorithms."
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
                # Keep the declared channel dimension identical to the
                # MineDojo Stable wrapper / decoder contract.  The actual
                # replay item is HxW and WorldModel.preprocess appends channel 1.
                "heatmap": gym.spaces.Box(
                    0, 255,
                    shape=(self._image_size, self._image_size, 1),
                    dtype=np.uint8,
                ),
                "is_first": gym.spaces.Box(0, 1, shape=(), dtype=np.uint8),
                "is_last": gym.spaces.Box(0, 1, shape=(), dtype=np.uint8),
                "is_terminal": gym.spaces.Box(0, 1, shape=(), dtype=np.uint8),
                "intrinsic": gym.spaces.Box(
                    -np.inf, np.inf, shape=(), dtype=np.float32
                ),
                "task_score": gym.spaces.Box(
                    0.0, 1.0, shape=(1,), dtype=np.float32
                ),
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
        return self._convert_obs(
            raw_obs, is_first=True, is_last=False, is_terminal=False
        )

    def step(self, action):
        raw_obs, native_reward, terminated, truncated, info = self._env.step(
            int(action)
        )
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
        return obs, np.float32(reward), done, info

    def _convert_obs(self, raw_obs, is_first, is_last, is_terminal):
        if not isinstance(raw_obs, dict) or "screen" not in raw_obs:
            raise KeyError(
                "ViZDoom Gymnasium observation must contain the 'screen' buffer"
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
            "heatmap": heatmap,
            "is_first": np.asarray(is_first, dtype=np.uint8),
            "is_last": np.asarray(is_last, dtype=np.uint8),
            "is_terminal": np.asarray(is_terminal, dtype=np.uint8),
            # ViZDoom Stable main profile is deliberately no-intrinsic.  This
            # key is retained only because the model head/API exists.
            "intrinsic": np.asarray(0.0, dtype=np.float32),
            "task_score": np.asarray([task_score], dtype=np.float32),
        }

    @staticmethod
    def _top_score(response, fraction=0.02):
        flat = np.asarray(response, dtype=np.float32).reshape(-1)
        k = max(1, int(math.ceil(float(fraction) * flat.size)))
        top = np.partition(flat, flat.size - k)[-k:]
        return float(np.clip(np.mean(top), 0.0, 1.0))

    def _green_goal_teacher(self, image):
        # Continuous RGB-only green-goal teacher.  MyWayHome's target is a
        # green vest.  This is intentionally a *teacher target only*: neither
        # heatmap nor task_score enters the RGB encoder or actor inputs.
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hue = hsv[..., 0]
        sat = hsv[..., 1] / 255.0
        val = hsv[..., 2] / 255.0

        # OpenCV hue is [0,179].  Goal-green is centred near H=60.  Circular
        # hue distance plus saturation/value suppresses grey/brown walls.
        hue_dist = np.abs(hue - 60.0)
        hue_dist = np.minimum(hue_dist, 180.0 - hue_dist)
        hue_score = np.exp(-0.5 * (hue_dist / 16.0) ** 2)
        response = hue_score * np.power(sat, 1.5) * np.power(val, 0.75)
        response = np.clip(response, 0.0, 1.0)
        response = cv2.GaussianBlur(response, (0, 0), sigmaX=1.0)
        response = np.clip(response, 0.0, 1.0)
        return response

    def _moving_target_teacher(self, image):
        """RGB-only temporal target evidence for PredictPosition.

        PredictPosition's relevant object is the moving target.  Instead of
        simulator labels, estimate it from consecutive RGB frames.  We first
        compensate the dominant global image translation (camera turn) using
        phase correlation, then retain localized residual motion.  This keeps
        the teacher training-only and pixel-only while providing non-trivial
        task semantics to S-Aff/Proto.
        """
        current = image.astype(np.float32) / 255.0
        previous_u8 = self._teacher_prev_image
        if previous_u8 is None:
            return np.zeros((self._image_size, self._image_size), np.float32)

        previous = previous_u8.astype(np.float32) / 255.0
        cur_gray = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY).astype(np.float32)
        prev_gray = cv2.cvtColor(previous, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # Estimate the dominant camera-induced translation.  Test both signs
        # because OpenCV's returned phase shift convention is easy to invert
        # when thinking in source/destination coordinates; choose the alignment
        # that actually minimizes robust residual error.
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
                previous,
                matrix,
                (self._image_size, self._image_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )
            residual = np.mean(np.abs(current - aligned), axis=-1)
            candidates.append((float(np.median(residual)), residual))
        residual = min(candidates, key=lambda x: x[0])[1]

        # Remove widespread background/camera residual and emphasize compact,
        # high-motion regions.  Saturation softly favors the rendered monster
        # over grey walls without hard-coding a particular enemy RGB color.
        lo = float(np.quantile(residual, 0.70))
        hi = float(np.quantile(residual, 0.995))
        denom = max(hi - lo, 1.0e-4)
        response = np.clip((residual - lo) / denom, 0.0, 1.0)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        sat = hsv[..., 1] / 255.0
        response *= (0.35 + 0.65 * np.sqrt(np.clip(sat, 0.0, 1.0)))
        response = cv2.GaussianBlur(response, (0, 0), sigmaX=1.0)
        # Suppress tiny isolated numerical speckles.
        response[response < 0.08] = 0.0
        return np.clip(response, 0.0, 1.0)

    def _teacher(self, image):
        if self._teacher_mode == "none":
            response = np.zeros((self._image_size, self._image_size), dtype=np.float32)
        elif self._teacher_mode == "green_goal":
            response = self._green_goal_teacher(image)
        elif self._teacher_mode == "moving_target":
            response = self._moving_target_teacher(image)
        else:
            raise RuntimeError(f"Unhandled teacher mode {self._teacher_mode!r}")

        task_score = self._top_score(response, fraction=0.02)
        heatmap = np.round(np.clip(response, 0.0, 1.0) * 255.0).astype(np.uint8)
        # Save the current observation only after constructing the teacher for
        # this transition.  The next step therefore compares consecutive frames.
        self._teacher_prev_image = image.copy()
        return heatmap, np.float32(task_score)

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()


def make_env(task, **kwargs):
    return VizDoomStableEnv(task, **kwargs)
