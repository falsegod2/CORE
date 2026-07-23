from abc import ABC
from collections import OrderedDict
import copy

import cv2
import gym.spaces as spaces
import numpy as np
from gym import Wrapper


BASIC_ACTIONS = {
    "noop": dict(),
    "attack": dict(attack=np.array(1)),
    "turn_up": dict(camera=np.array([-10.0, 0.0])),
    "turn_down": dict(camera=np.array([10.0, 0.0])),
    "turn_left": dict(camera=np.array([0.0, -10.0])),
    "turn_right": dict(camera=np.array([0.0, 10.0])),
    "forward": dict(forward=np.array(1)),
    "back": dict(back=np.array(1)),
    "left": dict(left=np.array(1)),
    "right": dict(right=np.array(1)),
    # This is the ordinary Minecraft jump action, not a latent long-term jump.
    "jump": dict(jump=np.array(1), forward=np.array(1)),
    "use": dict(use=np.array(1)),
}

NOOP_ACTION = {
    "camera": np.array([0.0, 0.0]),
    "smelt": "none",
    "craft": "none",
    "craft_with_table": "none",
    "forward": np.array(0),
    "back": np.array(0),
    "left": np.array(0),
    "right": np.array(0),
    "jump": np.array(0),
    "sneak": np.array(0),
    "sprint": np.array(0),
    "use": np.array(0),
    "attack": np.array(0),
    "drop": 0,
    "swap_slot": OrderedDict([("source_slot", 0), ("target_slot", 0)]),
    "pickItem": 0,
    "hotbar.1": 0,
    "hotbar.2": 0,
    "hotbar.3": 0,
    "hotbar.4": 0,
    "hotbar.5": 0,
    "hotbar.6": 0,
    "hotbar.7": 0,
    "hotbar.8": 0,
    "hotbar.9": 0,
}


class AgentWrapper(Wrapper, ABC):
    """Convert MineDojo observations/actions to the agent interface.

    This wrapper exposes the RGB observation and standard scalar fields only.
    It contains no auxiliary spatial-map or long-horizon transition logic.
    """

    def __init__(self, env, repeat=1, sticky_attack=0, sticky_jump=10, pitch_limit=(-70, 70)):
        super().__init__(env)
        self.wrapper_name = "AgentWrapper"
        self._noop_action = NOOP_ACTION
        actions = self._insert_defaults(BASIC_ACTIONS)
        self._action_names = tuple(actions.keys())
        self._action_values = tuple(actions.values())

        self.observation_space = spaces.Dict({
            "image": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "is_first": spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
            "is_last": spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
            "is_terminal": spaces.Box(-np.inf, np.inf, (1,), dtype=np.uint8),
            "mineclip_reward": spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            # Frozen global temporal MineCLIP feature emitted by ClipWrapper.
            # float16 limits replay growth; WorldModel.preprocess casts it to float32.
            "mineclip_embedding": spaces.Box(
                -np.inf, np.inf, (512,), dtype=np.float16
            ),
        })
        self.action_space = spaces.discrete.Discrete(len(BASIC_ACTIONS))
        self.action_space.discrete = True

        self._repeat = repeat
        self._sticky_attack_length = sticky_attack
        self._sticky_attack_counter = 0
        self._sticky_jump_length = sticky_jump
        self._sticky_jump_counter = 0
        self._pitch_limit = pitch_limit
        self._pitch = 0

    def reset(self):
        obs = self.env.reset()
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        obs = self._obs(obs)
        self._sticky_attack_counter = 0
        self._sticky_jump_counter = 0
        self._pitch = 0
        return obs

    def step(self, action):
        action = self._action(copy.deepcopy(self._action_values[action]))
        following = self._noop_action.copy()
        for key in ("attack", "forward", "back", "left", "right"):
            following[key] = action[key]
        for act in [action] + ([following] * (self._repeat - 1)):
            obs, reward, done, info = self.env.step(act)
            if "error" in info:
                done = True
                break
        obs["is_first"] = False
        obs["is_last"] = bool(done)
        obs["is_terminal"] = bool(info.get("is_terminal", info["real_done"]))
        return self._obs(obs), reward, done, info

    def _obs(self, obs):
        image = cv2.resize(obs["rgb"].transpose(1, 2, 0).astype(np.uint8), (64, 64))

        result = {
            "image": image,
            "is_first": np.asarray(obs["is_first"], dtype=np.uint8),
            "is_last": np.asarray(obs["is_last"], dtype=np.uint8),
            "is_terminal": np.asarray(obs["is_terminal"], dtype=np.uint8),
            "mineclip_reward": np.asarray(obs.get("mineclip_reward", 0.0), dtype=np.float32),
            "mineclip_embedding": np.asarray(
                obs.get("mineclip_embedding", np.zeros((512,), dtype=np.float16)),
                dtype=np.float16,
            ),
        }
        return result

    def _action(self, action):
        if self._sticky_attack_length:
            if action["attack"]:
                self._sticky_attack_counter = self._sticky_attack_length
            if self._sticky_attack_counter > 0:
                action["attack"] = np.array(1)
                action["jump"] = np.array(0)
                self._sticky_attack_counter -= 1
        if self._sticky_jump_length:
            if action["jump"]:
                self._sticky_jump_counter = self._sticky_jump_length
            if self._sticky_jump_counter > 0:
                action["jump"] = np.array(1)
                action["forward"] = np.array(1)
                self._sticky_jump_counter -= 1
        if self._pitch_limit and action["camera"][0]:
            lo, hi = self._pitch_limit
            if not (lo <= self._pitch + action["camera"][0] <= hi):
                action["camera"] = (0, action["camera"][1])
            self._pitch += action["camera"][0]
        return action

    def _insert_defaults(self, actions):
        actions = {name: action.copy() for name, action in actions.items()}
        for key, default in self._noop_action.items():
            for action in actions.values():
                action.setdefault(key, default)
        return actions
