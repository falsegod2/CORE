from gym import Wrapper
import torch as th


class ClipWrapper(Wrapper):
    def __init__(
        self,
        env,
        clip,
        prompts=None,
        dense_reward=.01,
        smoothing=1,
        target_object='log',
        emit_task_embedding=True,
        task_embedding_dtype='float16',
        **kwargs,
    ):
        super().__init__(env)
        self.clip = clip # ClipReward
        self.wrapper_name = "ClipWrapper"

        assert prompts is not None
        self.prompt = prompts
        self.expl_prompt = [f"Explore the widest possible area to find {target_object}"]
        self.dense_reward = dense_reward
        self.smoothing = smoothing
        
        self.buffer = None
        self._clip_state = None, None
        self.expl_buffer = None
        self._expl_clip_state = None, None
        self.last_score = 0
        self.expl_last_score = 0
        self.emit_task_embedding = bool(emit_task_embedding)
        self.task_embedding_dim = int(self.clip.feature_dim)
        if task_embedding_dtype not in ('float16', 'float32'):
            raise ValueError(
                f"Unsupported task embedding dtype: {task_embedding_dtype}"
            )
        self.task_embedding_dtype = task_embedding_dtype
        # Text is constant for a task, so encode it exactly once.
        self._task_embedding = self.clip.get_task_embedding(self.prompt)

    def _format_task_embedding(self):
        import numpy as np
        dtype = (
            np.float16
            if self.task_embedding_dtype == 'float16'
            else np.float32
        )
        array = self._task_embedding.numpy().astype(dtype, copy=False)
        if array.shape != (self.task_embedding_dim,):
            raise RuntimeError(
                f"Expected task embedding ({self.task_embedding_dim},), "
                f"got {array.shape}"
            )
        return array

    def reset(self, **kwargs):
        self._clip_state = None, self._clip_state[1]
        self._expl_clip_state = None, self._expl_clip_state[1]
        self.buffer = None
        self.expl_buffer = None
        self.last_score = 0
        self.expl_last_score = 0

        obs = self.env.reset(**kwargs)
        obs['mineclip_reward'] = 0.0
        if self.emit_task_embedding:
            obs['task_embedding'] = self._format_task_embedding()

        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.emit_task_embedding:
            obs['task_embedding'] = self._format_task_embedding()

        if len(self.prompt) > 0:
            logits, self._clip_state = self.clip.get_logits(obs, self.prompt, self._clip_state)
            logits = logits.detach().cpu()

            self.buffer = self._insert_buffer(self.buffer, logits[:1])
            score = self._get_score()

            if score > self.last_score:
                obs['mineclip_reward'] = self.dense_reward * score
                self.last_score = score
            else:
                obs['mineclip_reward'] = 0.0


        else:
            obs['mineclip_reward'] = 0.0
    
        if len(self.expl_prompt) > 0:
            logits, self._expl_clip_state = self.clip.get_logits(obs, self.expl_prompt, self._expl_clip_state)
            logits = logits.detach().cpu()

            self.expl_buffer = self._insert_buffer(self.expl_buffer, logits[:1])
            expl_score = self._get_expl_score()

            if expl_score > self.expl_last_score:
                info['expl_intrinsic'] = self.dense_reward * expl_score
                self.expl_last_score = expl_score
            else:
                info['expl_intrinsic'] = 0.0

        else:
            info['expl_intrinsic'] = 0.0


        return obs, reward, done, info 

    def _get_score(self):
        score = th.mean(self.buffer)
        return (1 / (1 + th.exp(1.2 * (21.8 - score)))).item()
    
    def _get_expl_score(self):
        score = th.mean(self.expl_buffer)
        return (1 / (1 + th.exp(1.2 * (21.8 - score)))).item()

    def _insert_buffer(self, buffer, logits):
        if buffer is None:
            buffer = logits.unsqueeze(0)
        elif buffer.shape[0] < self.smoothing:
            buffer = th.cat([buffer, logits.unsqueeze(0)], dim=0)
        else:
            buffer = th.cat([buffer[1:], logits.unsqueeze(0)], dim=0)
        return buffer
