"""Two independent RSSM branches behind the upstream PyTorch RSSM interface.

S receives the real action. Z receives a zero-width action tensor unless
``both_action`` is enabled. The public state retains the upstream ``deter``,
``stoch`` and ``logit`` keys, so prediction heads and behavior learning are
shared with the single-branch baseline. This does not establish a causal
disentanglement: posterior Z can still contain effects of earlier actions.
"""

import torch
from torch import nn

from ..vendor.dreamerv3_torch.networks import RSSM


class DualRSSM(RSSM):
    """Split total deterministic, stochastic and hidden widths equally.

    KL is computed on the joint categorical state before applying free nats,
    giving single and dual models the same total free-nats allowance.
    ``seed`` controls local branch initialization without consuming the global
    RNG. Z's common weights are identical with and without its action input.
    """

    def __init__(
        self,
        stoch=32,
        deter=512,
        hidden=512,
        rec_depth=1,
        discrete=32,
        act="SiLU",
        norm=True,
        mean_act="none",
        std_act="softplus",
        min_std=0.1,
        unimix_ratio=0.01,
        initial="learned",
        num_actions=None,
        embed=None,
        device=None,
        *,
        both_action=False,
        seed=None,
    ):
        nn.Module.__init__(self)
        for name, value in (("stoch", stoch), ("deter", deter), ("hidden", hidden)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 2 or value % 2:
                raise ValueError(f"DualRSSM {name} must be a positive even integer; got {value!r}.")
        if isinstance(discrete, bool) or not isinstance(discrete, int) or discrete < 2:
            raise ValueError("DualRSSM requires categorical latent states (discrete >= 2).")
        if isinstance(num_actions, bool) or not isinstance(num_actions, int) or num_actions < 1:
            raise ValueError("num_actions must be the positive number of real actions.")
        if isinstance(embed, bool) or not isinstance(embed, int) or embed < 1:
            raise ValueError("embed must be a positive encoder feature width.")
        if rec_depth != 1:
            raise ValueError("This pinned upstream RSSM supports rec_depth=1; use an explicit new implementation for deeper recurrence.")
        if not isinstance(both_action, bool):
            raise TypeError("both_action must be a bool.")
        if initial not in ("zeros", "learned"):
            raise ValueError("initial must be 'zeros' or 'learned'.")
        if not 0 <= unimix_ratio < 1:
            raise ValueError("unimix_ratio must be in [0, 1).")

        self._stoch = stoch
        self._deter = deter
        self._hidden = hidden
        self._discrete = discrete
        self._num_actions = num_actions
        self._embed = embed
        self._device = torch.device("cpu" if device is None else device)
        self._unimix_ratio = unimix_ratio
        self._initial = initial
        self._rec_depth = rec_depth
        self.both_action = both_action
        branch_args = dict(
            stoch=stoch // 2, deter=deter // 2, hidden=hidden // 2,
            rec_depth=rec_depth, discrete=discrete, act=act, norm=norm,
            mean_act=mean_act, std_act=std_act, min_std=min_std,
            unimix_ratio=unimix_ratio, initial=initial,
            num_actions=num_actions, embed=embed, device=self._device,
        )
        base_seed = int(torch.initial_seed() if seed is None else seed)
        # Layers are initialized on CPU, then the caller moves the entire model.
        # Seed only the CPU generator; do not disturb CUDA generators either.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(base_seed)
            self.s = RSSM(**branch_args)
            torch.random.default_generator.manual_seed((base_seed + 1) % (2**63 - 1))
            self.z = RSSM(**branch_args)
            if not both_action:
                # Initialize both variants with the same architecture first.
                # Removing only the action columns preserves every common weight.
                original = self.z._img_in_layers[0]
                latent_width = (stoch // 2) * discrete
                narrowed = nn.Linear(latent_width, hidden // 2, bias=False)
                with torch.no_grad():
                    narrowed.weight.copy_(original.weight[:, :latent_width])
                self.z._img_in_layers[0] = narrowed
                self.z._num_actions = 0

    def _split_state(self, state):
        self._validate_state(state)
        left, right = {}, {}
        for key, value in state.items():
            axis = -1 if key == "deter" else -2
            left[key], right[key] = torch.chunk(value, 2, dim=axis)
        return left, right

    @staticmethod
    def _merge_state(left, right):
        return {
            key: torch.cat([left[key], right[key]], dim=-1 if key == "deter" else -2)
            for key in left
        }

    def _validate_state(self, state):
        if not isinstance(state, dict) or set(state) != {"deter", "stoch", "logit"}:
            raise ValueError("An RSSM state must contain exactly deter, stoch and logit.")
        if state["deter"].ndim < 2 or state["deter"].shape[-1] != self._deter:
            raise ValueError(f"Expected total deter width {self._deter}.")
        for key in ("stoch", "logit"):
            if state[key].shape[-2:] != (self._stoch, self._discrete):
                raise ValueError(f"Expected {key} trailing shape {(self._stoch, self._discrete)}.")
            if state[key].shape[:-2] != state["deter"].shape[:-1]:
                raise ValueError("All RSSM state tensors must have the same batch shape.")

    def _validate_action(self, action, batch_shape):
        if action is None or tuple(action.shape) != tuple(batch_shape) + (self._num_actions,):
            actual = None if action is None else tuple(action.shape)
            raise ValueError(f"Expected real-action shape {tuple(batch_shape) + (self._num_actions,)}, got {actual}.")

    def initial(self, batch_size):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        return self._merge_state(self.s.initial(batch_size), self.z.initial(batch_size))

    def img_step(self, prev_state, prev_action, sample=True):
        left, right = self._split_state(prev_state)
        self._validate_action(prev_action, prev_state["deter"].shape[:-1])
        z_action = prev_action if self.both_action else prev_action[..., :0]
        return self._merge_state(
            self.s.img_step(left, prev_action, sample=sample),
            self.z.img_step(right, z_action, sample=sample),
        )

    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        if embed.ndim != 2 or embed.shape[-1] != self._embed:
            raise ValueError(f"Expected embed shape (batch, {self._embed}).")
        batch_size = embed.shape[0]
        if tuple(is_first.shape) != (batch_size,):
            raise ValueError("is_first must have shape (batch,).")
        if prev_action is None and prev_state is None:
            prev_action = embed.new_zeros(batch_size, self._num_actions)
        self._validate_action(prev_action, (batch_size,))
        if prev_state is None:
            prev_state = self.initial(batch_size)
            prev_action = torch.zeros_like(prev_action)
        else:
            self._validate_state(prev_state)
            if prev_state["deter"].shape[:-1] != (batch_size,):
                raise ValueError("Previous state and embed batch sizes must match.")
            first = is_first.to(dtype=torch.bool)
            initial = self.initial(batch_size)
            prev_state = {
                key: torch.where(first.reshape((batch_size,) + (1,) * (value.ndim - 1)), initial[key], value)
                for key, value in prev_state.items()
            }
            prev_action = torch.where(first[:, None], torch.zeros_like(prev_action), prev_action)
        # The upstream obs_step always samples its prior even when sample=False.
        # Respect sample=False on both paths for controlled dependency diagnostics.
        prior = self.img_step(prev_state, prev_action, sample=sample)
        s_prior, z_prior = self._split_state(prior)
        posts = []
        for branch, branch_prior in ((self.s, s_prior), (self.z, z_prior)):
            hidden = branch._obs_out_layers(torch.cat([branch_prior["deter"], embed], dim=-1))
            stats = branch._suff_stats_layer("obs", hidden)
            distribution = branch.get_dist(stats)
            stoch = distribution.sample() if sample else distribution.mode()
            posts.append(dict(stoch=stoch, deter=branch_prior["deter"], **stats))
        return self._merge_state(*posts), prior

    def get_stoch(self, deter):
        if deter.shape[-1] != self._deter:
            raise ValueError(f"Expected total deter width {self._deter}.")
        s_deter, z_deter = torch.chunk(deter, 2, dim=-1)
        return torch.cat([self.s.get_stoch(s_deter), self.z.get_stoch(z_deter)], dim=-2)

    # get_feat, get_dist, observe, imagine_with_action and kl_loss intentionally
    # inherit the upstream implementations on the merged total-width state.
    # In particular, KL sums over ALL categorical variables and clamps once.
