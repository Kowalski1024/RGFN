from typing import Callable, List

import gin
import torch
from torch import nn
from torch.distributions import Categorical
from torchtyping import TensorType

from rgfn.api.policy_base import PolicyBase

from .hypergrid_api import HyperGridAction, HyperGridActionSpace, HyperGridState
from .hypergrid_env import HyperGridEnv


@gin.configurable
class ForwardHyperGridPolicy(
    PolicyBase[HyperGridState, HyperGridActionSpace, HyperGridAction], nn.Module
):
    """
    A hypergrid policy that samples actions from the action space using a learned MLP.
    """

    def __init__(self, size: int, n_dimensions: int, env: HyperGridEnv, hidden_dim: int = 32):
        """
        Args:
            size: the size of the hyper-grid world.
            n_dimensions: the number of dimensions of the hyper-grid world.
            env: the environment.
            hidden_dim: the hidden dimension of the MLP.
        """
        super().__init__()
        self.size = size
        self.n_dimensions = n_dimensions
        self.num_actions = env.num_actions
        self.hidden_dim = hidden_dim
        self.score_mlp: Callable[[TensorType[float]], TensorType[float]] = nn.Sequential(
            nn.Linear(n_dimensions, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.num_actions),
        )
        self.device = "cpu"

    def forward(
        self, states: List[HyperGridState], action_spaces: List[HyperGridActionSpace]
    ) -> TensorType[float]:
        """
        Compute the log probabilities of actions for the given states.

        Args:
            states: a list of states of length `n_states`.
            action_spaces: a list of action spaces of length `n_states`. An action space describes the possible actions
                that can be taken in a given state.

        Returns:
            a tensor of log probabilities of shape `(n_states, num_actions)`.
        """
        state_encodings = torch.tensor(
            [state.coords for state in states], dtype=torch.float, device=self.device
        )
        logits = self.score_mlp(state_encodings)  # (batch_size, num_actions)
        mask = torch.tensor(
            [s.possible_actions_mask for s in action_spaces],
            dtype=torch.bool,
            device=self.device,
        )
        logits = logits.masked_fill(~mask, float("-inf"))
        return torch.log_softmax(logits, dim=1).float()

    def sample_actions(
        self, states: List[HyperGridState], action_spaces: List[HyperGridActionSpace]
    ) -> List[HyperGridAction]:
        """
        Sample actions for the given states and action spaces.

        Args:
            states: a list of states of length `n_states`.
            action_spaces: a list of action spaces of length `n_states`. An action space describes the possible actions
                that can be taken in a given state.

        Returns:
            a list of actions of length `n_states`.
        """
        log_probs = self.forward(states, action_spaces)
        probs = torch.exp(log_probs)
        action_indices = Categorical(probs=probs).sample()
        return [
            action_space.get_action_at_idx(idx)
            for action_space, idx in zip(action_spaces, action_indices)
        ]

    def compute_action_log_probs(
        self,
        states: List[HyperGridState],
        action_spaces: List[HyperGridActionSpace],
        actions: List[HyperGridAction],
        shared_embeddings: None = None,
    ) -> TensorType[float]:
        """
        Compute the log probabilities of the given actions take in the given states (and corresponding action spaces).

        Args:
            states: a list of states of length `n_states`.
            action_spaces: a list of action spaces of length `n_states`. An action space describes the possible actions
                that can be taken in a given state.
            actions: a list of actions chosen in the given states of length `n_states`.

        Returns:
            a tensor of log probabilities of shape `(n_states,)`.
        """
        log_probs = self.forward(states, action_spaces)
        return torch.stack([log_probs[i, action.idx] for i, action in enumerate(actions)]).float()

    def set_device(self, device: str) -> None:
        self.to(device)
        self.device = device

    def compute_states_log_flow(self, states: List[HyperGridState]) -> TensorType[float]:
        raise NotImplementedError()
