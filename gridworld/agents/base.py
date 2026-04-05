"""Base class for all Grid World agents."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseAgent(ABC):
    """Abstract base class for reinforcement learning agents."""

    def __init__(self, n_states: int, n_actions: int, gamma: float = 0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.training_history: list = []

    @abstractmethod
    def select_action(self, state: int) -> int:
        """Select an action given the current state."""
        ...

    @abstractmethod
    def train(self, env, **kwargs) -> dict:
        """Train the agent on an environment."""
        ...

    def get_policy(self) -> np.ndarray:
        """Return the greedy policy as an array of actions (one per state)."""
        raise NotImplementedError

    def get_value_function(self) -> np.ndarray:
        """Return the state value function V(s)."""
        raise NotImplementedError
