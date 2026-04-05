"""
Q-Learning Agent
================
Model-free, off-policy temporal difference control.
Learns the optimal action-value function Q*(s, a) from experience.
"""

import numpy as np
from typing import Optional, Dict, Any

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # fallback no-op wrapper
        return iterable

from .base import BaseAgent


class QLearningAgent(BaseAgent):
    """
    Tabular Q-Learning with ε-greedy exploration and optional learning rate decay.

    Parameters
    ----------
    n_states : int
    n_actions : int
    alpha : float
        Initial learning rate (step size).
    gamma : float
        Discount factor.
    epsilon : float
        Initial exploration probability.
    epsilon_min : float
        Minimum exploration probability after decay.
    epsilon_decay : float
        Multiplicative decay applied to epsilon each episode.
    alpha_decay : float
        Multiplicative decay applied to alpha each episode (1.0 = no decay).
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        alpha_decay: float = 1.0,
    ):
        super().__init__(n_states, n_actions, gamma)
        self.alpha = alpha
        self.alpha_init = alpha
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha_decay = alpha_decay

        # Q-table: shape (n_states, n_actions)
        self.Q = np.zeros((n_states, n_actions))

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def select_action(self, state: int) -> int:
        """ε-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """Single Q-learning update step."""
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.Q[next_state])
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error

    def train(
        self,
        env,
        n_episodes: int = 1000,
        seed: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the agent via Q-learning.

        Returns a history dict with per-episode rewards, lengths, and epsilon values.
        """
        if seed is not None:
            np.random.seed(seed)

        history = {
            "episode_rewards": [],
            "episode_lengths": [],
            "epsilons": [],
            "alphas": [],
        }

        iterator = tqdm(range(n_episodes), desc="Q-Learning", disable=not verbose)
        for episode in iterator:
            state = env.reset()
            total_reward = 0.0
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                self.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward

            # Decay exploration and learning rate
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self.alpha = max(1e-4, self.alpha * self.alpha_decay)

            history["episode_rewards"].append(total_reward)
            history["episode_lengths"].append(env.steps_taken)
            history["epsilons"].append(self.epsilon)
            history["alphas"].append(self.alpha)

            if verbose and (episode + 1) % 100 == 0:
                avg = np.mean(history["episode_rewards"][-100:])
                iterator.set_postfix(avg_reward=f"{avg:.3f}", epsilon=f"{self.epsilon:.3f}")

        self.training_history = history
        return history

    # ------------------------------------------------------------------
    # Policy / value extraction
    # ------------------------------------------------------------------

    def get_policy(self) -> np.ndarray:
        """Greedy policy derived from Q-table."""
        return np.argmax(self.Q, axis=1)

    def get_value_function(self) -> np.ndarray:
        """State values V(s) = max_a Q(s, a)."""
        return np.max(self.Q, axis=1)

    def get_q_table(self) -> np.ndarray:
        return self.Q.copy()

    def reset(self):
        """Reset Q-table and exploration parameters."""
        self.Q = np.zeros((self.n_states, self.n_actions))
        self.epsilon = 1.0
        self.alpha = self.alpha_init
        self.training_history = []
