"""
Value Iteration Agent
=====================
Model-based dynamic programming that iterates on the Bellman optimality equation
to find the optimal value function V*(s) and derive π*(s).
"""

import numpy as np
from typing import Dict, Any, Optional

from .base import BaseAgent


class ValueIterationAgent(BaseAgent):
    """
    Value Iteration (Bellman optimality sweeps).

    Requires access to the environment's transition model P[s][a].

    Parameters
    ----------
    n_states : int
    n_actions : int
    gamma : float
        Discount factor.
    theta : float
        Convergence threshold — stops when max|V_new - V_old| < theta.
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = 0.99,
        theta: float = 1e-8,
    ):
        super().__init__(n_states, n_actions, gamma)
        self.theta = theta
        self.V = np.zeros(n_states)
        self.policy = np.zeros(n_states, dtype=int)
        self.converged = False
        self.n_iterations = 0

    def train(self, env, max_iterations: int = 10_000, **kwargs) -> Dict[str, Any]:
        """
        Run value iteration until convergence.

        Parameters
        ----------
        env : GridWorld
            Must expose env.P (transition model), env.n_states, env.n_actions.
        max_iterations : int

        Returns
        -------
        history dict with delta per iteration.
        """
        P = env.P
        V = np.zeros(self.n_states)
        deltas = []

        for i in range(max_iterations):
            delta = 0.0
            for s in range(self.n_states):
                v_old = V[s]
                # Bellman optimality update
                action_values = self._compute_action_values(P, V, s)
                V[s] = np.max(action_values)
                delta = max(delta, abs(V[s] - v_old))

            deltas.append(delta)
            self.n_iterations = i + 1

            if delta < self.theta:
                self.converged = True
                break

        self.V = V
        self._extract_policy(P)

        history = {
            "deltas": deltas,
            "n_iterations": self.n_iterations,
            "converged": self.converged,
            "final_delta": deltas[-1] if deltas else None,
        }
        self.training_history = history
        return history

    def _compute_action_values(self, P, V: np.ndarray, s: int) -> np.ndarray:
        """Compute Q(s, a) for all actions using the model."""
        q_values = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            for prob, ns, reward, done in P[s][a]:
                q_values[a] += prob * (reward + (0 if done else self.gamma * V[ns]))
        return q_values

    def _extract_policy(self, P):
        """Derive greedy policy from converged value function."""
        for s in range(self.n_states):
            action_values = self._compute_action_values(P, self.V, s)
            self.policy[s] = int(np.argmax(action_values))

    def select_action(self, state: int) -> int:
        return int(self.policy[state])

    def get_policy(self) -> np.ndarray:
        return self.policy.copy()

    def get_value_function(self) -> np.ndarray:
        return self.V.copy()

    def get_action_values(self, env) -> np.ndarray:
        """Return full Q(s, a) table from the converged value function."""
        Q = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            Q[s] = self._compute_action_values(env.P, self.V, s)
        return Q
