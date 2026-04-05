"""
Policy Iteration Agent
======================
Model-based dynamic programming alternating between:
  1. Policy Evaluation  — solve V^π exactly (or approximately)
  2. Policy Improvement — act greedily w.r.t. V^π

Guaranteed to converge to the optimal policy in finite steps.
"""

import numpy as np
from typing import Dict, Any

from .base import BaseAgent


class PolicyIterationAgent(BaseAgent):
    """
    Policy Iteration via iterative policy evaluation.

    Parameters
    ----------
    n_states : int
    n_actions : int
    gamma : float
        Discount factor.
    eval_theta : float
        Convergence threshold for the inner policy evaluation loop.
    max_eval_steps : int
        Max sweeps during policy evaluation (use np.inf for exact solution).
    """

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float = 0.99,
        eval_theta: float = 1e-8,
        max_eval_steps: int = 10_000,
    ):
        super().__init__(n_states, n_actions, gamma)
        self.eval_theta = eval_theta
        self.max_eval_steps = max_eval_steps

        self.V = np.zeros(n_states)
        self.policy = np.zeros(n_states, dtype=int)  # start with all UP
        self.converged = False
        self.n_policy_updates = 0

    def train(self, env, max_policy_iterations: int = 1_000, **kwargs) -> Dict[str, Any]:
        """
        Run policy iteration until the policy is stable.

        Parameters
        ----------
        env : GridWorld
        max_policy_iterations : int

        Returns
        -------
        history dict.
        """
        P = env.P
        history = {
            "policy_changes": [],
            "eval_iters_per_update": [],
            "n_policy_updates": 0,
        }

        for pi_step in range(max_policy_iterations):
            # --- Policy Evaluation ---
            eval_iters = self._policy_evaluation(P)
            history["eval_iters_per_update"].append(eval_iters)

            # --- Policy Improvement ---
            policy_stable, n_changes = self._policy_improvement(P)
            history["policy_changes"].append(n_changes)
            self.n_policy_updates = pi_step + 1

            if policy_stable:
                self.converged = True
                break

        history["n_policy_updates"] = self.n_policy_updates
        history["converged"] = self.converged
        self.training_history = history
        return history

    def _policy_evaluation(self, P) -> int:
        """Iteratively evaluate V^π until convergence. Returns number of sweeps."""
        for sweep in range(self.max_eval_steps):
            delta = 0.0
            for s in range(self.n_states):
                v_old = self.V[s]
                a = self.policy[s]
                self.V[s] = sum(
                    prob * (reward + (0 if done else self.gamma * self.V[ns]))
                    for prob, ns, reward, done in P[s][a]
                )
                delta = max(delta, abs(self.V[s] - v_old))
            if delta < self.eval_theta:
                return sweep + 1
        return self.max_eval_steps

    def _policy_improvement(self, P):
        """
        Greedily improve policy w.r.t. current V.
        Returns (stable, n_state_changes).
        """
        policy_stable = True
        n_changes = 0
        for s in range(self.n_states):
            old_action = self.policy[s]
            # Compute Q(s, a) for all a
            q_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                q_values[a] = sum(
                    prob * (reward + (0 if done else self.gamma * self.V[ns]))
                    for prob, ns, reward, done in P[s][a]
                )
            self.policy[s] = int(np.argmax(q_values))
            if self.policy[s] != old_action:
                policy_stable = False
                n_changes += 1
        return policy_stable, n_changes

    def select_action(self, state: int) -> int:
        return int(self.policy[state])

    def get_policy(self) -> np.ndarray:
        return self.policy.copy()

    def get_value_function(self) -> np.ndarray:
        return self.V.copy()
