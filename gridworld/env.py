"""
GridWorld Environment
=====================
A customizable grid-world environment for reinforcement learning experiments.
Supports walls, goals, traps, stochastic transitions, and partial observability.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from enum import IntEnum


class Action(IntEnum):
    UP    = 0
    DOWN  = 1
    LEFT  = 2
    RIGHT = 3


class CellType(IntEnum):
    EMPTY = 0
    WALL  = 1
    GOAL  = 2
    TRAP  = 3
    START = 4


# Movement deltas: (row_delta, col_delta)
ACTION_DELTAS = {
    Action.UP:    (-1,  0),
    Action.DOWN:  ( 1,  0),
    Action.LEFT:  ( 0, -1),
    Action.RIGHT: ( 0,  1),
}

ACTION_NAMES = {
    Action.UP:    "↑",
    Action.DOWN:  "↓",
    Action.LEFT:  "←",
    Action.RIGHT: "→",
}


class GridWorld:
    """
    A flexible Grid World environment for tabular RL algorithms.

    Parameters
    ----------
    grid : list of list of int, optional
        2D grid layout using CellType values. If None, a default 5x5 grid is used.
    slip_prob : float
        Probability of slipping to a random perpendicular action (stochasticity).
    step_reward : float
        Reward for each non-terminal step.
    goal_reward : float
        Reward for reaching a goal cell.
    trap_reward : float
        Reward (penalty) for stepping into a trap cell.
    wall_reward : float
        Reward for bumping into a wall (agent stays in place).
    max_steps : int
        Maximum number of steps per episode.
    """

    PRESETS: Dict[str, List[List[int]]] = {
        "simple_5x5": [
            [4, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 2],
        ],
        "trap_maze": [
            [4, 0, 0, 3, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 3, 1, 0],
            [0, 0, 0, 0, 2],
        ],
        "cliff_walk": [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2],
        ],
    }

    def __init__(
        self,
        grid: Optional[List[List[int]]] = None,
        slip_prob: float = 0.0,
        step_reward: float = -0.04,
        goal_reward: float = 1.0,
        trap_reward: float = -1.0,
        wall_reward: float = -0.01,
        max_steps: int = 200,
    ):
        if grid is None:
            grid = self.PRESETS["simple_5x5"]

        self.grid = np.array(grid, dtype=np.int32)
        self.n_rows, self.n_cols = self.grid.shape
        self.n_states = self.n_rows * self.n_cols
        self.n_actions = len(Action)

        self.slip_prob = slip_prob
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.trap_reward = trap_reward
        self.wall_reward = wall_reward
        self.max_steps = max_steps

        # Identify special cells
        self.start_states = self._find_cells(CellType.START)
        self.goal_states  = self._find_cells(CellType.GOAL)
        self.trap_states  = self._find_cells(CellType.TRAP)
        self.wall_states  = self._find_cells(CellType.WALL)

        if not self.start_states:
            raise ValueError("Grid must contain at least one START cell (value 4).")
        if not self.goal_states:
            raise ValueError("Grid must contain at least one GOAL cell (value 2).")

        # Build transition and reward models
        self._build_model()

        # Episode state
        self.agent_pos: Optional[Tuple[int, int]] = None
        self.steps_taken = 0
        self.done = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_cells(self, cell_type: CellType) -> List[Tuple[int, int]]:
        positions = np.argwhere(self.grid == int(cell_type))
        return [tuple(p) for p in positions]

    def _rc_to_state(self, row: int, col: int) -> int:
        return row * self.n_cols + col

    def _state_to_rc(self, state: int) -> Tuple[int, int]:
        return divmod(state, self.n_cols)

    def _is_valid(self, row: int, col: int) -> bool:
        return (0 <= row < self.n_rows and
                0 <= col < self.n_cols and
                self.grid[row, col] != CellType.WALL)

    def _next_pos(self, row: int, col: int, action: Action) -> Tuple[int, int]:
        dr, dc = ACTION_DELTAS[action]
        nr, nc = row + dr, col + dc
        if self._is_valid(nr, nc):
            return nr, nc
        return row, col  # bump into wall — stay

    def _build_model(self):
        """
        Pre-compute transition model P[s][a] = list of (prob, next_state, reward, done)
        Compatible with dynamic programming agents.
        """
        self.P: Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]] = {}

        for row in range(self.n_rows):
            for col in range(self.n_cols):
                s = self._rc_to_state(row, col)
                self.P[s] = {}
                cell = self.grid[row, col]

                # Terminal states — self-loop with 0 reward
                if cell in (CellType.GOAL, CellType.TRAP):
                    for a in Action:
                        self.P[s][int(a)] = [(1.0, s, 0.0, True)]
                    continue

                # Wall states — unreachable, but keep consistent
                if cell == CellType.WALL:
                    for a in Action:
                        self.P[s][int(a)] = [(1.0, s, 0.0, False)]
                    continue

                for a in Action:
                    transitions = []
                    if self.slip_prob > 0:
                        perp = self._perpendicular(a)
                        prob_main = 1.0 - self.slip_prob
                        prob_side = self.slip_prob / 2.0
                        action_probs = [(prob_main, a)] + [(prob_side, p) for p in perp]
                    else:
                        action_probs = [(1.0, a)]

                    for prob, actual_action in action_probs:
                        nr, nc = self._next_pos(row, col, actual_action)
                        ns = self._rc_to_state(nr, nc)
                        next_cell = self.grid[nr, nc]

                        bumped = (nr == row and nc == col and
                                  actual_action != a or
                                  (nr, nc) == (row, col) and
                                  ACTION_DELTAS[actual_action] != (0, 0))
                        stayed = (nr == row and nc == col)

                        if next_cell == CellType.GOAL:
                            r, done = self.goal_reward, True
                        elif next_cell == CellType.TRAP:
                            r, done = self.trap_reward, True
                        elif stayed:
                            r, done = self.wall_reward, False
                        else:
                            r, done = self.step_reward, False

                        transitions.append((prob, ns, r, done))

                    # Merge duplicate next-states
                    merged: Dict[int, Tuple[float, float, bool]] = {}
                    for prob, ns, r, done in transitions:
                        if ns in merged:
                            old_p, old_r, old_d = merged[ns]
                            merged[ns] = (old_p + prob, old_r + prob * r, done)
                        else:
                            merged[ns] = (prob, prob * r, done)

                    self.P[s][int(a)] = [
                        (p, ns, r / p if p > 0 else 0.0, d)
                        for ns, (p, r, d) in merged.items()
                    ]

    @staticmethod
    def _perpendicular(action: Action) -> List[Action]:
        perp_map = {
            Action.UP:    [Action.LEFT, Action.RIGHT],
            Action.DOWN:  [Action.LEFT, Action.RIGHT],
            Action.LEFT:  [Action.UP, Action.DOWN],
            Action.RIGHT: [Action.UP, Action.DOWN],
        }
        return perp_map[action]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def from_preset(cls, name: str, **kwargs) -> "GridWorld":
        """Create a GridWorld from a named preset."""
        if name not in cls.PRESETS:
            raise ValueError(f"Unknown preset '{name}'. Available: {list(cls.PRESETS.keys())}")
        return cls(grid=cls.PRESETS[name], **kwargs)

    def reset(self, seed: Optional[int] = None) -> int:
        """Reset environment and return initial state."""
        if seed is not None:
            np.random.seed(seed)
        self.agent_pos = self.start_states[np.random.randint(len(self.start_states))]
        self.steps_taken = 0
        self.done = False
        return self._rc_to_state(*self.agent_pos)

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Returns
        -------
        next_state, reward, done, info
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() first.")

        action = Action(action)
        row, col = self.agent_pos

        # Stochastic slip
        if self.slip_prob > 0 and np.random.random() < self.slip_prob:
            perp = self._perpendicular(action)
            action = np.random.choice(perp)

        nr, nc = self._next_pos(row, col, action)
        next_cell = self.grid[nr, nc]
        self.agent_pos = (nr, nc)
        self.steps_taken += 1

        stayed = (nr == row and nc == col)
        if next_cell == CellType.GOAL:
            reward, self.done = self.goal_reward, True
        elif next_cell == CellType.TRAP:
            reward, self.done = self.trap_reward, True
        elif stayed:
            reward = self.wall_reward
        else:
            reward = self.step_reward

        if self.steps_taken >= self.max_steps:
            self.done = True

        next_state = self._rc_to_state(*self.agent_pos)
        info = {
            "steps": self.steps_taken,
            "position": self.agent_pos,
            "cell_type": int(next_cell),
        }
        return next_state, reward, self.done, info

    def render(self, policy: Optional[np.ndarray] = None, values: Optional[np.ndarray] = None) -> str:
        """Return a string representation of the grid."""
        cell_chars = {
            CellType.EMPTY: ".",
            CellType.WALL:  "█",
            CellType.GOAL:  "G",
            CellType.TRAP:  "T",
            CellType.START: "S",
        }
        lines = []
        border = "+" + "---+" * self.n_cols
        lines.append(border)
        for r in range(self.n_rows):
            row_str = "|"
            for c in range(self.n_cols):
                s = self._rc_to_state(r, c)
                cell = self.grid[r, c]
                if self.agent_pos == (r, c):
                    ch = "A"
                elif policy is not None and cell not in (CellType.WALL, CellType.GOAL, CellType.TRAP):
                    ch = ACTION_NAMES[Action(policy[s])]
                else:
                    ch = cell_chars[CellType(cell)]
                row_str += f" {ch} |"
            lines.append(row_str)
            lines.append(border)
        return "\n".join(lines)

    @property
    def observation_space_n(self) -> int:
        return self.n_states

    @property
    def action_space_n(self) -> int:
        return self.n_actions
