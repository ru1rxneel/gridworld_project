"""
Visualization utilities for Grid World experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, List
import os

from ..env import GridWorld, CellType, Action, ACTION_DELTAS


# Custom colormap for value functions
_VALUE_CMAP = LinearSegmentedColormap.from_list(
    "gridworld_values",
    ["#d73027", "#fee090", "#e0f3f8", "#4575b4"],
    N=256,
)


def plot_value_function(
    env: GridWorld,
    values: np.ndarray,
    title: str = "Value Function V(s)",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Heat-map of state values overlaid on the grid.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(5, env.n_cols * 1.2), max(4, env.n_rows * 1.2)))

    V_grid = values.reshape(env.n_rows, env.n_cols).astype(float)

    # Mask walls
    mask = env.grid == CellType.WALL
    V_masked = np.ma.array(V_grid, mask=mask)

    im = ax.imshow(V_masked, cmap=_VALUE_CMAP, aspect="equal",
                   vmin=V_grid[~mask].min() if (~mask).any() else -1,
                   vmax=V_grid[~mask].max() if (~mask).any() else 1)

    if fig:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotate cells
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            cell = env.grid[r, c]
            s = r * env.n_cols + c
            if cell == CellType.WALL:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           color="#2d2d2d", zorder=2))
                ax.text(c, r, "█", ha="center", va="center",
                        color="white", fontsize=10, zorder=3)
            elif cell == CellType.GOAL:
                ax.text(c, r, f"G\n{values[s]:.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="#155724", zorder=3)
            elif cell == CellType.TRAP:
                ax.text(c, r, f"T\n{values[s]:.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="#721c24", zorder=3)
            else:
                ax.text(c, r, f"{values[s]:.2f}", ha="center", va="center",
                        fontsize=9, color="black", zorder=3)

    _style_grid_axes(ax, env, title)
    if save_path:
        _save(fig or plt.gcf(), save_path)
    return fig or plt.gcf()


def plot_policy(
    env: GridWorld,
    policy: np.ndarray,
    values: Optional[np.ndarray] = None,
    title: str = "Optimal Policy",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Arrow diagram of the agent's policy, optionally coloured by value.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(max(5, env.n_cols * 1.2), max(4, env.n_rows * 1.2)))

    if values is not None:
        V_grid = values.reshape(env.n_rows, env.n_cols).astype(float)
        mask = env.grid == CellType.WALL
        V_masked = np.ma.array(V_grid, mask=mask)
        ax.imshow(V_masked, cmap=_VALUE_CMAP, aspect="equal", alpha=0.6)
    else:
        ax.imshow(np.zeros((env.n_rows, env.n_cols)), cmap="Greys",
                  aspect="equal", alpha=0.1)

    for r in range(env.n_rows):
        for c in range(env.n_cols):
            cell = env.grid[r, c]
            s = r * env.n_cols + c
            if cell == CellType.WALL:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                           color="#2d2d2d", zorder=2))
            elif cell == CellType.GOAL:
                ax.text(c, r, "G", ha="center", va="center",
                        fontsize=14, fontweight="bold", color="#155724", zorder=3)
            elif cell == CellType.TRAP:
                ax.text(c, r, "T", ha="center", va="center",
                        fontsize=14, fontweight="bold", color="#721c24", zorder=3)
            else:
                a = Action(policy[s])
                dr, dc = ACTION_DELTAS[a]
                ax.annotate(
                    "", xy=(c + dc * 0.35, r + dr * 0.35),
                    xytext=(c - dc * 0.2, r - dr * 0.2),
                    arrowprops=dict(arrowstyle="->", color="#1a1a2e",
                                   lw=2.0, mutation_scale=15),
                    zorder=3,
                )

    _style_grid_axes(ax, env, title)
    if save_path:
        _save(fig or plt.gcf(), save_path)
    return fig or plt.gcf()


def plot_training_curves(
    history: dict,
    window: int = 50,
    title: str = "Q-Learning Training Curves",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot episode rewards and epsilon over training."""
    rewards = np.array(history["episode_rewards"])
    epsilons = np.array(history["epsilons"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # --- Rewards ---
    ax = axes[0]
    ax.plot(rewards, alpha=0.3, color="#4575b4", linewidth=0.8, label="Episode reward")
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(rewards)), smoothed,
                color="#d73027", linewidth=2, label=f"{window}-ep moving avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Episode Rewards")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Epsilon ---
    ax = axes[1]
    ax.plot(epsilons, color="#74add1", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("ε (exploration)")
    ax.set_title("Exploration Rate (ε)")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


def plot_convergence(
    history: dict,
    title: str = "Value Iteration Convergence",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot Bellman error δ over iterations for value / policy iteration."""
    deltas = history.get("deltas", [])
    if not deltas:
        raise ValueError("History must contain 'deltas' list.")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(deltas, color="#d73027", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Max |ΔV| (log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)

    n_iter = history.get("n_iterations", len(deltas))
    converged = history.get("converged", False)
    status = f"Converged in {n_iter} iterations" if converged else f"Did not converge after {n_iter} iterations"
    ax.set_title(f"{title}\n{status}", fontsize=12)

    plt.tight_layout()
    if save_path:
        _save(fig, save_path)
    return fig


def plot_episode(
    env: GridWorld,
    trajectory: List[tuple],
    title: str = "Episode Trajectory",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a single episode's path through the grid."""
    fig, ax = plt.subplots(figsize=(max(5, env.n_cols * 1.2), max(4, env.n_rows * 1.2)))

    # Background grid
    background = np.zeros((env.n_rows, env.n_cols))
    for r in range(env.n_rows):
        for c in range(env.n_cols):
            cell = env.grid[r, c]
            if cell == CellType.WALL:   background[r, c] = -2
            elif cell == CellType.GOAL: background[r, c] = 1
            elif cell == CellType.TRAP: background[r, c] = -1

    cell_cmap = LinearSegmentedColormap.from_list(
        "cells", ["#721c24", "#f8f9fa", "#f8f9fa", "#155724"], N=256
    )
    ax.imshow(background, cmap=cell_cmap, aspect="equal",
              vmin=-2, vmax=2, alpha=0.5)

    # Draw trajectory
    if trajectory:
        rows = [t[0] for t in trajectory]
        cols = [t[1] for t in trajectory]
        ax.plot(cols, rows, "o-", color="#4575b4", linewidth=2,
                markersize=6, alpha=0.8, zorder=3)
        ax.plot(cols[0], rows[0], "s", color="green", markersize=12,
                zorder=4, label="Start")
        ax.plot(cols[-1], rows[-1], "*", color="gold", markersize=14,
                zorder=4, label="End")

    ax.legend(loc="upper right")
    _style_grid_axes(ax, env, title)
    if save_path:
        _save(fig, save_path)
    return fig


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _style_grid_axes(ax: plt.Axes, env: GridWorld, title: str):
    ax.set_xticks(np.arange(-0.5, env.n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, env.n_rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linewidth=0.5, alpha=0.5)
    ax.set_xticks(range(env.n_cols))
    ax.set_yticks(range(env.n_rows))
    ax.set_xticklabels(range(env.n_cols))
    ax.set_yticklabels(range(env.n_rows))
    ax.set_title(title, fontsize=12, fontweight="bold")


def _save(fig: plt.Figure, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {path}")
