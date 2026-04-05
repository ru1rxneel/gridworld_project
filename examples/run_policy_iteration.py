"""
Example: Solve Grid World with Policy Iteration.
Run: python examples/run_policy_iteration.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt

from gridworld import GridWorld
from gridworld.agents import PolicyIterationAgent, ValueIterationAgent
from gridworld.utils import plot_value_function, plot_policy


def compare_policies(env, pi_agent: PolicyIterationAgent, vi_agent: ValueIterationAgent):
    """Check if two agents agree on the optimal policy."""
    pi_policy = pi_agent.get_policy()
    vi_policy = vi_agent.get_policy()
    from gridworld.env import CellType
    non_terminal = [
        s for s in range(env.n_states)
        if env.grid.flat[s] not in (CellType.WALL, CellType.GOAL, CellType.TRAP)
    ]
    matches = sum(pi_policy[s] == vi_policy[s] for s in non_terminal)
    pct = 100 * matches / len(non_terminal) if non_terminal else 100.0
    print(f"  Policy agreement with VI: {matches}/{len(non_terminal)} states ({pct:.1f}%)")


def main():
    print("=" * 55)
    print("  Grid World — Policy Iteration Demo")
    print("=" * 55)

    env = GridWorld.from_preset("trap_maze", slip_prob=0.1)
    print(f"\nGrid: {env.n_rows}×{env.n_cols}, slip_prob={env.slip_prob}")
    print(env.render())

    # --- Policy Iteration ---
    pi_agent = PolicyIterationAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        gamma=0.99,
        eval_theta=1e-8,
    )
    print("\nRunning Policy Iteration...")
    pi_hist = pi_agent.train(env)
    print(f"  Converged         : {pi_hist['converged']}")
    print(f"  Policy updates    : {pi_hist['n_policy_updates']}")
    print(f"  Avg eval sweeps   : {sum(pi_hist['eval_iters_per_update']) / max(1, pi_hist['n_policy_updates']):.1f}")

    # --- Value Iteration for comparison ---
    vi_agent = ValueIterationAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        gamma=0.99,
        theta=1e-9,
    )
    print("\nRunning Value Iteration for comparison...")
    vi_agent.train(env)

    compare_policies(env, pi_agent, vi_agent)

    print("\nOptimal policy (Policy Iteration):")
    print(env.render(policy=pi_agent.get_policy()))

    # --- Plots ---
    os.makedirs("results", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Policy Iteration Results", fontsize=14, fontweight="bold")

    plot_value_function(env, pi_agent.get_value_function(),
                        title="V*(s) — Policy Iteration", ax=axes[0])
    plot_policy(env, pi_agent.get_policy(),
                values=pi_agent.get_value_function(),
                title="π*(s) — Policy Iteration", ax=axes[1])

    plt.tight_layout()
    plt.savefig("results/pi_results.png", dpi=150, bbox_inches="tight")
    print("\nPlot saved to results/pi_results.png")
    plt.show()


if __name__ == "__main__":
    main()
