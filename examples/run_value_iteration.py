"""
Example: Solve Grid World with Value Iteration.
Run: python examples/run_value_iteration.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt

from gridworld import GridWorld
from gridworld.agents import ValueIterationAgent
from gridworld.utils import plot_value_function, plot_policy, plot_convergence


def main():
    print("=" * 55)
    print("  Grid World — Value Iteration Demo")
    print("=" * 55)

    # --- Environment ---
    env = GridWorld.from_preset("simple_5x5", slip_prob=0.2)
    print(f"\nGrid: {env.n_rows}×{env.n_cols}, slip_prob={env.slip_prob}")
    print(env.render())

    # --- Agent ---
    agent = ValueIterationAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        gamma=0.99,
        theta=1e-9,
    )

    # --- Solve ---
    print("\nRunning Value Iteration...")
    history = agent.train(env)

    print(f"  Converged   : {history['converged']}")
    print(f"  Iterations  : {history['n_iterations']}")
    print(f"  Final delta : {history['final_delta']:.2e}")

    # --- Policy display ---
    print("\nOptimal policy:")
    print(env.render(policy=agent.get_policy()))

    # --- Plots ---
    os.makedirs("results", exist_ok=True)

    fig1 = plot_convergence(history, title="Value Iteration Convergence")
    fig1.savefig("results/vi_convergence.png", dpi=150, bbox_inches="tight")

    fig2 = plot_value_function(env, agent.get_value_function(),
                               title="Value Iteration: Optimal V*(s)")
    fig2.savefig("results/vi_values.png", dpi=150, bbox_inches="tight")

    fig3 = plot_policy(env, agent.get_policy(),
                       values=agent.get_value_function(),
                       title="Value Iteration: Optimal Policy π*(s)")
    fig3.savefig("results/vi_policy.png", dpi=150, bbox_inches="tight")

    print("\nPlots saved to results/")
    plt.show()


if __name__ == "__main__":
    main()
