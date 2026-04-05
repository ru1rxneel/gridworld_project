"""
Example: Train a Q-Learning agent on the Grid World.
Run: python examples/train_qlearning.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from gridworld import GridWorld
from gridworld.agents import QLearningAgent
from gridworld.utils import plot_value_function, plot_policy, plot_training_curves


def evaluate(env: GridWorld, agent: QLearningAgent, n_eval: int = 100) -> dict:
    """Evaluate a trained agent over multiple episodes."""
    rewards, lengths, successes = [], [], []
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0  # greedy evaluation

    for _ in range(n_eval):
        state = env.reset()
        total_reward, done = 0.0, False
        while not done:
            action = agent.select_action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
        lengths.append(info["steps"])
        from gridworld.env import CellType
        successes.append(int(info["cell_type"] == int(CellType.GOAL)))

    agent.epsilon = saved_epsilon
    return {
        "mean_reward": np.mean(rewards),
        "std_reward":  np.std(rewards),
        "mean_length": np.mean(lengths),
        "success_rate": np.mean(successes),
    }


def main():
    print("=" * 55)
    print("  Grid World — Q-Learning Demo")
    print("=" * 55)

    # --- Environment ---
    env = GridWorld.from_preset("trap_maze", slip_prob=0.1, step_reward=-0.04)
    print(f"\nEnvironment: {env.n_rows}×{env.n_cols} grid")
    print(f"States: {env.n_states}  |  Actions: {env.n_actions}")
    print(f"Slip probability: {env.slip_prob}")
    print("\nInitial grid:")
    print(env.render())

    # --- Agent ---
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    )

    # --- Train ---
    print("\nTraining Q-Learning agent for 2000 episodes...")
    history = agent.train(env, n_episodes=2000, seed=42, verbose=True)

    # --- Evaluate ---
    print("\nEvaluating trained agent (100 episodes, greedy)...")
    results = evaluate(env, agent)
    print(f"  Mean reward  : {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"  Mean steps   : {results['mean_length']:.1f}")
    print(f"  Success rate : {results['success_rate'] * 100:.1f}%")

    # --- Visualize policy ---
    print("\nLearned policy:")
    policy = agent.get_policy()
    print(env.render(policy=policy))

    # --- Plots ---
    os.makedirs("results", exist_ok=True)

    fig1 = plot_training_curves(history, title="Q-Learning Training Curves")
    fig1.savefig("results/qlearning_training.png", dpi=150, bbox_inches="tight")

    fig2 = plot_value_function(env, agent.get_value_function(),
                               title="Q-Learning: V(s) = max_a Q(s,a)")
    fig2.savefig("results/qlearning_values.png", dpi=150, bbox_inches="tight")

    fig3 = plot_policy(env, policy, values=agent.get_value_function(),
                       title="Q-Learning: Learned Policy")
    fig3.savefig("results/qlearning_policy.png", dpi=150, bbox_inches="tight")

    print("\nPlots saved to results/")
    plt.show()


if __name__ == "__main__":
    main()
