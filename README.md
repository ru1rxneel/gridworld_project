# 🧭 Grid World — Reinforcement Learning Playground

A clean, well-tested implementation of a **Grid World environment** with three classic RL algorithms:

| Algorithm | Type | Model-Free? |
|-----------|------|-------------|
| **Q-Learning** | TD Control | ✅ Yes |
| **Value Iteration** | Dynamic Programming | ❌ Requires model |
| **Policy Iteration** | Dynamic Programming | ❌ Requires model |

---
                                        
## 📁 Project Structure

```
gridworld_project/
├── gridworld/
│   ├── __init__.py
│   ├── env.py                  # Grid World environment
│   ├── agents/
│   │   ├── base.py             # Abstract base agent
│   │   ├── q_learning.py       # Q-Learning (ε-greedy)
│   │   ├── value_iteration.py  # Value Iteration (Bellman sweeps)
│   │   └── policy_iteration.py # Policy Iteration (eval + improve)
│   └── utils/
│       └── visualization.py    # Matplotlib plots
├── examples/
│   ├── train_qlearning.py
│   ├── run_value_iteration.py
│   └── run_policy_iteration.py
├── tests/
│   ├── test_env.py
│   └── test_agents.py
├── notebooks/
│   └── GridWorld_Demo.ipynb
├── requirements.txt
├── setup.py
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/gridworld_project.git
cd gridworld-rl

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. Run Examples

```bash
# Q-Learning (model-free)
python examples/train_qlearning.py

# Value Iteration (DP)
python examples/run_value_iteration.py

# Policy Iteration (DP)
python examples/run_policy_iteration.py
```

### 3. Run Tests

```bash
pytest tests/ -v
# With coverage:
pytest tests/ -v --cov=gridworld --cov-report=term-missing
```

---

## 🗺 Environment

### Grid Cells

| Symbol | Value | Meaning |
|--------|-------|---------|
| `S` | 4 | Start position |
| `.` | 0 | Empty (walkable) |
| `█` | 1 | Wall (impassable) |
| `G` | 2 | Goal (+1.0 reward, terminal) |
| `T` | 3 | Trap (−1.0 reward, terminal) |

### Built-in Presets

```python
from gridworld import GridWorld

env = GridWorld.from_preset("simple_5x5")   # Basic navigation
env = GridWorld.from_preset("trap_maze")    # Obstacles and traps
env = GridWorld.from_preset("cliff_walk")   # Classic cliff-walking
```

### Custom Grid

```python
grid = [
    [4, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 2],
]
env = GridWorld(
    grid=grid,
    slip_prob=0.1,       # 10% chance of slipping sideways
    step_reward=-0.04,   # Small penalty per step
    goal_reward=1.0,
    trap_reward=-1.0,
    max_steps=200,
)
```

### Standard Gym-like API

```python
state = env.reset(seed=42)

while not done:
    action = agent.select_action(state)          # 0=Up 1=Down 2=Left 3=Right
    next_state, reward, done, info = env.step(action)
    state = next_state

print(env.render(policy=agent.get_policy()))    # ASCII visualization
```

---

## 🤖 Agents

### Q-Learning

```python
from gridworld.agents import QLearningAgent

agent = QLearningAgent(
    n_states=env.n_states,
    n_actions=env.n_actions,
    alpha=0.1,           # Learning rate
    gamma=0.99,          # Discount factor
    epsilon=1.0,         # Initial exploration
    epsilon_decay=0.995, # Per-episode decay
    epsilon_min=0.01,
)

history = agent.train(env, n_episodes=2000, seed=42)

policy = agent.get_policy()         # np.ndarray of actions
V      = agent.get_value_function() # np.ndarray V(s) = max_a Q(s,a)
Q      = agent.get_q_table()        # np.ndarray shape (n_states, n_actions)
```

### Value Iteration

```python
from gridworld.agents import ValueIterationAgent

agent = ValueIterationAgent(
    n_states=env.n_states,
    n_actions=env.n_actions,
    gamma=0.99,
    theta=1e-9,   # Convergence threshold
)

history = agent.train(env)
print(f"Converged in {history['n_iterations']} iterations")
```

### Policy Iteration

```python
from gridworld.agents import PolicyIterationAgent

agent = PolicyIterationAgent(
    n_states=env.n_states,
    n_actions=env.n_actions,
    gamma=0.99,
    eval_theta=1e-8,
)

history = agent.train(env)
print(f"Policy updates: {history['n_policy_updates']}")
```

---

## 📊 Visualization

```python
from gridworld.utils import (
    plot_value_function,
    plot_policy,
    plot_training_curves,
    plot_convergence,
)

# Value heatmap
plot_value_function(env, agent.get_value_function())

# Policy arrows
plot_policy(env, agent.get_policy(), values=agent.get_value_function())

# Q-Learning curves
plot_training_curves(history)

# DP convergence (delta per iteration)
plot_convergence(history)
```

---

## 🧪 Testing

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_agents.py -v -k "qlearning"

# Coverage report
pytest tests/ --cov=gridworld --cov-report=html
open htmlcov/index.html
```

---

## 📘 Key Concepts

### Markov Decision Process (MDP)

The environment defines an MDP `(S, A, P, R, γ)`:
- **S** — finite set of states (grid cells)
- **A** — {Up, Down, Left, Right}
- **P(s'|s,a)** — stochastic transition (controlled by `slip_prob`)
- **R(s,a,s')** — step / goal / trap / wall rewards
- **γ** — discount factor (controls far-sightedness)

### Bellman Equations

**Value Iteration** iterates:
```
V(s) ← max_a Σ P(s'|s,a) [R(s,a,s') + γ V(s')]
```

**Policy Iteration** alternates:
1. *Evaluation*: solve `V^π(s) = Σ P(s'|s,π(s)) [R + γ V^π(s')]`
2. *Improvement*: `π(s) ← argmax_a Σ P(s'|s,a) [R + γ V^π(s')]`

**Q-Learning** updates online:
```
Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
