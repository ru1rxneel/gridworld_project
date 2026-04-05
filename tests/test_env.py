"""Tests for the GridWorld environment."""

import pytest
import numpy as np
from gridworld.env import GridWorld, Action, CellType


@pytest.fixture
def simple_env():
    return GridWorld.from_preset("simple_5x5")


@pytest.fixture
def trap_env():
    return GridWorld.from_preset("trap_maze")


class TestGridWorldInit:
    def test_default_grid_shape(self, simple_env):
        assert simple_env.n_rows == 5
        assert simple_env.n_cols == 5

    def test_state_count(self, simple_env):
        assert simple_env.n_states == 25

    def test_action_count(self, simple_env):
        assert simple_env.n_actions == 4

    def test_start_state_exists(self, simple_env):
        assert len(simple_env.start_states) >= 1

    def test_goal_state_exists(self, simple_env):
        assert len(simple_env.goal_states) >= 1

    def test_invalid_grid_no_start(self):
        bad_grid = [[0, 0], [0, 2]]  # no START cell
        with pytest.raises(ValueError, match="START"):
            GridWorld(grid=bad_grid)

    def test_invalid_grid_no_goal(self):
        bad_grid = [[4, 0], [0, 0]]  # no GOAL cell
        with pytest.raises(ValueError, match="GOAL"):
            GridWorld(grid=bad_grid)

    def test_custom_rewards(self):
        env = GridWorld.from_preset("simple_5x5", goal_reward=10.0, trap_reward=-5.0)
        assert env.goal_reward == 10.0
        assert env.trap_reward == -5.0


class TestGridWorldReset:
    def test_reset_returns_valid_state(self, simple_env):
        state = simple_env.reset()
        assert 0 <= state < simple_env.n_states

    def test_reset_seed_deterministic(self, simple_env):
        s1 = simple_env.reset(seed=0)
        s2 = simple_env.reset(seed=0)
        assert s1 == s2

    def test_reset_clears_done(self, simple_env):
        simple_env.reset()
        simple_env.done = True
        simple_env.reset()
        assert not simple_env.done


class TestGridWorldStep:
    def test_step_returns_tuple(self, simple_env):
        simple_env.reset(seed=0)
        result = simple_env.step(Action.RIGHT)
        assert len(result) == 4  # next_state, reward, done, info

    def test_step_on_done_raises(self, simple_env):
        simple_env.reset()
        simple_env.done = True
        with pytest.raises(RuntimeError):
            simple_env.step(Action.UP)

    def test_wall_bump_stays_in_place(self):
        """Agent bumping into a wall should stay in the same position."""
        env = GridWorld(grid=[
            [4, 1],
            [0, 2],
        ])
        state = env.reset(seed=0)
        # Agent starts at (0,0). Bumping UP or LEFT should keep them at (0,0).
        env.agent_pos = (0, 0)
        old_pos = env.agent_pos
        _, _, _, info = env.step(Action.UP)  # wall above (out of bounds)
        assert info["position"] == old_pos

    def test_goal_terminates_episode(self, simple_env):
        """Reaching the goal should set done=True."""
        # Force agent next to goal at (4,4) → move right to reach it
        simple_env.reset()
        simple_env.agent_pos = (4, 3)
        _, reward, done, _ = simple_env.step(Action.RIGHT)
        assert done
        assert reward == simple_env.goal_reward

    def test_max_steps_terminates(self, simple_env):
        env = GridWorld.from_preset("simple_5x5", max_steps=5)
        env.reset(seed=42)
        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(Action.UP)
            steps += 1
            if steps > 20:
                break
        assert steps <= 5 or done


class TestTransitionModel:
    def test_probabilities_sum_to_one(self, simple_env):
        for s in range(simple_env.n_states):
            for a in range(simple_env.n_actions):
                total_prob = sum(t[0] for t in simple_env.P[s][a])
                assert abs(total_prob - 1.0) < 1e-9, \
                    f"P[{s}][{a}] probabilities sum to {total_prob}"

    def test_deterministic_transitions(self):
        env = GridWorld.from_preset("simple_5x5", slip_prob=0.0)
        for s in range(env.n_states):
            for a in range(env.n_actions):
                transitions = env.P[s][a]
                # Deterministic: each (s, a) should have exactly 1 transition
                assert len(transitions) == 1

    def test_stochastic_transitions(self):
        env = GridWorld.from_preset("simple_5x5", slip_prob=0.2)
        # In a stochastic env, most states/actions have >1 transition
        multi_count = sum(
            1 for s in range(env.n_states)
            for a in range(env.n_actions)
            if len(env.P[s][a]) > 1
        )
        assert multi_count > 0


class TestPresets:
    @pytest.mark.parametrize("preset", ["simple_5x5", "trap_maze", "cliff_walk"])
    def test_all_presets_load(self, preset):
        env = GridWorld.from_preset(preset)
        assert env.n_states > 0

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            GridWorld.from_preset("nonexistent_preset")


class TestRender:
    def test_render_returns_string(self, simple_env):
        simple_env.reset()
        output = simple_env.render()
        assert isinstance(output, str)
        assert "A" in output  # agent marker

    def test_render_with_policy(self, simple_env):
        simple_env.reset()
        policy = np.zeros(simple_env.n_states, dtype=int)
        output = simple_env.render(policy=policy)
        assert isinstance(output, str)
