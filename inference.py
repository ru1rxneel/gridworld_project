from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

# IMPORTANT FIX
app = FastAPI(root_path="")

GRID_SIZE = 5
agent = [0, 0]
goal = [4, 4]

# Request model
class Action(BaseModel):
    action: int

def get_obs():
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    grid[goal[0], goal[1]] = 2
    grid[agent[0], agent[1]] = 1
    return grid.tolist()

# Root (helps HF routing)
@app.get("/")
def root():
    return {"message": "API is running"}

# Health check
@app.get("/health")
def health():
    return {"status": "ok"}

# Reset endpoint
@app.post("/reset")
def reset():
    global agent
    agent = [0, 0]

    return {
        "observation": get_obs(),
        "reward": 0,
        "done": False,
        "info": {}
    }

# Step endpoint (FIXED JSON input)
@app.post("/step")
def step(action: Action):
    global agent

    move = action.action

    if move == 0:
        agent[0] = max(0, agent[0] - 1)
    elif move == 1:
        agent[0] = min(GRID_SIZE - 1, agent[0] + 1)
    elif move == 2:
        agent[1] = max(0, agent[1] - 1)
    elif move == 3:
        agent[1] = min(GRID_SIZE - 1, agent[1] + 1)

    done = agent == goal
    reward = 1 if done else -0.01

    return {
        "observation": get_obs(),
        "reward": reward,
        "done": done,
        "info": {}
    }

# Optional but useful
@app.get("/state")
def state():
    return {
        "agent": agent,
        "goal": goal,
        "observation": get_obs()
    }
