from fastapi import FastAPI
import numpy as np

app = FastAPI()

GRID_SIZE = 5
agent = [0, 0]
goal = [4, 4]

def get_obs():
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    grid[goal[0], goal[1]] = 2
    grid[agent[0], agent[1]] = 1
    return grid.tolist()

@app.get("/health")
def health():
    return {"status": "ok"}

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

@app.post("/step")
def step(action: int):
    global agent

    if action == 0:
        agent[0] = max(0, agent[0]-1)
    elif action == 1:
        agent[0] = min(GRID_SIZE-1, agent[0]+1)
    elif action == 2:
        agent[1] = max(0, agent[1]-1)
    elif action == 3:
        agent[1] = min(GRID_SIZE-1, agent[1]+1)

    done = agent == goal
    reward = 1 if done else -0.01

    return {
        "observation": get_obs(),
        "reward": reward,
        "done": done,
        "info": {}
    }
