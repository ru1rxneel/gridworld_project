from fastapi import FastAPI

app = FastAPI()

# Dummy environment state
state = {"position": 0}

@app.post("/reset")
def reset():
    global state
    state = {"position": 0}
    return {"state": state}

@app.post("/step")
def step(action: dict):
    global state
    
    move = action.get("action", 0)
    state["position"] += move
    
    return {
        "state": state,
        "reward": 1,
        "done": False
    }

@app.get("/state")
def get_state():
    return state
