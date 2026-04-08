import gradio as gr
import numpy as np

# simple gridworld demo
GRID_SIZE = 5
agent = [0, 0]
goal = [4, 4]

def reset():
    global agent
    agent = [0, 0]
    return render_grid()

def move(direction):
    global agent

    if direction == "Up":
        agent[0] = max(0, agent[0]-1)
    elif direction == "Down":
        agent[0] = min(GRID_SIZE-1, agent[0]+1)
    elif direction == "Left":
        agent[1] = max(0, agent[1]-1)
    elif direction == "Right":
        agent[1] = min(GRID_SIZE-1, agent[1]+1)

    return render_grid()

def render_grid():
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    grid[goal[0], goal[1]] = 0.5
    grid[agent[0], agent[1]] = 1
    return grid

with gr.Blocks() as demo:
    gr.Markdown("# Gridworld Demo")

    grid_display = gr.Image(value=render_grid(), type="numpy")

    with gr.Row():
        up = gr.Button("Up")
    with gr.Row():
        left = gr.Button("Left")
        down = gr.Button("Down")
        right = gr.Button("Right")

    reset_btn = gr.Button("Reset")

    up.click(lambda: move("Up"), outputs=grid_display)
    down.click(lambda: move("Down"), outputs=grid_display)
    left.click(lambda: move("Left"), outputs=grid_display)
    right.click(lambda: move("Right"), outputs=grid_display)
    reset_btn.click(reset, outputs=grid_display)

demo.launch()
