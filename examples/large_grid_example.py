import numpy as np

class GridWorld3D:
    def __init__(self, size=(256,256,256)):
        self.size = size
        self.grid = np.zeros(size, dtype=np.int8)

        # Define start and goal
        self.start = (0,0,0)
        self.goal = (255,255,255)

        self.agent_pos = list(self.start)

    def reset(self):
        self.agent_pos = list(self.start)
        return tuple(self.agent_pos)

    def step(self, action):
        x,y,z = self.agent_pos

        moves = {
            "up": (0,1,0),
            "down": (0,-1,0),
            "left": (-1,0,0),
            "right": (1,0,0),
            "forward": (0,0,1),
            "backward": (0,0,-1)
        }

        dx,dy,dz = moves[action]

        nx = max(0, min(self.size[0]-1, x+dx))
        ny = max(0, min(self.size[1]-1, y+dy))
        nz = max(0, min(self.size[2]-1, z+dz))

        self.agent_pos = [nx,ny,nz]

        done = tuple(self.agent_pos) == self.goal
        reward = 1 if done else -0.01

        return tuple(self.agent_pos), reward, done


if __name__ == "__main__":
    env = GridWorld3D((256,256,256))

    state = env.reset()

    for i in range(1000):
        action = np.random.choice(
            ["up","down","left","right","forward","backward"]
        )

        state, reward, done = env.step(action)

        if done:
            print("Reached goal!")
            break
