import random

WIDTH = 256
HEIGHT = 256
SEED = 42   # ensures the maze is always identical

random.seed(SEED)

def generate_maze(width, height):
    maze = [["#" for _ in range(width)] for _ in range(height)]

    stack = [(1,1)]
    maze[1][1] = " "

    directions = [(2,0),(-2,0),(0,2),(0,-2)]

    while stack:
        x,y = stack[-1]

        neighbors = []
        for dx,dy in directions:
            nx,ny = x+dx, y+dy
            if 0 < nx < width-1 and 0 < ny < height-1 and maze[ny][nx] == "#":
                neighbors.append((nx,ny,dx,dy))

        if neighbors:
            nx,ny,dx,dy = random.choice(neighbors)

            maze[y + dy//2][x + dx//2] = " "
            maze[ny][nx] = " "

            stack.append((nx,ny))
        else:
            stack.pop()

    return maze


maze = generate_maze(WIDTH, HEIGHT)

with open("maze256.txt","w") as f:
    for row in maze:
        f.write("".join(row) + "\n")

print("maze256.txt generated")
