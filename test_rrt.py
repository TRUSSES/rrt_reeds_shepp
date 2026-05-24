# test_rrt.py
# Benjamin Aziel

# test script for rrt connect with reeds shepp state space
# generates 5 plans w/ different seeds on fixed map
# run with `uv run python test_rrt.py`

import numpy as np
import matplotlib.pyplot as plt
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou

GRID_SIZE = 100
RESOLUTION = 0.1
TURNING_RADIUS = 0.5
OBSTACLES = [
    [2.0, 2.0, 0.5],
    [5.0, 5.0, 1.0],
    [7.0, 3.0, 0.7],
    [3.0, 7.0, 0.8],
    [8.0, 7.0, 0.5],
]


def make_grid():
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            x = c * RESOLUTION
            y = r * RESOLUTION
            for ox, oy, radius in OBSTACLES:
                if np.sqrt((x - ox) ** 2 + (y - oy) ** 2) < radius:
                    grid[r, c] = 1
                    break
    return grid


def is_free(grid, x, y):
    col = int(x / RESOLUTION)
    row = int(y / RESOLUTION)
    if row < 0 or row >= grid.shape[0]:
        return False
    if col < 0 or col >= grid.shape[1]:
        return False
    return grid[row, col] == 0


class ValidityChecker(ob.StateValidityChecker):
    def __init__(self, si, grid):
        super().__init__(si)
        self.grid = grid

    def isValid(self, state):
        return bool(is_free(self.grid, state.getX(), state.getY()))


def plan(grid):
    space = ob.ReedsSheppStateSpace(TURNING_RADIUS)
    b = ob.RealVectorBounds(2)
    b.setLow(0, 0.0)
    b.setHigh(0, 10.0)
    b.setLow(1, 0.0)
    b.setHigh(1, 10.0)
    space.setBounds(b)

    si = ob.SpaceInformation(space)
    si.setStateValidityChecker(ValidityChecker(si, grid))
    si.setup()

    start = space.allocState()
    start.setX(1.0)
    start.setY(1.0)
    start.setYaw(0.0)

    goal = space.allocState()
    goal.setX(9.0)
    goal.setY(9.0)
    goal.setYaw(0.0)

    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start, goal)

    planner = og.RRTConnect(si)
    planner.setProblemDefinition(pdef)
    planner.setup()

    solved = planner.solve(5.0)
    if not solved:
        return None

    pdef.getSolutionPath().interpolate()
    states = pdef.getSolutionPath().getStates()
    return [(s.getX(), s.getY()) for s in states]


def plot(grid, paths):
    fig, axes = plt.subplots(1, 5)
    for i, (ax, path) in enumerate(zip(axes, paths)):
        ax.imshow(grid, origin="lower", extent=[0, 10, 0, 10], cmap="gray_r")
        if path:
            xs, ys = zip(*path)
            ax.plot(xs, ys, "b-")
            ax.plot(xs[0], ys[0], "go")
            ax.plot(xs[-1], ys[-1], "ro")
    plt.show()


if __name__ == "__main__":
    grid = make_grid()
    paths = []
    for seed in range(5):
        ou.RNG.setSeed(seed)
        paths.append(plan(grid))
    plot(grid, paths)
