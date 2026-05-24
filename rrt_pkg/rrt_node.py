#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from ompl import base as ob
from ompl import geometric as og


from nav_msgs.msg import OccupancyGrid, Path
# i'm going to assume occupancy grid comes as the conventional nav_msgs type

from geometry_msgs.msg import PoseStamped
# assumption for init/goal poses


class Grid:
    def __init__(self, grid, resolution, origin):
        # grid is a 2d array s.t. 0=free, 1=occ
        # resolution: m/cell (make this a float)
        # origin is (x, y) of grid[0, 0]

        self.grid = grid
        self.resolution = resolution
        self.origin = origin

    def is_free(self, x, y):
        # true if (x,y) maps ot free cell
        col = int((x - self.origin[0]) / self.resolution)
        row = int((y - self.origin[1]) / self.resolution)
        if row < 0 or row >= self.grid.shape[0]:
            return False
        if col < 0 or col >= self.grid.shape[1]:
            return False
        return self.grid[row, col] == 0


class RRTPlanner:
    def __init__(self, grid, turning_radius, bounds):
        # grid: occ grid for collision checking
        # turning radius: min for reeds shepp

        # bounds (x_min, x_max, y_min, y_max) in world frame so sampling stays finite

        self.grid = grid
        self.turning_radius = turning_radius
        self.bounds = bounds
        self.space = self.make_state_space()

    def make_state_space(self):
        space = ob.ReedsSheppStateSpace(self.turning_radius)

        b = ob.RealVectorBounds(2)
        b.setLow(0, self.bounds[0])
        b.setHigh(0, self.bounds[1])
        b.setLow(1, self.bounds[2])
        b.setHigh(1, self.bounds[3])
        space.setBounds(b)
        return space


class RRTNode(Node):
    def __init__(self):
        super().__init__("rrt_node")

        self.occ_grid = None
        self.pose_current = None
        self.pose_goal = None

        self.map_sub = self.create_subscription(
            OccupancyGrid, "map", self.map_callback, 10
        )

        self.pose_sub = self.create_subscription(
            PoseStamped, "pose_current", self.pose_callback, 10
        )

        self.goal_sub = self.create_subscription(
            PoseStamped, "goal_pose", self.goal_callback, 10
        )

        self.path_pub = self.create_publisher(Path, "rrt_path", 10)

    def map_callback(self, msg: OccupancyGrid):
        grid = np.reshape(msg.data, (msg.info.height, msg.info.width))
        origin = [msg.info.origin.position.x, msg.info.origin.position.y]
        bounds = [
            origin[0],
            origin[0] + msg.info.width * msg.info.resolution,
            origin[1],
            origin[1] + msg.info.height * msg.info.resolution,
        ]
        self.occ_grid = Grid(grid, msg.info.resolution, origin)
        self.bounds = bounds

    def pose_callback(self, msg: PoseStamped):
        self.pose_current = msg

    def goal_callback(self, msg: PoseStamped):
        self.goal_pose = msg


def main(args=None):
    rclpy.init(args=args)
    node = RRTNode()
    rclpy.spin(node)
    rclpy.shutdown()
