#!/usr/bin/env python3

# rrt_node.py
# Benjamin Aziel

# rrt connect with reeds-shepp state space

import rclpy
from rclpy.node import Node
import numpy as np
from ompl import base as ob
from ompl import geometric as og


from nav_msgs.msg import OccupancyGrid, Path
# i'm going to assume occupancy grid comes as the conventional nav_msgs type

from geometry_msgs.msg import PoseStamped
# assumption for init/goal poses

TURNING_RADIUS = 0.5  # min turning radius for reeds shepp
PLAN_TIMEOUT = 5.0  # seconds before giving up


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


class ValidityChecker(ob.StateValidityChecker):
    def __init__(self, si, grid):
        super().__init__(si)
        self.grid = grid

    def isValid(self, state):
        return bool(self.grid.is_free(state.getX(), state.getY()))


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

    def plan(self, start, goal, timeout=PLAN_TIMEOUT):
        si = ob.SpaceInformation(self.space)
        si.setStateValidityChecker(ValidityChecker(si, self.grid))
        si.setup()

        s = self.space.allocState()
        s.setX(start[0])
        s.setY(start[1])
        s.setYaw(start[2])

        g = self.space.allocState()
        g.setX(goal[0])
        g.setY(goal[1])
        g.setYaw(goal[2])

        pdef = ob.ProblemDefinition(si)
        pdef.setStartAndGoalStates(s, g)

        planner = og.RRTConnect(si)
        planner.setProblemDefinition(pdef)
        planner.setup()

        solved = planner.solve(timeout)
        if not solved:
            return None

        pdef.getSolutionPath().interpolate()
        states = pdef.getSolutionPath().getStates()
        return [(state.getX(), state.getY()) for state in states]


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
            PoseStamped, "pose_goal", self.goal_callback, 10
        )

        self.path_pub = self.create_publisher(Path, "rrt_path", 10)

    def map_callback(self, msg: OccupancyGrid):
        grid = np.reshape(msg.data, (msg.info.height, msg.info.width))
        origin = [msg.info.origin.position.x, msg.info.origin.position.y]
        self.bounds = [
            origin[0],
            origin[0] + msg.info.width * msg.info.resolution,
            origin[1],
            origin[1] + msg.info.height * msg.info.resolution,
        ]
        self.occ_grid = Grid(grid, msg.info.resolution, origin)
        self.get_logger().info("map received")
        self.try_plan()

    def pose_callback(self, msg: PoseStamped):
        self.pose_current = msg
        self.try_plan()

    def goal_callback(self, msg: PoseStamped):
        self.pose_goal = msg
        self.try_plan()

    # replan on every map, pose, and goal update
    def try_plan(self):
        if self.occ_grid is None or self.pose_current is None or self.pose_goal is None:
            return

        start = (
            self.pose_current.pose.position.x,
            self.pose_current.pose.position.y,
            self.yaw_from_pose(self.pose_current),
        )
        goal = (
            self.pose_goal.pose.position.x,
            self.pose_goal.pose.position.y,
            self.yaw_from_pose(self.pose_goal),
        )

        planner = RRTPlanner(
            self.occ_grid, turning_radius=TURNING_RADIUS, bounds=self.bounds
        )
        path = planner.plan(start, goal)

        if path is None:
            self.get_logger().error("planning failed")
            return

        self.get_logger().info(f"found path with {len(path)} states")
        self.path_pub.publish(self.make_path_msg(path))

    def yaw_from_pose(self, msg):
        q = msg.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return np.arctan2(siny_cosp, cosy_cosp)

    def make_path_msg(self, path):
        msg = Path()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in path:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            msg.poses.append(pose)
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = RRTNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
