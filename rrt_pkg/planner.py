"""Shared RRTConnect/Reeds-Shepp planning utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VehicleFootprint:
    length: float
    width: float

    def sample_offsets(self, resolution: float) -> list[tuple[float, float]]:
        step = max(resolution * 0.5, 1e-6)
        half_length = self.length / 2.0
        half_width = self.width / 2.0
        x_count = max(1, math.ceil(self.length / step))
        y_count = max(1, math.ceil(self.width / step))
        xs = np.linspace(-half_length, half_length, x_count + 1)
        ys = np.linspace(-half_width, half_width, y_count + 1)
        offsets = [(float(x), float(y)) for x in xs for y in ys]
        corners = [
            (-half_length, -half_width),
            (-half_length, half_width),
            (half_length, -half_width),
            (half_length, half_width),
        ]
        return offsets + corners

    def __post_init__(self):
        if self.length <= 0.0 or self.width <= 0.0:
            raise ValueError("vehicle footprint dimensions must be positive")


@dataclass(frozen=True)
class PathDirectionMetrics:
    forward_length: float
    reverse_length: float
    gear_switches: int
    switch_indices: tuple[int, ...]


def yaw_from_quaternion(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def interpolate_yaw(start: float, goal: float, t: float) -> float:
    delta = math.atan2(math.sin(goal - start), math.cos(goal - start))
    return start + t * delta


class Grid:
    """Occupancy grid wrapper where 0 is free and non-zero is occupied."""

    def __init__(
        self,
        grid: np.ndarray,
        resolution: float,
        origin: tuple[float, float] | list[float] = (0.0, 0.0),
        origin_yaw: float = 0.0,
        frame_id: str = "map",
    ):
        self.grid = np.asarray(grid, dtype=np.uint8)
        self.resolution = resolution
        self.origin = (float(origin[0]), float(origin[1]))
        self.origin_yaw = origin_yaw
        self.frame_id = frame_id

    @classmethod
    def from_occupancy_grid_msg(
        cls,
        msg,
        occupancy_threshold: int = 50,
        unknown_is_occupied: bool = True,
        inflation_radius: float = 0.0,
    ) -> "Grid":
        data = np.asarray(msg.data, dtype=np.int16).reshape(
            msg.info.height,
            msg.info.width,
        )
        occupied = data >= occupancy_threshold
        unknown = data < 0
        if unknown_is_occupied:
            occupied |= unknown

        origin_pose = msg.info.origin
        origin = (origin_pose.position.x, origin_pose.position.y)
        origin_yaw = yaw_from_quaternion(origin_pose.orientation)
        grid = cls(
            occupied.astype(np.uint8),
            resolution=msg.info.resolution,
            origin=origin,
            origin_yaw=origin_yaw,
            frame_id=msg.header.frame_id or "map",
        )
        return grid.inflate_obstacles(inflation_radius)

    @classmethod
    def from_extrapolated_map_msg(
        cls,
        msg,
        traversability_threshold: float,
        inflation_radius: float = 0.0,
    ) -> "Grid":
        """Convert a risk_mapping ExtrapolatedMap into an occupancy grid.

        Terrain values below ``traversability_threshold`` are treated as
        occupied. ExtrapolatedMap data uses ROS row-major grid ordering.
        """
        width = int(msg.width or msg.meta.width)
        height = int(msg.height or msg.meta.height)
        if width <= 0 or height <= 0:
            raise ValueError("extrapolated map dimensions must be positive")

        data = np.asarray(msg.data, dtype=float)
        expected_size = width * height
        if data.size != expected_size:
            raise ValueError(
                "extrapolated map data size does not match its dimensions: "
                f"got {data.size}, expected {expected_size}"
            )
        if msg.meta.resolution <= 0.0:
            raise ValueError("extrapolated map resolution must be positive")

        origin_pose = msg.meta.origin
        origin = (origin_pose.position.x, origin_pose.position.y)
        origin_yaw = yaw_from_quaternion(origin_pose.orientation)
        occupied = (~np.isfinite(data)) | (data < traversability_threshold)
        grid = cls(
            occupied.reshape(height, width).astype(np.uint8),
            resolution=msg.meta.resolution,
            origin=origin,
            origin_yaw=origin_yaw,
            frame_id=msg.header.frame_id or "map",
        )
        return grid.inflate_obstacles(inflation_radius)

    @property
    def height(self) -> int:
        return self.grid.shape[0]

    @property
    def width(self) -> int:
        return self.grid.shape[1]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        corners = [
            self._map_to_world(0.0, 0.0),
            self._map_to_world(self.width * self.resolution, 0.0),
            self._map_to_world(0.0, self.height * self.resolution),
            self._map_to_world(
                self.width * self.resolution,
                self.height * self.resolution,
            ),
        ]
        xs, ys = zip(*corners)
        return (min(xs), max(xs), min(ys), max(ys))

    def planning_bounds(
        self,
        margin_cells: int = 1,
    ) -> tuple[float, float, float, float]:
        x_min, x_max, y_min, y_max = self.bounds
        margin = margin_cells * self.resolution
        if x_max - x_min <= 2.0 * margin or y_max - y_min <= 2.0 * margin:
            return self.bounds
        return (x_min + margin, x_max - margin, y_min + margin, y_max - margin)

    def inflate_obstacles(self, radius: float) -> "Grid":
        if radius < 0.0:
            raise ValueError("inflation radius cannot be negative")
        radius_cells = math.ceil(radius / self.resolution)
        if radius_cells <= 0:
            return self

        occupied = self.grid != 0
        padded = np.pad(occupied, radius_cells, constant_values=False)
        inflated = occupied.copy()

        for row_offset in range(-radius_cells, radius_cells + 1):
            for col_offset in range(-radius_cells, radius_cells + 1):
                if row_offset * row_offset + col_offset * col_offset > radius_cells**2:
                    continue
                row_start = radius_cells + row_offset
                col_start = radius_cells + col_offset
                inflated |= padded[
                    row_start:row_start + self.height,
                    col_start:col_start + self.width,
                ]

        return Grid(
            inflated.astype(np.uint8),
            resolution=self.resolution,
            origin=self.origin,
            origin_yaw=self.origin_yaw,
            frame_id=self.frame_id,
        )

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        dx = x - self.origin[0]
        dy = y - self.origin[1]
        cos_yaw = math.cos(self.origin_yaw)
        sin_yaw = math.sin(self.origin_yaw)
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        col = math.floor(local_x / self.resolution)
        row = math.floor(local_y / self.resolution)
        return row, col

    def grid_to_world(self, row: int, col: int) -> tuple[float, float]:
        local_x = (col + 0.5) * self.resolution
        local_y = (row + 0.5) * self.resolution
        return self._map_to_world(local_x, local_y)

    def is_in_bounds(self, x: float, y: float) -> bool:
        row, col = self.world_to_grid(x, y)
        return self.is_cell_in_bounds(row, col)

    def is_free(self, x: float, y: float) -> bool:
        row, col = self.world_to_grid(x, y)
        if not self.is_cell_in_bounds(row, col):
            return False
        return self.grid[row, col] == 0

    def pose_is_free(
        self,
        x: float,
        y: float,
        yaw: float,
        footprint: VehicleFootprint | None = None,
    ) -> bool:
        if footprint is None:
            return self.is_free(x, y)

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        for local_x, local_y in footprint.sample_offsets(self.resolution):
            world_x = x + cos_yaw * local_x - sin_yaw * local_y
            world_y = y + sin_yaw * local_x + cos_yaw * local_y
            if not self.is_free(world_x, world_y):
                return False
        return True

    def is_cell_in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def segment_is_free(
        self,
        start: tuple[float, float],
        goal: tuple[float, float],
        step_size: float | None = None,
    ) -> bool:
        if step_size is None:
            step_size = self.resolution * 0.5

        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        distance = math.hypot(dx, dy)
        steps = max(1, math.ceil(distance / step_size))
        for i in range(steps + 1):
            t = i / steps
            x = start[0] + t * dx
            y = start[1] + t * dy
            if not self.is_free(x, y):
                return False
        return True

    def pose_segment_is_free(
        self,
        start: tuple[float, float, float],
        goal: tuple[float, float, float],
        footprint: VehicleFootprint | None = None,
        step_size: float | None = None,
    ) -> bool:
        if step_size is None:
            step_size = self.resolution * 0.5

        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        distance = math.hypot(dx, dy)
        steps = max(1, math.ceil(distance / step_size))
        for i in range(steps + 1):
            t = i / steps
            x = start[0] + t * dx
            y = start[1] + t * dy
            yaw = interpolate_yaw(start[2], goal[2], t)
            if not self.pose_is_free(x, y, yaw, footprint=footprint):
                return False
        return True

    def path_is_free(
        self,
        path: list[tuple[float, float]],
        step_size: float | None = None,
    ) -> bool:
        if not path:
            return False
        return all(
            self.segment_is_free(start, goal, step_size=step_size)
            for start, goal in zip(path[:-1], path[1:])
        )

    def pose_path_is_free(
        self,
        path: list[tuple[float, float, float]],
        footprint: VehicleFootprint | None = None,
        step_size: float | None = None,
    ) -> bool:
        if not path:
            return False
        if not all(
            self.pose_is_free(x, y, yaw, footprint=footprint)
            for x, y, yaw in path
        ):
            return False
        return all(
            self.pose_segment_is_free(
                start,
                goal,
                footprint=footprint,
                step_size=step_size,
            )
            for start, goal in zip(path[:-1], path[1:])
        )

    def _map_to_world(self, local_x: float, local_y: float) -> tuple[float, float]:
        cos_yaw = math.cos(self.origin_yaw)
        sin_yaw = math.sin(self.origin_yaw)
        x = self.origin[0] + cos_yaw * local_x - sin_yaw * local_y
        y = self.origin[1] + sin_yaw * local_x + cos_yaw * local_y
        return x, y


def path_length(path: list[tuple[float, ...]] | None) -> float:
    if not path or len(path) < 2:
        return 0.0
    return sum(
        math.hypot(goal[0] - start[0], goal[1] - start[1])
        for start, goal in zip(path[:-1], path[1:])
    )


def path_direction_metrics(
    path: list[tuple[float, float, float]] | None,
    projection_tolerance: float = 1e-8,
) -> PathDirectionMetrics:
    """Measure forward/reverse travel and Reeds-Shepp cusp locations."""
    if projection_tolerance < 0.0:
        raise ValueError("projection_tolerance cannot be negative")
    if not path or len(path) < 2:
        return PathDirectionMetrics(0.0, 0.0, 0, ())

    forward_length = 0.0
    reverse_length = 0.0
    previous_direction = 0
    switch_indices: list[int] = []

    for index, (start, goal) in enumerate(zip(path[:-1], path[1:])):
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]
        distance = math.hypot(dx, dy)
        if distance <= projection_tolerance:
            continue

        midpoint_yaw = interpolate_yaw(start[2], goal[2], 0.5)
        projection = dx * math.cos(midpoint_yaw) + dy * math.sin(midpoint_yaw)
        if abs(projection) <= projection_tolerance:
            continue

        direction = 1 if projection > 0.0 else -1
        if direction > 0:
            forward_length += distance
        else:
            reverse_length += distance

        if previous_direction != 0 and direction != previous_direction:
            # This state joins the segments on either side of the cusp.
            switch_indices.append(index)
        previous_direction = direction

    return PathDirectionMetrics(
        forward_length,
        reverse_length,
        len(switch_indices),
        tuple(switch_indices),
    )
