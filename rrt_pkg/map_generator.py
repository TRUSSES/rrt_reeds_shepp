"""
Map generators for RRT/Reeds-Shepp planning experiments.

The generated grids follow the same convention used by ``rrt_node.py``:
0 means free space and 1 means occupied space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class MapSpec:
    name: str
    width: int = 100
    height: int = 100
    resolution: float = 0.1
    start: tuple[float, float, float] = (1.0, 1.0, 0.0)
    goal: tuple[float, float, float] = (9.0, 9.0, 0.0)
    description: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (0.0, self.width * self.resolution, 0.0, self.height * self.resolution)


MapFactory = Callable[[MapSpec, np.random.Generator], np.ndarray]


def world_mesh(spec: MapSpec) -> tuple[np.ndarray, np.ndarray]:
    xs = np.arange(spec.width) * spec.resolution
    ys = np.arange(spec.height) * spec.resolution
    return np.meshgrid(xs, ys)


def add_circle(grid: np.ndarray, spec: MapSpec, cx: float, cy: float, radius: float) -> None:
    xx, yy = world_mesh(spec)
    grid[(xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2] = 1


def add_rectangle(
    grid: np.ndarray,
    spec: MapSpec,
    cx: float,
    cy: float,
    width: float,
    height: float,
    yaw: float = 0.0,
) -> None:
    xx, yy = world_mesh(spec)
    dx = xx - cx
    dy = yy - cy
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    local_x = cos_yaw * dx + sin_yaw * dy
    local_y = -sin_yaw * dx + cos_yaw * dy
    grid[(np.abs(local_x) <= width / 2.0) & (np.abs(local_y) <= height / 2.0)] = 1


def add_polygon(grid: np.ndarray, spec: MapSpec, vertices: list[tuple[float, float]]) -> None:
    xx, yy = world_mesh(spec)
    inside = np.zeros_like(grid, dtype=bool)
    poly = np.asarray(vertices)
    x = xx
    y = yy
    x0, y0 = poly[-1]
    for x1, y1 in poly:
        crosses = ((y0 > y) != (y1 > y)) & (
            x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-12) + x0
        )
        inside ^= crosses
        x0, y0 = x1, y1
    grid[inside] = 1


def add_border(grid: np.ndarray, thickness: int = 1) -> None:
    grid[:thickness, :] = 1
    grid[-thickness:, :] = 1
    grid[:, :thickness] = 1
    grid[:, -thickness:] = 1


def _clear_pose_area(
    grid: np.ndarray,
    spec: MapSpec,
    pose: tuple[float, float, float],
    radius: float = 0.35,
) -> None:
    xx, yy = world_mesh(spec)
    grid[(xx - pose[0]) ** 2 + (yy - pose[1]) ** 2 <= radius**2] = 0


def _finish(grid: np.ndarray, spec: MapSpec) -> np.ndarray:
    _clear_pose_area(grid, spec, spec.start)
    _clear_pose_area(grid, spec, spec.goal)
    return grid.astype(np.uint8)


def empty_map(spec: MapSpec, rng: np.random.Generator) -> np.ndarray:
    del rng
    grid = np.zeros((spec.height, spec.width), dtype=np.uint8)
    add_border(grid)
    return _finish(grid, spec)


def circle_field(spec: MapSpec, rng: np.random.Generator) -> np.ndarray:
    del rng
    grid = np.zeros((spec.height, spec.width), dtype=np.uint8)
    for cx, cy, radius in [
        (2.0, 2.0, 0.5),
        (5.0, 5.0, 1.0),
        (7.0, 3.0, 0.7),
        (3.0, 7.0, 0.8),
        (8.0, 7.0, 0.5),
    ]:
        add_circle(grid, spec, cx, cy, radius)
    add_border(grid)
    return _finish(grid, spec)


def rectangle_field(spec: MapSpec, rng: np.random.Generator) -> np.ndarray:
    del rng
    grid = np.zeros((spec.height, spec.width), dtype=np.uint8)
    rectangles = [
        (3.0, 2.4, 0.7, 2.8, 0.2),
        (5.5, 5.0, 3.0, 0.6, -0.35),
        (7.8, 7.0, 0.8, 2.5, 0.45),
        (2.4, 7.3, 2.0, 0.6, 0.0),
    ]
    for rect in rectangles:
        add_rectangle(grid, spec, *rect)
    add_border(grid)
    return _finish(grid, spec)


def mixed_shapes(spec: MapSpec, rng: np.random.Generator) -> np.ndarray:
    del rng
    grid = np.zeros((spec.height, spec.width), dtype=np.uint8)
    add_circle(grid, spec, 2.8, 2.8, 0.7)
    add_circle(grid, spec, 7.3, 6.8, 0.9)
    add_rectangle(grid, spec, 5.0, 4.5, 3.0, 0.65, 0.25)
    add_rectangle(grid, spec, 3.2, 7.4, 1.6, 0.6, -0.45)
    add_polygon(grid, spec, [(6.6, 2.0), (8.2, 2.5), (7.5, 4.0), (6.3, 3.2)])
    add_border(grid)
    return _finish(grid, spec)


def narrow_passage(spec: MapSpec, rng: np.random.Generator) -> np.ndarray:
    del rng
    grid = np.zeros((spec.height, spec.width), dtype=np.uint8)
    add_rectangle(grid, spec, 5.0, 3.0, 7.0, 0.8)
    add_rectangle(grid, spec, 5.0, 7.0, 7.0, 0.8)
    add_rectangle(grid, spec, 3.0, 5.0, 0.8, 3.2)
    add_rectangle(grid, spec, 7.0, 5.0, 0.8, 3.2)
    add_border(grid)
    return _finish(grid, spec)


def maze(spec: MapSpec, rng: np.random.Generator) -> np.ndarray:
    del rng
    grid = np.zeros((spec.height, spec.width), dtype=np.uint8)
    walls = [
        (2.0, 4.0, 0.45, 6.0),
        (4.0, 6.0, 0.45, 6.0),
        (6.0, 4.0, 0.45, 6.0),
        (8.0, 6.0, 0.45, 6.0),
        (3.0, 2.0, 2.0, 0.45),
        (5.0, 8.0, 2.0, 0.45),
        (7.0, 2.0, 2.0, 0.45),
    ]
    for wall in walls:
        add_rectangle(grid, spec, *wall)
    add_border(grid)
    return _finish(grid, spec)


def parking_lot(spec: MapSpec, rng: np.random.Generator) -> np.ndarray:
    del rng
    grid = np.zeros((spec.height, spec.width), dtype=np.uint8)
    for x in [2.5, 4.0, 5.5, 7.0]:
        add_rectangle(grid, spec, x, 6.4, 0.18, 2.3)
    add_rectangle(grid, spec, 4.75, 7.6, 5.0, 0.18)
    add_rectangle(grid, spec, 4.75, 5.2, 5.0, 0.18)
    add_rectangle(grid, spec, 7.8, 3.2, 1.2, 2.4, 0.25)
    add_circle(grid, spec, 3.0, 3.1, 0.45)
    add_border(grid)
    return _finish(grid, spec)


def cluttered_random(spec: MapSpec, rng: np.random.Generator) -> np.ndarray:
    grid = np.zeros((spec.height, spec.width), dtype=np.uint8)
    for _ in range(14):
        cx = rng.uniform(1.2, 8.8)
        cy = rng.uniform(1.2, 8.8)
        if rng.random() < 0.55:
            add_circle(grid, spec, cx, cy, rng.uniform(0.25, 0.65))
        else:
            add_rectangle(
                grid,
                spec,
                cx,
                cy,
                rng.uniform(0.4, 1.3),
                rng.uniform(0.4, 1.8),
                rng.uniform(-np.pi, np.pi),
            )
    add_border(grid)
    return _finish(grid, spec)


def blocked(spec: MapSpec, rng: np.random.Generator) -> np.ndarray:
    del rng
    grid = np.zeros((spec.height, spec.width), dtype=np.uint8)
    add_rectangle(grid, spec, 5.0, 5.0, 0.9, 10.0)
    add_border(grid)
    return _finish(grid, spec)


MAP_FACTORIES: dict[str, MapFactory] = {
    "empty": empty_map,
    "circle_field": circle_field,
    "rectangle_field": rectangle_field,
    "mixed_shapes": mixed_shapes,
    "narrow_passage": narrow_passage,
    "maze": maze,
    "parking_lot": parking_lot,
    "cluttered_random": cluttered_random,
    "blocked": blocked,
}


DEFAULT_MAP_SPECS: list[MapSpec] = [
    MapSpec("empty", description="No internal obstacles; sanity-check baseline."),
    MapSpec("circle_field", description="Original fixed circular-obstacle map."),
    MapSpec("rectangle_field", description="Fixed rectangle and rotated-rectangle map."),
    MapSpec("mixed_shapes", description="Circles, rectangles, and polygon obstacles."),
    MapSpec("narrow_passage", description="Four walls forming a constrained central passage."),
    MapSpec("maze", description="Alternating wall maze with several turns."),
    MapSpec(
        "parking_lot",
        start=(1.0, 1.2, 0.0),
        goal=(6.35, 6.4, np.pi / 2.0),
        description="Parking-style scene where heading matters.",
    ),
    MapSpec("cluttered_random", description="Seeded random mix of circles and rectangles."),
    MapSpec("blocked", description="Intentionally unsolvable separated map."),
]


def generate_map(spec: MapSpec, seed: int = 0) -> np.ndarray:
    try:
        factory = MAP_FACTORIES[spec.name]
    except KeyError as exc:
        names = ", ".join(sorted(MAP_FACTORIES))
        raise ValueError(f"unknown map '{spec.name}'. Available maps: {names}") from exc
    return factory(spec, np.random.default_rng(seed))
