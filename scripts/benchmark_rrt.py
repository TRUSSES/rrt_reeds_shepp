#!/usr/bin/env python3
"""Benchmark Reeds-Shepp paths and retain useful length/reversal tradeoffs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rrt_pkg.map_generator import MapSpec  # noqa: E402
from rrt_pkg.ompl_planner import RRTPlanner  # noqa: E402
from rrt_pkg.planner import (  # noqa: E402
    Grid,
    VehicleFootprint,
    path_direction_metrics,
    path_length,
)
from rrt_pkg.risk_map_generator import (  # noqa: E402
    RISK_MAP_SPECS,
    generate_risk_map_pair,
    make_risk_map_specs,
)


TURNING_RADIUS = 1.5
ROBOT_RADIUS = 0.0
VEHICLE_LENGTH = 0.0
VEHICLE_WIDTH = 0.0
PLAN_TIMEOUT = 5.0
PLANNER_TYPE = "auto"
PathState = tuple[float, float, float]
DEFAULT_PARAMS_FILE = REPO_ROOT / "config" / "parameters.yaml"
OBJECTIVE_ROLES = {
    "length": "shortest distance",
    "time": "fastest time",
    "uncertainty": "lowest uncertainty",
    "overall": "overall best",
}


@dataclass(frozen=True)
class OverallScoreConfig:
    distance_weight: float
    time_weight: float
    uncertainty_weight: float
    reverse_weight: float
    distance_scale: float
    time_scale: float
    uncertainty_scale: float
    reverse_scale: float
    gear_switch_penalty: float

    @property
    def planner_weights(self) -> tuple[float, float, float]:
        return (
            self.distance_weight / self.distance_scale,
            self.time_weight / self.time_scale,
            self.uncertainty_weight / self.uncertainty_scale,
        )


@dataclass
class BenchmarkResult:
    map_name: str
    seed: int
    solved: bool
    approximate_solution: bool
    elapsed_s: float
    path_length: float
    state_count: int
    forward_length: float
    reverse_length: float
    gear_switches: int
    travel_time_s: float
    total_uncertainty: float
    overall_score: float
    planner_type: str
    path_role: str = "candidate"


@dataclass(frozen=True)
class DisplayPath:
    result: BenchmarkResult
    path: list[PathState]


def integrate_path_field(
    path: list[PathState] | None,
    field: np.ndarray,
    spec: MapSpec,
    reciprocal: bool = False,
) -> float:
    """Integrate a raster field along a path using segment midpoints."""
    if not path or len(path) < 2:
        return math.inf
    total = 0.0
    for start, goal in zip(path[:-1], path[1:]):
        distance = math.hypot(goal[0] - start[0], goal[1] - start[1])
        x = 0.5 * (start[0] + goal[0])
        y = 0.5 * (start[1] + goal[1])
        col = min(spec.width - 1, max(0, math.floor(x / spec.resolution)))
        row = min(spec.height - 1, max(0, math.floor(y / spec.resolution)))
        value = float(field[row, col])
        total += distance / max(value, 1e-6) if reciprocal else distance * value
    return total


def plan(
    grid_data: np.ndarray,
    spec: MapSpec,
    seed: int,
    turning_radius: float,
    robot_radius: float,
    vehicle_length: float,
    vehicle_width: float,
    timeout: float,
    planner_type: str,
    optimization_objective: str,
    objective_field: np.ndarray | tuple[np.ndarray, np.ndarray] | None,
    objective_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[list[PathState] | None, float, bool, bool]:
    grid = Grid(grid_data, spec.resolution, origin=(0.0, 0.0)).inflate_obstacles(
        robot_radius
    )
    footprint = None
    if vehicle_length > 0.0 and vehicle_width > 0.0:
        footprint = VehicleFootprint(vehicle_length, vehicle_width)

    planner = RRTPlanner(
        grid,
        turning_radius=turning_radius,
        bounds=grid.planning_bounds(),
        planner_type=planner_type,
        footprint=footprint,
        optimization_objective=optimization_objective,
        objective_field=objective_field,
        objective_weights=objective_weights,
    )
    result = planner.plan(spec.start, spec.goal, timeout=timeout, seed=seed)
    return result.path, result.elapsed_s, result.solved, result.approximate_solution


def plot_map_result(
    map_data: np.ndarray,
    spec: MapSpec,
    paths: list[DisplayPath],
    output_path: Path,
    map_kind: str,
    robot_radius: float = 0.0,
    footprint: VehicleFootprint | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    xmax = spec.width * spec.resolution
    ymax = spec.height * spec.resolution
    cmap = "Blues" if map_kind == "traversability" else "Greys"
    image = ax.imshow(
        map_data,
        origin="lower",
        extent=[0, xmax, 0, ymax],
        cmap=cmap,
        interpolation="nearest",
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label(
        "traversability (m/s)" if map_kind == "traversability" else "uncertainty"
    )
    ax.scatter(
        spec.start[0],
        spec.start[1],
        s=150,
        marker="o",
        facecolor="black",
        edgecolor="white",
        linewidth=2.0,
        label="start",
        zorder=10,
    )
    ax.scatter(
        spec.goal[0],
        spec.goal[1],
        s=150,
        marker="o",
        facecolor="white",
        edgecolor="black",
        linewidth=2.0,
        label="goal",
        zorder=10,
    )

    role_styles = {
        "shortest distance": ("#E41A1C", "-"),
        "fastest time": ("#FF7F00", "-"),
        "lowest uncertainty": ("#C51BFF", "-"),
        "overall best": ("#00A651", "-"),
    }
    for display_index, display_path in enumerate(paths):
        result = display_path.result
        path = display_path.path
        color, linestyle = role_styles[result.path_role]
        label = (
            f"{result.path_role}: {result.path_length:.2f} m, "
            f"{result.travel_time_s:.2f} s, U={result.total_uncertainty:.2f}, "
            f"J={result.overall_score:.2f}"
        )
        if footprint is None and robot_radius <= 0.0:
            xs = [state[0] for state in path]
            ys = [state[1] for state in path]
            ax.plot(
                xs,
                ys,
                color=color,
                linestyle=linestyle,
                linewidth=2.8,
                alpha=1.0,
                label=label,
            )
        else:
            if footprint is not None:
                polygons = footprint_polygons(path, footprint)
            else:
                polygons = circle_footprint_polygons(path, robot_radius)
            ax.add_collection(
                PolyCollection(
                    polygons,
                    facecolors=color,
                    edgecolors=color,
                    alpha=0.14,
                    linewidths=0.35,
                )
            )
            ax.plot(
                [state[0] for state in path],
                [state[1] for state in path],
                color=color,
                linestyle=linestyle,
                linewidth=2.8,
                alpha=1.0,
                label=label,
            )

        metrics = path_direction_metrics(path)
        if metrics.switch_indices:
            ax.scatter(
                [path[index][0] for index in metrics.switch_indices],
                [path[index][1] for index in metrics.switch_indices],
                marker="X",
                s=38,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=5,
            )

    ax.set_title(f"{spec.name} — {map_kind}")
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="0.85", linewidth=0.4)
    ax.legend(loc="upper left", bbox_to_anchor=(1.22, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def footprint_polygons(
    path: list[PathState],
    footprint: VehicleFootprint,
    max_footprints: int = 90,
) -> list[list[tuple[float, float]]]:
    if not path:
        return []

    stride = max(1, math.ceil(len(path) / max_footprints))
    indices = list(range(0, len(path), stride))
    if indices[-1] != len(path) - 1:
        indices.append(len(path) - 1)

    half_length = footprint.length / 2.0
    half_width = footprint.width / 2.0
    local_corners = [
        (-half_length, -half_width),
        (half_length, -half_width),
        (half_length, half_width),
        (-half_length, half_width),
    ]
    polygons = []
    for index in indices:
        x, y, yaw = path[index]
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        polygons.append(
            [
                (
                    x + cos_yaw * local_x - sin_yaw * local_y,
                    y + sin_yaw * local_x + cos_yaw * local_y,
                )
                for local_x, local_y in local_corners
            ]
        )
    return polygons


def circle_footprint_polygons(
    path: list[PathState],
    radius: float,
    max_footprints: int = 90,
    vertices: int = 32,
) -> list[list[tuple[float, float]]]:
    if not path or radius <= 0.0:
        return []

    stride = max(1, math.ceil(len(path) / max_footprints))
    indices = list(range(0, len(path), stride))
    if indices[-1] != len(path) - 1:
        indices.append(len(path) - 1)

    angles = [2.0 * math.pi * index / vertices for index in range(vertices)]
    return [
        [
            (x + radius * math.cos(angle), y + radius * math.sin(angle))
            for angle in angles
        ]
        for x, y, _yaw in (path[index] for index in indices)
    ]


def write_csv(results: list[BenchmarkResult], output_path: Path) -> None:
    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "map_name",
                "seed",
                "solved",
                "approximate_solution",
                "elapsed_s",
                "path_length",
                "state_count",
                "forward_length",
                "reverse_length",
                "gear_switches",
                "travel_time_s",
                "total_uncertainty",
                "overall_score",
                "planner_type",
                "path_role",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "map_name": result.map_name,
                    "seed": result.seed,
                    "solved": result.solved,
                    "approximate_solution": result.approximate_solution,
                    "elapsed_s": f"{result.elapsed_s:.6f}",
                    "path_length": f"{result.path_length:.6f}",
                    "state_count": result.state_count,
                    "forward_length": f"{result.forward_length:.6f}",
                    "reverse_length": f"{result.reverse_length:.6f}",
                    "gear_switches": result.gear_switches,
                    "travel_time_s": f"{result.travel_time_s:.6f}",
                    "total_uncertainty": f"{result.total_uncertainty:.6f}",
                    "overall_score": f"{result.overall_score:.6f}",
                    "planner_type": result.planner_type,
                    "path_role": result.path_role,
                }
            )


def print_summary(results: list[BenchmarkResult]) -> None:
    by_map: dict[str, list[BenchmarkResult]] = {}
    for result in results:
        by_map.setdefault(result.map_name, []).append(result)

    print("\nBenchmark summary")
    print(
        f"{'map':<18} {'success':>9} {'avg time':>10} "
        f"{'avg length':>11} {'avg states':>11}"
    )
    for map_name, items in by_map.items():
        solved = [item for item in items if item.solved]
        success = f"{len(solved)}/{len(items)}"
        avg_time = sum(item.elapsed_s for item in items) / len(items)
        avg_length = sum(item.path_length for item in solved) / len(solved) if solved else 0.0
        avg_states = sum(item.state_count for item in solved) / len(solved) if solved else 0.0
        print(
            f"{map_name:<18} {success:>9} {avg_time:>9.3f}s "
            f"{avg_length:>11.3f} {avg_states:>11.1f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params-file", default=str(DEFAULT_PARAMS_FILE))
    parser.add_argument("--output-dir", default="benchmark_results")
    parser.add_argument("--timeout", type=float, default=PLAN_TIMEOUT)
    parser.add_argument("--turning-radius", type=float, default=TURNING_RADIUS)
    parser.add_argument("--robot-radius", type=float, default=ROBOT_RADIUS)
    parser.add_argument("--vehicle-length", type=float, default=VEHICLE_LENGTH)
    parser.add_argument("--vehicle-width", type=float, default=VEHICLE_WIDTH)
    parser.add_argument(
        "--planner-type",
        choices=["auto", "rrt_connect", "rrt", "rrt_star"],
        default=PLANNER_TYPE,
    )
    parser.add_argument(
        "--max-alternatives",
        type=int,
        default=2,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34],
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-map", help=argparse.SUPPRESS)
    parser.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-objective",
        choices=list(OBJECTIVE_ROLES),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--maps",
        nargs="+",
        default=[spec.name for spec in RISK_MAP_SPECS],
        help="Risk situations to run. Defaults to all five.",
    )
    return parser.parse_args()


def load_risk_parameters(parameters_path: Path) -> dict[str, str]:
    """Read scalar values below risk_benchmark without requiring PyYAML."""
    values: dict[str, str] = {}
    in_section = False
    for raw_line in parameters_path.read_text().splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not raw_line.startswith((" ", "\t")):
            in_section = stripped == "risk_benchmark:"
            continue
        if in_section and ":" in stripped:
            key, value = stripped.split(":", 1)
            scalar = value.split("#", 1)[0].strip()
            if scalar:
                values[key.strip()] = scalar
    return values


def load_risk_specs(parameters_path: Path) -> list[MapSpec]:
    values = load_risk_parameters(parameters_path)

    required = {"grid_width", "grid_height", "grid_resolution"}
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(
            f"{parameters_path} risk_benchmark is missing: {', '.join(missing)}"
        )
    width = int(values["grid_width"])
    height = int(values["grid_height"])
    resolution = float(values["grid_resolution"])
    if width <= 0 or height <= 0 or resolution <= 0.0:
        raise ValueError("benchmark grid width, height, and resolution must be positive")
    if width * resolution <= 9.0 or height * resolution <= 9.0:
        raise ValueError(
            "benchmark grid is too small for the configured goal; "
            "width*resolution and height*resolution must exceed 9.0 m"
        )
    return make_risk_map_specs(width, height, resolution)


def load_overall_score_config(parameters_path: Path) -> OverallScoreConfig:
    values = load_risk_parameters(parameters_path)
    names = {
        "distance_weight",
        "time_weight",
        "uncertainty_weight",
        "reverse_weight",
        "distance_scale",
        "time_scale",
        "uncertainty_scale",
        "reverse_scale",
        "gear_switch_penalty",
    }
    missing = sorted(names - values.keys())
    if missing:
        raise ValueError(
            f"{parameters_path} overall_score is missing: {', '.join(missing)}"
        )
    config = OverallScoreConfig(**{name: float(values[name]) for name in names})
    weights = (
        config.distance_weight,
        config.time_weight,
        config.uncertainty_weight,
        config.reverse_weight,
    )
    scales = (
        config.distance_scale,
        config.time_scale,
        config.uncertainty_scale,
        config.reverse_scale,
    )
    if any(weight < 0.0 for weight in weights):
        raise ValueError("overall score weights cannot be negative")
    if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
        raise ValueError("overall score weights must sum to 1.0")
    if any(scale <= 0.0 for scale in scales):
        raise ValueError("overall score scales must be positive")
    if config.gear_switch_penalty < 0.0:
        raise ValueError("gear_switch_penalty cannot be negative")
    return config


def calculate_overall_score(
    path_length_value: float,
    travel_time_s: float,
    total_uncertainty: float,
    reverse_length: float,
    gear_switches: int,
    config: OverallScoreConfig,
) -> float:
    reverse_cost = reverse_length + config.gear_switch_penalty * gear_switches
    return (
        config.distance_weight * path_length_value / config.distance_scale
        + config.time_weight * travel_time_s / config.time_scale
        + config.uncertainty_weight
        * total_uncertainty
        / config.uncertainty_scale
        + config.reverse_weight * reverse_cost / config.reverse_scale
    )


def result_from_worker(
    spec: MapSpec,
    seed: int,
    timeout: float,
    turning_radius: float,
    robot_radius: float,
    vehicle_length: float,
    vehicle_width: float,
    planner_type: str,
    optimization_objective: str,
    traversability: np.ndarray,
    uncertainty: np.ndarray,
    parameters_path: Path,
    overall_config: OverallScoreConfig,
    output_dir: Path,
) -> tuple[BenchmarkResult, list[PathState] | None]:
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix=f"{spec.name}_{seed}_",
        dir=output_dir,
        delete=False,
    ) as tmp:
        worker_output = Path(tmp.name)

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--worker-map",
        spec.name,
        "--worker-seed",
        str(seed),
        "--worker-output",
        str(worker_output),
        "--params-file",
        str(parameters_path),
        "--timeout",
        str(timeout),
        "--turning-radius",
        str(turning_radius),
        "--robot-radius",
        str(robot_radius),
        "--vehicle-length",
        str(vehicle_length),
        "--vehicle-width",
        str(vehicle_width),
        "--planner-type",
        planner_type,
        "--worker-objective",
        optimization_objective,
    ]
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)
        raise RuntimeError(f"worker failed for map={spec.name} seed={seed}")

    with worker_output.open() as json_file:
        payload = json.load(json_file)
    worker_output.unlink(missing_ok=True)

    path = payload["path"]
    if path is not None:
        path = [(float(x), float(y), float(yaw)) for x, y, yaw in path]

    direction_metrics = path_direction_metrics(path)
    length_value = float(payload["path_length"])
    travel_time = integrate_path_field(
        path,
        traversability,
        spec,
        reciprocal=True,
    )
    uncertainty_cost = integrate_path_field(path, uncertainty, spec)
    result = BenchmarkResult(
        map_name=spec.name,
        seed=seed,
        solved=bool(payload["solved"]),
        approximate_solution=bool(payload["approximate_solution"]),
        elapsed_s=float(payload["elapsed_s"]),
        path_length=length_value,
        state_count=int(payload["state_count"]),
        forward_length=direction_metrics.forward_length,
        reverse_length=direction_metrics.reverse_length,
        gear_switches=direction_metrics.gear_switches,
        travel_time_s=travel_time,
        total_uncertainty=uncertainty_cost,
        overall_score=calculate_overall_score(
            length_value,
            travel_time,
            uncertainty_cost,
            direction_metrics.reverse_length,
            direction_metrics.gear_switches,
            overall_config,
        ),
        planner_type=planner_type,
        path_role=OBJECTIVE_ROLES[optimization_objective],
    )
    return result, path


def run_worker(args: argparse.Namespace) -> None:
    specs = {
        spec.name: spec for spec in load_risk_specs(Path(args.params_file))
    }
    if args.worker_map not in specs:
        raise SystemExit(f"Unknown worker map: {args.worker_map}")

    spec = specs[args.worker_map]
    risk_maps = generate_risk_map_pair(spec, seed=0)
    overall_config = load_overall_score_config(Path(args.params_file))
    grid = risk_maps.occupancy
    planner_type = "rrt_star" if args.planner_type == "auto" else args.planner_type
    objective_field = {
        "length": None,
        "time": risk_maps.traversability,
        "uncertainty": risk_maps.uncertainty,
        "overall": (risk_maps.traversability, risk_maps.uncertainty),
    }[args.worker_objective]
    path, elapsed_s, solved, approximate_solution = plan(
        grid,
        spec,
        seed=args.worker_seed,
        turning_radius=args.turning_radius,
        robot_radius=args.robot_radius,
        vehicle_length=args.vehicle_length,
        vehicle_width=args.vehicle_width,
        timeout=args.timeout,
        planner_type=planner_type,
        optimization_objective=args.worker_objective,
        objective_field=objective_field,
        objective_weights=overall_config.planner_weights,
    )
    payload = {
        "solved": solved,
        "approximate_solution": approximate_solution,
        "elapsed_s": elapsed_s,
        "path_length": path_length(path),
        "state_count": len(path) if path else 0,
        "path": path,
    }
    with Path(args.worker_output).open("w") as json_file:
        json.dump(payload, json_file)


def select_display_paths(
    candidates: list[tuple[BenchmarkResult, list[PathState] | None]],
) -> list[DisplayPath]:
    """Select the best candidate for each requested risk-aware criterion."""
    solved = [(result, path) for result, path in candidates if result.solved and path]
    if not solved:
        return []

    criteria = [
        ("shortest distance", lambda item: item[0].path_length),
        ("fastest time", lambda item: item[0].travel_time_s),
        ("lowest uncertainty", lambda item: item[0].total_uncertainty),
        ("overall best", lambda item: item[0].overall_score),
    ]
    selected = []
    for role, key in criteria:
        objective_candidates = [
            item for item in solved if item[0].path_role == role
        ]
        if not objective_candidates:
            continue
        result, path = min(objective_candidates, key=key)
        selected.append(DisplayPath(result, path))
    return selected


def write_selected_paths(
    selected_by_map: dict[str, list[DisplayPath]],
    output_path: Path,
) -> None:
    payload = {}
    for map_name, paths in selected_by_map.items():
        payload[map_name] = [
            {
                "criterion": item.result.path_role,
                "seed": item.result.seed,
                "planner_type": item.result.planner_type,
                "path_length": item.result.path_length,
                "travel_time_s": item.result.travel_time_s,
                "total_uncertainty": item.result.total_uncertainty,
                "overall_score": item.result.overall_score,
                "reverse_length": item.result.reverse_length,
                "gear_switches": item.result.gear_switches,
                "path": item.path,
            }
            for item in paths
        ]
    with output_path.open("w") as output_file:
        json.dump(payload, output_file, indent=2)


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_specs = load_risk_specs(Path(args.params_file))
    overall_config = load_overall_score_config(Path(args.params_file))
    if args.planner_type not in {"auto", "rrt_star"}:
        raise SystemExit(
            "independent length/time/uncertainty optimization requires "
            "--planner-type auto or rrt_star"
        )
    specs = [spec for spec in all_specs if spec.name in set(args.maps)]
    unknown_maps = sorted(set(args.maps) - {spec.name for spec in all_specs})
    if unknown_maps:
        raise SystemExit(f"Unknown maps: {', '.join(unknown_maps)}")

    results: list[BenchmarkResult] = []
    selected_by_map: dict[str, list[DisplayPath]] = {}
    for spec in specs:
        # Keep both risk maps fixed per situation; OMPL randomness varies by seed.
        risk_maps = generate_risk_map_pair(spec, seed=0)
        grid = risk_maps.occupancy
        planner_type = "rrt_star"
        candidates: list[tuple[BenchmarkResult, list[PathState] | None]] = []
        print(f"Running {spec.name} with three independent and one overall objective...")
        for objective, role in OBJECTIVE_ROLES.items():
            print(f"  Optimizing only: {role}")
            for seed in args.seeds:
                result, path = result_from_worker(
                    spec,
                    seed,
                    args.timeout,
                    args.turning_radius,
                    args.robot_radius,
                    args.vehicle_length,
                    args.vehicle_width,
                    planner_type,
                    objective,
                    risk_maps.traversability,
                    risk_maps.uncertainty,
                    Path(args.params_file),
                    overall_config,
                    output_dir,
                )
                candidates.append((result, path))
                results.append(result)
                if result.solved:
                    status = "solved"
                elif result.approximate_solution:
                    status = "approx"
                else:
                    status = "failed"
                print(
                    f"    seed {seed:<3} {status:<6} "
                    f"length={result.path_length:.3f} "
                    f"time-cost={result.travel_time_s:.3f}s "
                    f"uncertainty={result.total_uncertainty:.3f}"
                )

        footprint = None
        if args.vehicle_length > 0.0 and args.vehicle_width > 0.0:
            footprint = VehicleFootprint(args.vehicle_length, args.vehicle_width)
        display_paths = select_display_paths(candidates)
        selected_by_map[spec.name] = display_paths
        plot_map_result(
            risk_maps.traversability,
            spec,
            display_paths,
            output_dir / f"{spec.name}_traversability.png",
            map_kind="traversability",
            robot_radius=args.robot_radius,
            footprint=footprint,
        )
        plot_map_result(
            risk_maps.uncertainty,
            spec,
            display_paths,
            output_dir / f"{spec.name}_uncertainty.png",
            map_kind="uncertainty",
            robot_radius=args.robot_radius,
            footprint=footprint,
        )

    write_csv(results, output_dir / "results.csv")
    write_selected_paths(selected_by_map, output_dir / "selected_paths.json")
    print_summary(results)
    print(f"\nWrote results to {output_dir}")


if __name__ == "__main__":
    main()
