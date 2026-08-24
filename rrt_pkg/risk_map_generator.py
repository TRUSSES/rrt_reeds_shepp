"""Deterministic random risk-map pairs for planner benchmarking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rrt_pkg.map_generator import MapSpec, generate_map


@dataclass(frozen=True)
class RiskMapPair:
    traversability: np.ndarray
    uncertainty: np.ndarray
    occupancy: np.ndarray


def make_risk_map_specs(
    width: int = 100,
    height: int = 100,
    resolution: float = 0.1,
) -> list[MapSpec]:
    common = {"width": width, "height": height, "resolution": resolution}
    return [
        MapSpec(
            "empty",
            **common,
            description="Open terrain with spatial speed variation.",
        ),
        MapSpec(
            "mixed_shapes",
            **common,
            description="Mixed obstacle shapes and route choices.",
        ),
        MapSpec(
            "narrow_passage",
            **common,
            description="Constrained passages with limited clearance.",
        ),
        MapSpec(
            "parking_lot",
            **common,
            start=(1.0, 1.2, 0.0),
            goal=(6.35, 6.4, np.pi / 2.0),
            description="Parking manoeuvre where Reeds-Shepp reversing is useful.",
        ),
        MapSpec(
            "cluttered_random",
            **common,
            description="Dense random terrain and obstacles.",
        ),
    ]


RISK_MAP_SPECS = make_risk_map_specs()


def _smooth_random(
    shape: tuple[int, int],
    rng: np.random.Generator,
    passes: int = 10,
) -> np.ndarray:
    values = rng.random(shape)
    for _ in range(passes):
        values = (
            values
            + np.roll(values, 1, axis=0)
            + np.roll(values, -1, axis=0)
            + np.roll(values, 1, axis=1)
            + np.roll(values, -1, axis=1)
        ) / 5.0
    minimum = float(values.min())
    span = float(values.max() - minimum)
    return (values - minimum) / span if span > 0.0 else np.zeros(shape)


def generate_risk_map_pair(spec: MapSpec, seed: int = 0) -> RiskMapPair:
    """Generate traversability and uncertainty maps for one test situation."""
    rng = np.random.default_rng(seed)
    occupancy = generate_map(spec, seed=seed)
    speed_field = _smooth_random(occupancy.shape, rng)
    uncertainty_field = _smooth_random(occupancy.shape, rng)

    # Metres per second. Obstacles have zero traversability.
    traversability = 0.35 + 1.65 * speed_field
    traversability[occupancy != 0] = 0.0

    # Standard-deviation-like risk in [0.02, 1.0]. Obstacles are maximally unknown.
    uncertainty = 0.02 + 0.98 * uncertainty_field
    uncertainty[occupancy != 0] = 1.0
    return RiskMapPair(traversability, uncertainty, occupancy.astype(np.uint8))


def choose_planner(spec: MapSpec, occupancy: np.ndarray) -> str:
    """Choose between asymptotic quality and rapid constrained-space exploration."""
    occupied_fraction = float(np.mean(occupancy != 0))
    if spec.name in {"narrow_passage", "parking_lot"} or occupied_fraction >= 0.18:
        return "rrt_connect"
    return "rrt_star"
