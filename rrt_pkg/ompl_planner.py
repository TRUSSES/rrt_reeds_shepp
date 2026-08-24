"""OMPL-backed Reeds-Shepp + RRTConnect planner."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from ompl import base as ob
from ompl import geometric as og
from ompl import util as ou

from rrt_pkg.planner import Grid, VehicleFootprint


@dataclass(frozen=True)
class PlanResult:
    path: list[tuple[float, float, float]] | None
    elapsed_s: float
    solved: bool
    approximate_solution: bool = False


class ValidityChecker(ob.StateValidityChecker):
    def __init__(
        self,
        si: ob.SpaceInformation,
        grid: Grid,
        footprint: VehicleFootprint | None = None,
    ):
        super().__init__(si)
        self.grid = grid
        self.footprint = footprint

    def isValid(self, state: ob.State) -> bool:
        return bool(
            self.grid.pose_is_free(
                state.getX(),
                state.getY(),
                state.getYaw(),
                footprint=self.footprint,
            )
        )


class RasterCostObjective(ob.StateCostIntegralObjective):
    """Integrate one raster-derived cost without mixing other criteria."""

    def __init__(
        self,
        si: ob.SpaceInformation,
        grid: Grid,
        field: np.ndarray,
        reciprocal: bool,
    ):
        super().__init__(si, True)
        self.grid = grid
        self.field = np.asarray(field, dtype=float)
        self.reciprocal = reciprocal
        if self.field.shape != self.grid.grid.shape:
            raise ValueError("objective cost field must match the occupancy grid")

    def stateCost(self, state: ob.State) -> ob.Cost:
        row, col = self.grid.world_to_grid(state.getX(), state.getY())
        if not self.grid.is_cell_in_bounds(row, col):
            return ob.Cost(float("inf"))
        value = float(self.field[row, col])
        if not math.isfinite(value):
            return ob.Cost(float("inf"))
        cost = 1.0 / max(value, 1e-6) if self.reciprocal else max(value, 0.0)
        return ob.Cost(cost)


class RRTPlanner:
    def __init__(
        self,
        grid: Grid,
        turning_radius: float,
        bounds: tuple[float, float, float, float] | list[float],
        planner_type: str = "rrt_connect",
        collision_check_step: float | None = None,
        footprint: VehicleFootprint | None = None,
        optimization_objective: str = "length",
        objective_field: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
        objective_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ):
        if turning_radius <= 0.0:
            raise ValueError("turning_radius must be greater than zero")
        if collision_check_step is not None and collision_check_step <= 0.0:
            raise ValueError("collision_check_step must be greater than zero")
        self.grid = grid
        self.turning_radius = turning_radius
        self.bounds = bounds
        self.planner_type = planner_type.lower()
        self.collision_check_step = collision_check_step
        self.footprint = footprint
        self.optimization_objective = optimization_objective.lower()
        self.objective_field = objective_field
        self.objective_weights = objective_weights
        if self.optimization_objective not in {
            "length",
            "time",
            "uncertainty",
            "overall",
        }:
            raise ValueError(
                "optimization_objective must be length, time, uncertainty, or overall"
            )
        if self.optimization_objective != "length" and objective_field is None:
            raise ValueError(
                f"{self.optimization_objective} optimization requires a cost field"
            )
        if self.planner_type != "rrt_star" and self.optimization_objective != "length":
            raise ValueError(
                f"{self.optimization_objective} optimization requires rrt_star"
            )
        self.space = self._make_state_space()

    def _make_state_space(self) -> ob.ReedsSheppStateSpace:
        space = ob.ReedsSheppStateSpace(self.turning_radius)
        bounds = ob.RealVectorBounds(2)
        bounds.setLow(0, self.bounds[0])
        bounds.setHigh(0, self.bounds[1])
        bounds.setLow(1, self.bounds[2])
        bounds.setHigh(1, self.bounds[3])
        space.setBounds(bounds)
        return space

    def plan(
        self,
        start: tuple[float, float, float],
        goal: tuple[float, float, float],
        timeout: float,
        seed: int | None = None,
    ) -> PlanResult:
        if seed is not None:
            ou.RNG.setSeed(seed)

        si = ob.SpaceInformation(self.space)
        si.setStateValidityChecker(ValidityChecker(si, self.grid, self.footprint))
        si.setMotionValidator(ob.ReedsSheppMotionValidator(si))
        if self.collision_check_step is not None:
            fraction = min(
                1.0,
                self.collision_check_step / self.space.getMaximumExtent(),
            )
            si.setStateValidityCheckingResolution(fraction)
        si.setup()

        start_state = self.space.allocState()
        start_state.setX(start[0])
        start_state.setY(start[1])
        start_state.setYaw(start[2])

        goal_state = self.space.allocState()
        goal_state.setX(goal[0])
        goal_state.setY(goal[1])
        goal_state.setYaw(goal[2])

        pdef = ob.ProblemDefinition(si)
        pdef.setStartAndGoalStates(start_state, goal_state)
        if self.planner_type == "rrt_star":
            pdef.setOptimizationObjective(self._make_optimization_objective(si))

        planner = self._make_planner(si)
        planner.setProblemDefinition(pdef)
        planner.setup()

        begin = time.perf_counter()
        solved = planner.solve(timeout)
        elapsed_s = time.perf_counter() - begin
        if not solved:
            return PlanResult(None, elapsed_s, False, False)

        approximate = bool(pdef.hasApproximateSolution())
        if not pdef.hasExactSolution():
            return PlanResult(None, elapsed_s, False, approximate)

        solution = pdef.getSolutionPath()
        solution.interpolate()
        path_poses = [
            (state.getX(), state.getY(), state.getYaw())
            for state in solution.getStates()
        ]
        states = solution.getStates()
        if not all(si.isValid(state) for state in states):
            return PlanResult(None, elapsed_s, False, False)
        motions_are_valid = all(
            si.checkMotion(start, goal)
            for start, goal in zip(states[:-1], states[1:])
        )
        if not motions_are_valid:
            return PlanResult(None, elapsed_s, False, False)
        return PlanResult(path_poses, elapsed_s, True, False)

    def _make_optimization_objective(
        self,
        si: ob.SpaceInformation,
    ) -> ob.OptimizationObjective:
        if self.optimization_objective == "length":
            return ob.PathLengthOptimizationObjective(si)
        if self.optimization_objective == "overall":
            speed_field, uncertainty_field = self.objective_field
            objective = ob.MultiOptimizationObjective(si)
            objective.addObjective(
                ob.PathLengthOptimizationObjective(si),
                self.objective_weights[0],
            )
            objective.addObjective(
                RasterCostObjective(si, self.grid, speed_field, reciprocal=True),
                self.objective_weights[1],
            )
            objective.addObjective(
                RasterCostObjective(si, self.grid, uncertainty_field, reciprocal=False),
                self.objective_weights[2],
            )
            return objective
        return RasterCostObjective(
            si,
            self.grid,
            self.objective_field,
            reciprocal=self.optimization_objective == "time",
        )

    def _make_planner(self, si: ob.SpaceInformation) -> ob.Planner:
        planners = {
            "rrt_connect": og.RRTConnect,
            "rrt": og.RRT,
            "rrt_star": og.RRTstar,
        }
        try:
            planner_class = planners[self.planner_type]
        except KeyError as exc:
            choices = ", ".join(sorted(planners))
            raise ValueError(
                f"unknown planner_type '{self.planner_type}'; choose from {choices}"
            ) from exc
        return planner_class(si)
