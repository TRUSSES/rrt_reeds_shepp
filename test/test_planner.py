import math
import unittest

import numpy as np

from rrt_pkg.planner import Grid, VehicleFootprint, path_direction_metrics
from rrt_pkg.risk_map_generator import (
    RISK_MAP_SPECS,
    generate_risk_map_pair,
    make_risk_map_specs,
)


class TestGrid(unittest.TestCase):
    def test_five_risk_situations_each_generate_two_maps(self):
        self.assertEqual(len(RISK_MAP_SPECS), 5)
        for spec in RISK_MAP_SPECS:
            maps = generate_risk_map_pair(spec, seed=7)
            expected_shape = (spec.height, spec.width)
            self.assertEqual(maps.traversability.shape, expected_shape)
            self.assertEqual(maps.uncertainty.shape, expected_shape)
            self.assertTrue(np.all(maps.traversability >= 0.0))
            self.assertTrue(np.all(maps.uncertainty >= 0.0))
            self.assertTrue(np.all(maps.uncertainty <= 1.0))

    def test_configurable_risk_grid_preserves_world_size(self):
        specs = make_risk_map_specs(width=50, height=50, resolution=0.2)
        self.assertEqual(len(specs), 5)
        for spec in specs:
            self.assertEqual((spec.height, spec.width), (50, 50))
            self.assertEqual(spec.bounds, (0.0, 10.0, 0.0, 10.0))

    def test_extrapolated_map_threshold_and_metadata(self):
        class Value:
            pass

        msg = Value()
        msg.width = 2
        msg.height = 2
        msg.data = [1.0, 0.49, float("nan"), 0.5]
        msg.meta = Value()
        msg.meta.width = 2
        msg.meta.height = 2
        msg.meta.resolution = 0.25
        msg.meta.origin = Value()
        msg.meta.origin.position = Value()
        msg.meta.origin.position.x = -1.0
        msg.meta.origin.position.y = -2.0
        msg.meta.origin.orientation = Value()
        msg.meta.origin.orientation.x = 0.0
        msg.meta.origin.orientation.y = 0.0
        msg.meta.origin.orientation.z = 0.0
        msg.meta.origin.orientation.w = 1.0
        msg.header = Value()
        msg.header.frame_id = "risk_map"

        grid = Grid.from_extrapolated_map_msg(
            msg,
            traversability_threshold=0.5,
        )

        np.testing.assert_array_equal(grid.grid, [[0, 1], [1, 0]])
        self.assertEqual(grid.origin, (-1.0, -2.0))
        self.assertEqual(grid.resolution, 0.25)
        self.assertEqual(grid.frame_id, "risk_map")

    def test_occupancy_threshold_is_inclusive(self):
        class Value:
            pass

        msg = Value()
        msg.data = [0, 49, 50, -1]
        msg.info = Value()
        msg.info.width = 2
        msg.info.height = 2
        msg.info.resolution = 1.0
        msg.info.origin = Value()
        msg.info.origin.position = Value()
        msg.info.origin.position.x = 0.0
        msg.info.origin.position.y = 0.0
        msg.info.origin.orientation = Value()
        msg.info.origin.orientation.x = 0.0
        msg.info.origin.orientation.y = 0.0
        msg.info.origin.orientation.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.header = Value()
        msg.header.frame_id = "map"

        grid = Grid.from_occupancy_grid_msg(msg, occupancy_threshold=50)
        np.testing.assert_array_equal(grid.grid, [[0, 0], [1, 1]])

    def test_rotated_coordinate_conversion_and_bounds(self):
        grid = Grid(
            np.zeros((2, 3), dtype=np.uint8),
            1.0,
            origin=(10.0, 20.0),
            origin_yaw=math.pi / 2.0,
        )
        self.assertEqual(grid.world_to_grid(9.5, 20.5), (0, 0))
        x, y = grid.grid_to_world(0, 0)
        self.assertAlmostEqual(x, 9.5)
        self.assertAlmostEqual(y, 20.5)
        self.assertFalse(grid.is_free(100.0, 100.0))

    def test_inflation_and_negative_radius(self):
        data = np.zeros((5, 5), dtype=np.uint8)
        data[2, 2] = 1
        grid = Grid(data, 1.0)
        self.assertEqual(int(grid.inflate_obstacles(1.0).grid.sum()), 5)
        with self.assertRaises(ValueError):
            grid.inflate_obstacles(-0.1)

    def test_rectangle_footprint_checks_orientation_and_map_edge(self):
        data = np.zeros((7, 7), dtype=np.uint8)
        data[3, 5] = 1
        grid = Grid(data, 1.0)
        footprint = VehicleFootprint(length=4.0, width=1.0)
        self.assertFalse(grid.pose_is_free(3.5, 3.5, 0.0, footprint))
        self.assertTrue(grid.pose_is_free(3.5, 3.5, math.pi / 2.0, footprint))
        self.assertFalse(grid.pose_is_free(0.1, 0.1, 0.0, footprint))

    def test_invalid_footprint_dimensions(self):
        with self.assertRaises(ValueError):
            VehicleFootprint(0.0, 1.0)

    def test_path_direction_metrics_detects_reeds_shepp_cusps(self):
        path = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (1.5, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.5, 0.0, 0.0),
        ]
        metrics = path_direction_metrics(path)
        self.assertAlmostEqual(metrics.forward_length, 2.5)
        self.assertAlmostEqual(metrics.reverse_length, 1.0)
        self.assertEqual(metrics.gear_switches, 2)
        self.assertEqual(metrics.switch_indices, (2, 4))


if __name__ == "__main__":
    unittest.main()
