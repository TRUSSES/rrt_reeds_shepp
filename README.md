# Risk-Aware RRT* with Reeds–Shepp Motion

Original RRT/Reeds–Shepp implementation by Benjamin Aziel. Extended with
risk-aware planning, vehicle collision checking, ROS 2 integration, configurable
benchmarks, and result evaluation.

## Changes from the original `main` version

The original version contained a single-file RRTConnect planner, a point-robot
collision check, one fixed circular-obstacle map, hard-coded parameters, and
two-dimensional path output.

The current version adds:

- Separate map, collision, OMPL, ROS node, and benchmark modules.
- `OccupancyGrid` and risk-mapping `ExtrapolatedMap` inputs.
- Traversability and uncertainty maps.
- Point, circular, and orientation-aware rectangular robot footprints.
- Obstacle inflation, safety margins, map-boundary checks, and complete motion
  validation.
- RRT, RRTConnect, and RRT* planner wrappers using a Reeds–Shepp state space.
- Forward/reverse distance and gear-switch detection.
- Four RRT* objectives:
  - shortest distance: `sum(ds)`
  - fastest traversal: `sum(ds / velocity)`
  - lowest uncertainty: `sum(uncertainty * ds)`
  - configurable overall score
- Five benchmark situations and multi-seed evaluation.
- Paired traversability/uncertainty visualizations.
- CSV metrics and JSON path-coordinate output.
- YAML configuration, ROS planning status, parameter validation, and unit tests.

## Planning objectives

The first three objectives are optimized in separate RRT* searches. They do not
use weighted combinations:

```text
Distance     D = sum(ds)
Time         T = sum(ds / velocity)
Uncertainty  U = sum(uncertainty * ds)
```

The fourth search uses the configured overall score:

```text
J = wd(D/Sd) + wt(T/St) + wu(U/Su) + wr((Lreverse + ps Nswitch)/Sr)
```

Here, `Lreverse` is reverse distance, `Nswitch` is the number of direction
changes, and `ps` is the gear-switch penalty. Lower cost is better.

All objectives retain Reeds–Shepp forward/reverse motion. The benchmark uses
RRT* because RRTConnect finds feasible paths but does not optimize path cost.

## Benchmark situations

Each situation generates both a traversability map and an uncertainty map:

1. Open terrain
2. Mixed obstacle shapes
3. Narrow passage
4. Parking/reversing manoeuvre
5. Dense random obstacles

The four selected paths are drawn on both maps:

- red: shortest distance
- orange: fastest traversal
- magenta: lowest uncertainty
- green: best overall score

Black and white circles mark the start and goal. `X` markers indicate
forward/reverse switches.

## Configuration

Edit `config/parameters.yaml`.

### Benchmark grid and overall score

```yaml
risk_benchmark:
  grid_width: 50
  grid_height: 50
  grid_resolution: 0.2

  overall_score:
    distance_weight: 0.25
    time_weight: 0.35
    uncertainty_weight: 0.30
    reverse_weight: 0.10
    distance_scale: 10.0
    time_scale: 10.0
    uncertainty_scale: 5.0
    reverse_scale: 5.0
    gear_switch_penalty: 0.5
```

`grid_resolution` is the physical size of one cell in metres. The default grid
is `50 * 0.2 = 10 m` wide and high. The four weights must sum to `1.0`.

### ROS planner and vehicle

```yaml
rrt_node:
  ros__parameters:
    turning_radius: 0.5
    planning_timeout: 5.0
    planner_type: rrt_connect
    collision_check_step: 0.05
    footprint_type: rectangle
    vehicle_length: 0.7
    vehicle_width: 0.45
    safety_margin: 0.05
```

`footprint_type` accepts `point`, `circle`, or `rectangle`.

## Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

The pip OMPL package does not provide `ReedsSheppStateSpace`. Build OMPL with
Python bindings and install it into this project's `.venv`:

```bash
sudo apt install castxml
git clone https://github.com/ompl/ompl.git /tmp/ompl
cd /tmp/ompl
git submodule update --init --recursive
mkdir build && cd build
cmake .. \
  -DOMPL_BUILD_PYTHON_BINDINGS=ON \
  -DOMPL_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=<path_to_repo>/.venv
make -j$(nproc)
make install

cp /tmp/ompl/py-bindings/ompl/*.py \
  <path_to_repo>/.venv/lib/python3.10/site-packages/ompl/
cp /tmp/ompl/build/py-bindings/ompl/_ompl*.so \
  <path_to_repo>/.venv/lib/python3.10/site-packages/ompl/
```

Verify the installation:

```bash
uv run python -c "from ompl import base as ob; print(ob.ReedsSheppStateSpace)"
```

## Run the complete benchmark

From the repository root:

```bash
uv run python scripts/benchmark_rrt.py \
  --params-file config/parameters.yaml \
  --planner-type auto \
  --vehicle-length 0.7 \
  --vehicle-width 0.45 \
  --timeout 10 \
  --output-dir results/<run_name>
```

The default run performs four objectives for twelve seeds in each of five
situations. Use a new output directory for each run to preserve earlier results.

### Quick test

```bash
uv run python scripts/benchmark_rrt.py \
  --params-file config/parameters.yaml \
  --maps empty \
  --seeds 1 4 7 \
  --vehicle-length 0.7 \
  --vehicle-width 0.45 \
  --timeout 2 \
  --output-dir results/quick_test
```

## Benchmark output

```text
results/<run_name>/
├── <situation>_traversability.png
├── <situation>_uncertainty.png
├── results.csv
└── selected_paths.json
```

`results.csv` contains all candidate metrics, including distance, time,
uncertainty, overall score, forward/reverse distance, gear switches, and planner
type. `selected_paths.json` contains the full coordinates and metrics for the
four selected paths per situation.

## ROS 2 interface

Subscriptions:

```text
map               nav_msgs/OccupancyGrid
extrapolated_map  trusses_custom_interfaces/ExtrapolatedMap
pose_current      geometry_msgs/PoseStamped
pose_goal         geometry_msgs/PoseStamped
```

Publications:

```text
rrt_path    nav_msgs/Path
rrt_status  std_msgs/String
```

The risk-map subscription is enabled when `trusses_custom_interfaces` is
available. The standard OccupancyGrid interface remains available independently.

Run the ROS node after building and sourcing the workspace:

```bash
ros2 run rrt_pkg rrt_node --ros-args \
  --params-file install/rrt_pkg/share/rrt_pkg/config/parameters.yaml
```

## Tests

```bash
UV_CACHE_DIR=/tmp/rrt-uv-cache \
uv run python -m unittest discover -s test -v
```

Tests cover occupancy/risk-map conversion, coordinate transforms, obstacle
inflation, rectangular footprint orientation, map boundaries, configurable
grids, and Reeds–Shepp direction-change metrics.

## Project structure

```text
config/parameters.yaml          ROS and benchmark parameters
rrt_pkg/map_generator.py        geometric benchmark maps
rrt_pkg/risk_map_generator.py   traversability/uncertainty map pairs
rrt_pkg/planner.py              grids, footprints, collision and path metrics
rrt_pkg/ompl_planner.py         Reeds–Shepp and RRT-family planners/objectives
rrt_pkg/rrt_node.py             ROS 2 subscriptions, planning and publications
scripts/benchmark_rrt.py        multi-scenario benchmark and visualization
test/test_planner.py            unit tests
results/                        saved benchmark runs
```
