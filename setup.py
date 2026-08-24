from glob import glob

from setuptools import find_packages, setup

package_name = "rrt_pkg"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="azielben",
    maintainer_email="benaziel@seas.upenn.edu",
    description=(
        "ROS 2 RRT/Reeds-Shepp path planner with occupancy-grid collision checking"
    ),
    license="N/A",
    entry_points={
        "console_scripts": [
            "rrt_node = rrt_pkg.rrt_node:main",
        ],
    },
)
