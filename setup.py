from setuptools import find_packages, setup

package_name = "rrt_pkg"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="azielben",
    maintainer_email="benaziel@seas.upenn.edu",
    description="rrt w/ reeds shepp",
    license="N/A",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "rrt_node = rrt_pkg.rrt_node:main",
        ],
    },
)
