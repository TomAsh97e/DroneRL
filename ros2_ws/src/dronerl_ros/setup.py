from glob import glob

from setuptools import setup

NODES = ["depth_node", "policy_node", "safety_node", "px4_bridge", "mission_node", "monitor_node"]

setup(
    name="dronerl_ros",
    version="0.1.0",
    packages=["dronerl_ros"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/dronerl_ros"]),
        ("share/dronerl_ros", ["package.xml"]),
        ("share/dronerl_ros/launch", glob("launch/*.py")),
        ("share/dronerl_ros/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": [f"{n} = dronerl_ros.{n}:main" for n in NODES]},
)
