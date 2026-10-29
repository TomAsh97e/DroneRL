# M4 simulation stack (plan/M4.md): ROS 2 Jazzy, Gazebo Harmonic, PX4 SITL build tools, Micro
# XRCE-DDS Agent, px4_msgs and the Python side of dronerl. The repository is mounted at /work and
# PX4 from external/PX4-Autopilot is built there, so its build survives the container.
#
# Build from the repository root: docker build -f docker/Dockerfile -t dronerl-m4 .
FROM ros:jazzy-ros-base-noble

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Gazebo Harmonic from the OSRF repository, as PX4 Tools/setup/ubuntu.sh installs it
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
        -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg]" \
        "http://packages.osrfoundation.org/gazebo/ubuntu-stable noble main" \
        > /etc/apt/sources.list.d/gazebo-stable.list \
    && rm -rf /var/lib/apt/lists/*

# PX4 SITL build dependencies (ubuntu.sh without NuttX), Gazebo, ROS 2 bridges and bag storage,
# OpenGL libraries the NVIDIA runtime plugs its driver into
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ccache cmake file g++ gcc gdb git libssl-dev libxml2-dev libxml2-utils \
        make ninja-build python3-dev python3-pip python3-setuptools python3-wheel python3-numpy \
        rsync unzip zip bc \
        gz-harmonic libunwind-dev cppzmq-dev libeigen3-dev libopencv-dev protobuf-compiler \
        gstreamer1.0-plugins-bad gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
        gstreamer1.0-libav libgstreamer-plugins-base1.0-dev pkg-config \
        ros-jazzy-ros-gz-bridge ros-jazzy-ros-gz-image ros-jazzy-cv-bridge \
        ros-jazzy-rosbag2-storage-mcap python3-opencv python3-colcon-common-extensions \
        libgl1 libegl1 libglx0 libglvnd0 mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# Python: PX4 build tools, ONNX Runtime with CUDA from its wheels, torch on the CPU for the
# contract and the image reduction, gymnasium that dronerl.envs registers with (NavEnv in the
# checks). numpy stays the apt one that cv_bridge is built against.
COPY external/PX4-Autopilot/Tools/setup/requirements.txt /tmp/px4-requirements.txt
RUN python3 -c "import numpy; print(f'numpy=={numpy.__version__}')" > /tmp/constraints.txt \
    && pip3 install --no-cache-dir --break-system-packages -c /tmp/constraints.txt \
        -r /tmp/px4-requirements.txt "onnxruntime-gpu[cuda,cudnn]==1.31.1" gymnasium==1.4.0 \
    && pip3 install --no-cache-dir --break-system-packages -c /tmp/constraints.txt \
        --index-url https://download.pytorch.org/whl/cpu torch==2.14.1

# Micro XRCE-DDS Agent with its own Fast DDS in /usr/local, run without the ROS environment
COPY external/Micro-XRCE-DDS-Agent /tmp/agent
RUN cmake -S /tmp/agent -B /tmp/agent/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/agent/build -j"$(nproc)" \
    && cmake --install /tmp/agent/build && ldconfig && rm -rf /tmp/agent

# px4_msgs matching the PX4 release
COPY external/px4_msgs /opt/px4_ws/src/px4_msgs
RUN source /opt/ros/jazzy/setup.bash \
    && colcon --log-base /tmp/colcon-log build --base-paths /opt/px4_ws/src \
        --build-base /tmp/px4_build --install-base /opt/px4_ws/install \
    && rm -rf /tmp/px4_build /tmp/colcon-log

COPY docker/entrypoint.sh /entrypoint.sh
ENV NVIDIA_DRIVER_CAPABILITIES=all PYTHONPATH=/work
# uid 1000 of the image is the host user, so files built in /work keep their owner
USER ubuntu
WORKDIR /work
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
