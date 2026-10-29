#!/usr/bin/env bash
# Host setup for M4 (plan/M4.md, M4.1): Docker Engine with GPU access through the NVIDIA Container
# Toolkit. The Docker and CUDA apt repositories are already configured on this machine.
# Run as: sudo bash docker/install_host.sh
set -euo pipefail

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin \
    docker-compose-plugin nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl enable docker
systemctl restart docker
usermod -aG docker "${SUDO_USER:-$USER}"

# GPU check; the M4 image starts from ros:jazzy, CUDA for ONNX Runtime comes with its pip wheels
docker run --rm --gpus all nvidia/cuda:13.0.2-base-ubuntu24.04 nvidia-smi
docker pull ros:jazzy-ros-base-noble
echo "Docker with GPU works. Log out and in again to use docker without sudo."
