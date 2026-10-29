#!/usr/bin/env bash
# Starts the M4 container (plan/M4.md) with the GPU, the host network and the repository in /work.
# HEADLESS=1 runs without the X display; the arguments are the command, bash by default.
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
args=(--rm --gpus all --network host --ipc host -v "$repo":/work)
if [[ -z "${HEADLESS:-}" ]]; then
    args+=(-e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro)
    if [[ -n "${XAUTHORITY:-}" ]]; then
        args+=(-e XAUTHORITY=/tmp/.Xauthority -v "$XAUTHORITY":/tmp/.Xauthority:ro)
    fi
fi
if [[ -t 0 ]]; then
    args+=(-it)
fi
exec docker run "${args[@]}" dronerl-m4 "${@:-bash}"
