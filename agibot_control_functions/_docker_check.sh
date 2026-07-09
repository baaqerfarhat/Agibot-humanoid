#!/usr/bin/env bash
S="echo 1 | sudo -S"
echo "=== docker info: runtimes ==="
echo 1 | sudo -S docker info 2>/dev/null | grep -iE "Runtimes|Default Runtime|Server Version"
echo "=== nvidia-container-toolkit packages ==="
dpkg -l 2>/dev/null | grep -iE "nvidia-container|nvidia-docker" | awk '{print $2, $3}'
echo "=== /etc/docker/daemon.json ==="
cat /etc/docker/daemon.json 2>/dev/null || echo "(no daemon.json)"
echo "=== existing docker images ==="
echo 1 | sudo -S docker images 2>/dev/null | head -20
echo "=== nvidia runtime binary ==="
which nvidia-container-runtime nvidia-ctk 2>/dev/null || echo "(nvidia-container-runtime not on PATH)"
