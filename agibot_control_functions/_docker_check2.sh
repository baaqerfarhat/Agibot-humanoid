#!/usr/bin/env bash
echo "=== docker daemon works? ==="
echo 1 | sudo -S docker ps 2>&1 | head -5
echo "=== docker version ==="
echo 1 | sudo -S docker version --format '{{.Server.Version}}' 2>/dev/null
echo "=== apt policy nvidia-container-toolkit ==="
apt-cache policy nvidia-container-toolkit 2>/dev/null | head -8
echo "=== nvidia apt sources ==="
ls /etc/apt/sources.list.d/ 2>/dev/null | grep -iE "nvidia|libnvidia|l4t"
echo "=== ollama already present? ==="
which ollama 2>/dev/null || echo "(ollama not installed)"
