#!/usr/bin/env bash
# Probe the humanoid's onboard computer for VLM feasibility + camera topics.

echo "############## MODEL / L4T ##############"
cat /proc/device-tree/model 2>/dev/null; echo
cat /etc/nv_tegra_release 2>/dev/null
head -1 /etc/os-release 2>/dev/null

echo "############## CPU ##############"
nproc
grep -m1 "model name" /proc/cpuinfo 2>/dev/null
lscpu 2>/dev/null | grep -E "Architecture|CPU\(s\)|Model name" | head -5

echo "############## MEMORY ##############"
free -h

echo "############## DISK (root) ##############"
df -h / 2>/dev/null | tail -2

echo "############## CUDA / TENSORRT ##############"
ls -d /usr/local/cuda* 2>/dev/null
(nvcc --version 2>/dev/null | grep release) || echo "nvcc: not on PATH"
dpkg -l 2>/dev/null | grep -iE "tensorrt|nvinfer|cudnn" | awk '{print $2, $3}' | head -20

echo "############## GPU LIVE (tegrastats 2 samples) ##############"
(timeout 3 tegrastats 2>/dev/null | head -2) || echo "tegrastats unavailable"

echo "############## PYTHON ML STACK ##############"
python3 --version
for pkg in torch torchvision torchaudio onnxruntime onnxruntime_gpu tensorrt transformers accelerate opencv-python cv2 numpy pillow ultralytics llama_cpp; do
  python3 -c "import ${pkg} as m; print('  ${pkg}:', getattr(m,'__version__','ok'))" 2>/dev/null
done
echo "-- torch CUDA? --"
python3 -c "import torch; print('  torch', torch.__version__, 'cuda_avail', torch.cuda.is_available(), 'dev', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'))" 2>/dev/null || echo "  torch not importable"

echo "############## CAMERA ROS TOPICS ##############"
source /opt/ros/humble/setup.bash 2>/dev/null
source ~/aimdk_ws/install/setup.bash 2>/dev/null
ros2 topic list 2>/dev/null | grep -iE "cam|image|color|depth|rgb|orbbec|sensor" | sort

echo "############## running camera-ish nodes ##############"
ros2 node list 2>/dev/null | grep -iE "cam|orbbec|image" | sort

echo "DONE"
