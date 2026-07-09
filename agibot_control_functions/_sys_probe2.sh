#!/usr/bin/env bash
echo "############## POWER MODE ##############"
(sudo -n nvpmodel -q 2>/dev/null) || (nvpmodel -q 2>/dev/null) || echo "nvpmodel needs sudo"
cat /etc/nvpmodel.conf 2>/dev/null | grep -E "^< PM_CONFIG|POWER_MODEL ID" | head -20

echo "############## DOCKER ##############"
(docker --version 2>/dev/null) || echo "docker: none"
(groups 2>/dev/null | grep -q docker && echo "user in docker group") || echo "user NOT in docker group"
(docker ps 2>/dev/null | head -5) || echo "cannot run docker without sudo"

echo "############## INTERNET ##############"
(timeout 5 curl -sI https://huggingface.co 2>/dev/null | head -1) || echo "no https to huggingface"
(timeout 5 curl -sI https://github.com 2>/dev/null | head -1) || echo "no https to github"
(timeout 5 curl -sI https://pypi.org 2>/dev/null | head -1) || echo "no https to pypi"

echo "############## ONNXRUNTIME PROVIDERS ##############"
python3 -c "import onnxruntime as o; print('  providers:', o.get_available_providers())" 2>/dev/null || echo "  ort import failed"

echo "############## PIP: torch/vision availability ##############"
pip3 list 2>/dev/null | grep -iE "torch|pillow|transformers|ultralytics|timm|clip" || echo "  none of torch/transformers/ultralytics installed"

echo "############## CAMERA TOPIC DETAIL ##############"
source /opt/ros/humble/setup.bash 2>/dev/null
source ~/aimdk_ws/install/setup.bash 2>/dev/null
echo "-- rgb_image type --"
ros2 topic info /aima/hal/sensor/rgb_head_front_center/rgb_image 2>/dev/null
echo "-- compressed type --"
ros2 topic info /aima/hal/sensor/rgb_head_front_center/rgb_image/compressed 2>/dev/null
echo "-- camera_info (1 msg) --"
timeout 5 ros2 topic echo --once /aima/hal/sensor/rgb_head_front_center/camera_info 2>/dev/null | head -30
echo "-- rgb_image publish rate (5s) --"
timeout 7 ros2 topic hz /aima/hal/sensor/rgb_head_front_center/rgb_image 2>/dev/null | head -5
echo "DONE"
