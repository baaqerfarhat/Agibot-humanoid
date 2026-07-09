#!/usr/bin/env python3
"""Camera -> HTTP streamer that runs ON THE ROBOT.

Subscribes to the head Orbbec RGB topic, rotates 180 deg (camera is mounted
inverted), JPEG-encodes, and serves it over HTTP on the wired link so the
laptop can pull frames and run a VLM locally.

Endpoints:
  http://<robot-ip>:8099/frame.jpg   -> latest single JPEG
  http://<robot-ip>:8099/stream.mjpg -> multipart MJPEG stream
  http://<robot-ip>:8099/            -> viewer page

Run on robot with ROS sourced and the AGI DDS profile:
  export FASTRTPS_DEFAULT_PROFILES_FILE=/agibot/software/entry/cfg/ros_dds_configuration.xml
  python3 camera_stream_server.py
"""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

RAW_TOPIC = "/aima/hal/sensor/rgbd_head_front/rgb_image"
COMP_TOPIC = "/aima/hal/sensor/rgbd_head_front/rgb_image/compressed"
PORT = 8099
ROTATE_180 = True
JPEG_QUALITY = 80

_lock = threading.Lock()
_latest_jpeg = None
_frame_count = 0
_last_stamp = 0.0


def _decode_raw(msg):
    h, w, enc = msg.height, msg.width, msg.encoding.lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if enc == "rgb8":
        img = buf.reshape(h, w, 3)[:, :, ::-1]
    elif enc == "bgr8":
        img = buf.reshape(h, w, 3)
    elif enc in ("yuv422_yuy2", "yuyv", "yuv422"):
        img = cv2.cvtColor(buf.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
    elif enc == "mono8":
        img = cv2.cvtColor(buf.reshape(h, w), cv2.COLOR_GRAY2BGR)
    else:
        img = buf.reshape(h, w, 3)
    return np.ascontiguousarray(img)


def _store(img):
    global _latest_jpeg, _frame_count, _last_stamp
    if ROTATE_180:
        img = cv2.rotate(img, cv2.ROTATE_180)
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if ok:
        with _lock:
            _latest_jpeg = jpg.tobytes()
            _frame_count += 1
            _last_stamp = time.time()


class CamNode(Node):
    def __init__(self):
        super().__init__("camera_stream_server")
        qos = QoSProfile(depth=2)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.history = HistoryPolicy.KEEP_LAST
        qos.durability = DurabilityPolicy.VOLATILE
        self.create_subscription(Image, RAW_TOPIC, self.raw_cb, qos)
        self.create_subscription(CompressedImage, COMP_TOPIC, self.comp_cb, qos)
        self._got_raw = False

    def raw_cb(self, msg):
        self._got_raw = True
        try:
            _store(_decode_raw(msg))
        except Exception as e:
            self.get_logger().warn(f"raw decode failed: {e}")

    def comp_cb(self, msg):
        if self._got_raw:
            return  # prefer raw
        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            _store(img)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/frame.jpg"):
            with _lock:
                data = _latest_jpeg
            if data is None:
                self.send_error(503, "no frame yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        elif self.path.startswith("/stream.mjpg"):
            self.send_response(200)
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with _lock:
                        data = _latest_jpeg
                    if data is not None:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(data)}\r\n\r\n".encode())
                        self.wfile.write(data)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path.startswith("/status"):
            with _lock:
                fc, ts = _frame_count, _last_stamp
            age = time.time() - ts if ts else -1
            body = f'{{"frames": {fc}, "age_s": {age:.2f}}}'.encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            html = (b"<html><body style='margin:0;background:#111'>"
                    b"<img src='/stream.mjpg' style='width:100%'></body></html>")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html)


def main():
    rclpy.init()
    node = CamNode()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[camera_stream_server] serving on 0.0.0.0:{PORT} "
          f"(/frame.jpg, /stream.mjpg, /status)")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
