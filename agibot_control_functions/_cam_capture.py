import time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

RAW_TOPICS = [
    ("/aima/hal/sensor/rgbd_head_front/rgb_image", "rgbd_front"),
    ("/aima/hal/sensor/rgb_head_front_center/rgb_image", "rgb_center"),
]
COMP_TOPICS = [
    ("/aima/hal/sensor/rgbd_head_front/rgb_image/compressed", "rgbd_front_c"),
    ("/aima/hal/sensor/rgb_head_front_center/rgb_image/compressed", "rgb_center_c"),
]
OUT = "/tmp/cam"

def decode_raw(msg):
    h, w, enc = msg.height, msg.width, msg.encoding.lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if enc in ("rgb8",):
        img = buf.reshape(h, w, 3)[:, :, ::-1]
    elif enc in ("bgr8",):
        img = buf.reshape(h, w, 3)
    elif enc in ("mono8",):
        img = buf.reshape(h, w)
    elif enc in ("yuv422_yuy2", "yuyv", "yuv422"):
        img = cv2.cvtColor(buf.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUYV)
    elif enc in ("nv12",):
        img = cv2.cvtColor(buf.reshape(int(h * 3 / 2), w), cv2.COLOR_YUV2BGR_NV12)
    else:
        # last resort: assume 3-channel
        try:
            img = buf.reshape(h, w, 3)
        except Exception:
            return None, enc
    return np.ascontiguousarray(img), enc

class Cap(Node):
    def __init__(self):
        super().__init__("cam_capture")
        qos = QoSProfile(depth=4)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        qos.history = HistoryPolicy.KEEP_LAST
        qos.durability = DurabilityPolicy.VOLATILE
        self.saved = {}
        for topic, tag in RAW_TOPICS:
            self.create_subscription(Image, topic,
                lambda m, t=tag: self.raw_cb(m, t), qos)
        for topic, tag in COMP_TOPICS:
            self.create_subscription(CompressedImage, topic,
                lambda m, t=tag: self.comp_cb(m, t), qos)

    def raw_cb(self, msg, tag):
        if tag in self.saved:
            return
        img, enc = decode_raw(msg)
        if img is None:
            print(f"[{tag}] got frame but could not decode enc={enc}")
            self.saved[tag] = "decode_fail"
            return
        path = f"{OUT}_{tag}.jpg"
        cv2.imwrite(path, img)
        self.saved[tag] = path
        print(f"[{tag}] SAVED {path}  {msg.width}x{msg.height} enc={enc}")

    def comp_cb(self, msg, tag):
        if tag in self.saved:
            return
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[{tag}] got compressed but decode failed fmt={msg.format}")
            self.saved[tag] = "decode_fail"
            return
        path = f"{OUT}_{tag}.jpg"
        cv2.imwrite(path, img)
        self.saved[tag] = path
        print(f"[{tag}] SAVED {path}  {img.shape[1]}x{img.shape[0]} fmt={msg.format}")

def main():
    rclpy.init()
    n = Cap()
    t0 = time.time()
    want = {t for _, t in RAW_TOPICS} | {t for _, t in COMP_TOPICS}
    while time.time() - t0 < 25.0:
        rclpy.spin_once(n, timeout_sec=0.1)
        if set(n.saved.keys()) >= want:
            break
    print("=== RESULT ===")
    for _, tag in RAW_TOPICS + COMP_TOPICS:
        print(f"  {tag}: {n.saved.get(tag, 'NO DATA')}")
    n.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
