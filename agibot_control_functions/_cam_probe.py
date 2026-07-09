import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

RAW = "/aima/hal/sensor/rgb_head_front_center/rgb_image"
COMP = "/aima/hal/sensor/rgb_head_front_center/rgb_image/compressed"

class Probe(Node):
    def __init__(self):
        super().__init__("cam_probe")
        self.counts = {}
        self.info = {}
        for rel in (ReliabilityPolicy.RELIABLE, ReliabilityPolicy.BEST_EFFORT):
            qos = QoSProfile(depth=5)
            qos.reliability = rel
            qos.history = HistoryPolicy.KEEP_LAST
            qos.durability = DurabilityPolicy.VOLATILE
            tag_raw = f"RAW/{rel.name}"
            tag_comp = f"COMP/{rel.name}"
            self.counts[tag_raw] = []
            self.counts[tag_comp] = []
            self.create_subscription(Image, RAW,
                lambda m, t=tag_raw: self.cb(m, t, True), qos)
            self.create_subscription(CompressedImage, COMP,
                lambda m, t=tag_comp: self.cb(m, t, False), qos)

    def cb(self, msg, tag, is_raw):
        self.counts[tag].append(time.time())
        if tag not in self.info:
            if is_raw:
                self.info[tag] = f"{msg.width}x{msg.height} enc={msg.encoding} bytes={len(msg.data)}"
            else:
                self.info[tag] = f"compressed fmt={msg.format} bytes={len(msg.data)}"

def main():
    rclpy.init()
    n = Probe()
    t0 = time.time()
    while time.time() - t0 < 8.0:
        rclpy.spin_once(n, timeout_sec=0.05)
    for tag, ts in n.counts.items():
        if len(ts) > 1:
            hz = (len(ts) - 1) / (ts[-1] - ts[0])
            print(f"{tag}: {len(ts)} frames  {hz:.1f} Hz  [{n.info.get(tag,'')}]")
        elif len(ts) == 1:
            print(f"{tag}: 1 frame  [{n.info.get(tag,'')}]")
        else:
            print(f"{tag}: 0 frames")
    n.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
