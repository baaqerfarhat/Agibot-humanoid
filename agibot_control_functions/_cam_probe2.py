import sys, time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

TOPIC = sys.argv[1] if len(sys.argv) > 1 else "/aima/hal/sensor/rgbd_head_front/rgb_image"

class Probe(Node):
    def __init__(self):
        super().__init__("cam_probe2")
        qos = QoSProfile(depth=5)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.history = HistoryPolicy.KEEP_LAST
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.sub = self.create_subscription(Image, TOPIC, self.cb, qos)
        self.t = []
        self.info = None

    def cb(self, msg):
        self.t.append(time.time())
        if self.info is None:
            self.info = f"{msg.width}x{msg.height} enc={msg.encoding} bytes={len(msg.data)}"

def main():
    rclpy.init()
    n = Probe()
    t0 = time.time()
    printed = False
    while time.time() - t0 < 12.0:
        rclpy.spin_once(n, timeout_sec=0.05)
        if not printed and time.time() - t0 > 2.0:
            pubs = n.get_publishers_info_by_topic(TOPIC)
            print(f"MATCHED PUBLISHERS: {len(pubs)}")
            for p in pubs:
                print(f"  node={p.node_name} rel={p.qos_profile.reliability} dur={p.qos_profile.durability}")
            printed = True
    if n.info:
        print(f"INFO: {n.info}")
    if len(n.t) > 1:
        hz = (len(n.t) - 1) / (n.t[-1] - n.t[0])
        print(f"FRAMES: {len(n.t)}  {hz:.1f} Hz")
    else:
        print(f"FRAMES: {len(n.t)}")
    n.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
