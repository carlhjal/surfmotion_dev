#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from tf2_ros import TransformListener, Buffer
from rclpy.duration import Duration
from rclpy.time import Time
from geometry_msgs.msg import TransformStamped
from ament_index_python.packages import get_package_share_directory
from datetime import datetime
from pathlib import Path
import csv


class EEPoseLogger(Node):
    def __init__(self):
        super().__init__('ee_pose_logger')

        # TF Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Frame IDs
        self.base_frame = 'ur20_base_link'
        self.ee_frame = 'tool_endpoint'

        self.log_enabled = False
        self.timer_period = 0.01  # 100 Hz
        self.timer = self.create_timer(self.timer_period, self.timer_callback)

        # Subscriber to start trigger
        self.create_subscription(Bool, '/start_logging', self.trigger_logging_callback, 10)

        # Output setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(get_package_share_directory("eval_tools")) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = output_dir / f"eef_transform_{timestamp}.csv"
        self.csvfile = open(self.filepath, 'w', newline='')
        self.writer = csv.writer(self.csvfile)
        self.writer.writerow(['time', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])

        self.get_logger().info(f"Waiting for /start_logging trigger...")
        self.get_logger().info(f"Logging EEF transform to: {self.filepath}")

    def trigger_logging_callback(self, msg: Bool):
        if msg.data:
            if not self.log_enabled:
                self.log_enabled = True
                self.get_logger().info("Received /start_logging trigger — logging enabled.")
            else:
                self.destroy_node()
                rclpy.shutdown()

    def timer_callback(self):
        if not self.log_enabled:
            return

        try:
            now = Time()
            trans: TransformStamped = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                now,
                timeout=Duration(seconds=0.1)
            )

            t = trans.transform.translation
            r = trans.transform.rotation
            timestamp = trans.header.stamp.sec + trans.header.stamp.nanosec * 1e-9

            self.writer.writerow([
                timestamp,
                t.x, t.y, t.z,
                r.x, r.y, r.z, r.w
            ])
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")

    def destroy_node(self):
        self.csvfile.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = EEPoseLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted, exiting.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
