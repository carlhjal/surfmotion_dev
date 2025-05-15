#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import csv
from datetime import datetime
import os
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Bool
from tf2_ros import TransformListener, Buffer
from rclpy.duration import Duration
from rclpy.time import Time
from geometry_msgs.msg import TransformStamped

def destroy_node(node: Node):
    node.destroy_node()
    rclpy.shutdown() 

class CombinedLogger(Node):
    def __init__(self):
        super().__init__('joint_state_logger')
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )
        
        self.subscription = self.create_subscription(
            Bool,
            '/start_logging',
            self.trigger_logging_callback,
            10
        )

        # TF Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Frame IDs
        self.base_frame = 'iiwa_base'
        self.ee_frame = 'tool_endpoint'
        self.timer_period = 0.01  # 100 Hz
        self.timer = self.create_timer(self.timer_period, self.eef_timer_callback)

        self.log_enabled = False        
        
        # Output file setup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(get_package_share_directory("eval_tools")) / "output"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Joint file
        self.joint_file = out_dir / f"joint_states_{timestamp}.csv"
        self.joint_csv = open(self.joint_file, 'w', newline='')
        self.joint_writer = csv.writer(self.joint_csv)
        self.joint_writer.writerow(['time', 'joint', 'position'])

        # EEF file
        self.eef_file = out_dir / f"eef_transform_{timestamp}.csv"
        self.eef_csv = open(self.eef_file, 'w', newline='')
        self.eef_writer = csv.writer(self.eef_csv)
        self.eef_writer.writerow(['time', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        

        self.get_logger().info("Waiting for publisher on /start_logging...")
        
        while self.count_publishers('/start_logging') == 0 and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
        
        self.get_logger().info(f"Waiting for /start_logging trigger...")
        self.get_logger().info(f"Starting logging data...")

    def trigger_logging_callback(self, msg: Bool):
        if msg.data is True:
            self.log_enabled = True
            self.get_logger().info("Received /start_logging trigger — logging enabled.")
        else:
            self.get_logger().info("Received stop trigger — exiting.")
            self.destroy_node()
            rclpy.shutdown()

    def joint_callback(self, msg: JointState):
        if not self.log_enabled:
            return

        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        for name, pos in zip(msg.name, msg.position):
            self.joint_writer.writerow([timestamp, name, pos])

    def eef_timer_callback(self):
        if not self.log_enabled:
            return

        try:
            now = Time()
            tf: TransformStamped = self.tf_buffer.lookup_transform(
                self.base_frame, self.ee_frame, now, timeout=Duration(seconds=0.1)
            )

            t = tf.transform.translation
            r = tf.transform.rotation
            timestamp = tf.header.stamp.sec + tf.header.stamp.nanosec * 1e-9

            self.eef_writer.writerow([timestamp, t.x, t.y, t.z, r.x, r.y, r.z, r.w])
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            
    def destroy_node(self):
        self.joint_csv.close()
        self.eef_csv.close()
        super().destroy_node()
        
def main(args=None):
    rclpy.init(args=args)
    node = CombinedLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted, exiting.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
