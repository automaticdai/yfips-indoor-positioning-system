"""Optional ROS 2 publisher. No-op if rclpy is not installed."""

from __future__ import annotations

import json
from typing import Any


class RosPublisher:
    def __init__(self, ros_cfg: dict[str, Any]) -> None:
        self.enabled = bool(ros_cfg.get("enabled", False))
        self.topic = ros_cfg.get("topic", "/yfips/detections")
        self.node: Any = None
        self.pub: Any = None
        self._rclpy: Any = None
        if not self.enabled:
            return
        try:
            import rclpy
            from rclpy.node import Node
            from std_msgs.msg import String
        except ImportError:
            print("[ros] rclpy not installed; ROS publishing disabled")
            self.enabled = False
            return

        rclpy.init(args=None)
        self._rclpy = rclpy
        self.node = Node("yfips_publisher")
        self.pub = self.node.create_publisher(String, self.topic, 10)
        self._String = String
        print(f"[ros] publishing to {self.topic}")

    def send(self, payload: dict[str, Any]) -> None:
        if not self.enabled or self.pub is None:
            return
        msg = self._String()
        msg.data = json.dumps(payload)
        self.pub.publish(msg)

    def shutdown(self) -> None:
        if self.node is not None:
            self.node.destroy_node()
        if self._rclpy is not None:
            self._rclpy.shutdown()
