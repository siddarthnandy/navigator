#!/usr/bin/env python3
"""
hybrid_drivable_grid_node.py

Publishes /grid/drivable directly from the perception-based drivable grid.

PERCEPTION-ONLY: the MapManager/HD-map fallback has been removed on request.
Previously this node started from the HD map as a base layer and only let
perception OPEN cells (never close them), falling back to HD map wherever
perception was uncertain. That meant /grid/drivable was effectively
MapManager-driven everywhere perception hadn't directly confirmed a cell.
Now it's a straight passthrough of /grid/drivable/segmented (resized to
300x300 if needed) — no HD map subscription, no fallback.

Inputs:
  /grid/drivable/segmented     PerceptionDrivableGridNode  (only source)

Output:
  /grid/drivable               300x300 OccupancyGrid, 0.2m/cell, base_link
"""

import threading
import numpy as np
import rclpy
from rclpy.node    import Node
from nav_msgs.msg  import OccupancyGrid

# Perception occupancy BELOW this threshold → high-confidence road, open to 0.
PERC_ROAD_THRESHOLD = np.int8(30)


class HybridDrivableGridNode(Node):

    def __init__(self):
        super().__init__('hybrid_drivable_grid_node')
        self._lock = threading.Lock()

        self._perc: np.ndarray | None = None   # /grid/drivable/segmented

        qos_be = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.QoSReliabilityPolicy.BEST_EFFORT,
            history=rclpy.qos.QoSHistoryPolicy.KEEP_LAST, depth=1)

        self.create_subscription(OccupancyGrid, '/grid/drivable/segmented',
            self._cb_perc, qos_be)

        self.pub = self.create_publisher(OccupancyGrid, '/grid/drivable', 10)

        # 20-Hz (matches route_costmap_node, which reads this for goal selection)
        self.create_timer(0.05, self._publish_loop)

        self.get_logger().info('HybridDrivableGridNode ready — perception-only (MapManager/HD-map fallback removed).')

    # -------------------------------------------------------------------------

    def _cb_perc(self, msg: OccupancyGrid):
        arr = np.array(msg.data, dtype=np.int8).reshape(
            msg.info.height, msg.info.width)
        with self._lock:
            self._perc = arr

    # -------------------------------------------------------------------------

    def _publish_loop(self):
        with self._lock:
            perc = self._perc.copy() if self._perc is not None else None

        if perc is None:
            self.get_logger().warn(
                'No perception data yet.', throttle_duration_sec=5.0)
            return

        # ── helper: resize any grid to 300×300 ───────────────────────────────
        target_shape = (300, 300)

        def _resize(arr):
            if arr.shape == target_shape:
                return arr
            import cv2
            return cv2.resize(arr.astype(np.float32),
                              (target_shape[1], target_shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(np.int8)

        perc = _resize(perc)

        # High-confidence perception road → open to 0. Everything else passes
        # through the perception grid's own values unchanged (no HD map base).
        output = perc.copy()
        output[perc < PERC_ROAD_THRESHOLD] = np.int8(0)

        # NOTE: No occupancy veto here.
        # /grid/occupancy/current already flows into grid_summation_node as its
        # own independent channel and is combined via np.maximum with steering cost.
        # Applying it again here would double-count obstacle evidence and incorrectly
        # mark road cells as 100, blocking the path planner from finding any route.

        # ── publish ────────────────────────────────────────────────────────────
        msg                           = OccupancyGrid()
        msg.header.stamp              = self.get_clock().now().to_msg()
        msg.header.frame_id           = 'base_link'
        msg.info.resolution           = 0.2
        msg.info.width                = target_shape[1]
        msg.info.height               = target_shape[0]
        msg.info.origin.position.x    = -20.0
        msg.info.origin.position.y    = -30.0
        msg.info.origin.position.z    = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data                      = output.ravel().tolist()

        try:
            self.pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f'Publish failed: {e}',
                                   throttle_duration_sec=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = HybridDrivableGridNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
