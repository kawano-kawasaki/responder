import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
import numpy as np
import os
import open3d as o3d
from datetime import datetime
import json
from collections import defaultdict, deque
import threading
import queue

class LidarFiveFrameSaver(Node):
    def __init__(self):
        super().__init__('lidar_fiveframe_saver')

        # --- パス設定 ここから ---
        self.ABSOLUTE_BASE_PATH = "/root/avia_data"
        dt = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        self.session_output_dir = os.path.join(self.ABSOLUTE_BASE_PATH, f'lidar_data_{dt}')
        self.bin_output_dir = os.path.join(self.session_output_dir, 'bin_data')
        self.txt_output_dir = os.path.join(self.session_output_dir, 'txt_data')

        os.makedirs(self.bin_output_dir, exist_ok=True)
        os.makedirs(self.txt_output_dir, exist_ok=True)

        self.get_logger().info(f'Base directory "{self.session_output_dir}" created.')
        self.get_logger().info(f'BIN files will be saved in "{self.bin_output_dir}".')
        self.get_logger().info(f'TXT metadata files will be saved in "{self.txt_output_dir}".')
        self.get_logger().info('======================================================================')
        # --- パス設定 ここまで ---

        self.subscription = self.create_subscription(
            PointCloud2,
            '/livox/lidar',
            self.listener_callback,
            10
        )
        
        # フレームバッファ (最新10フレームを保持)
        self.frame_buffer = deque(maxlen=10)
        self.timestamp_buffer = deque(maxlen=10)
        self.sequence_id = 0
        self.voxel_size = 0.05

        # 保存用キューとワーカースレッド（複数スレッドで I/O 並列化）
        self.save_queue = queue.Queue()
        self.worker_threads = []
        for _ in range(3):  # 3スレッドに増やす
            t = threading.Thread(target=self._save_worker, daemon=True)
            t.start()
            self.worker_threads.append(t)

        # タイマー：0.2秒ごとにキューにジョブ追加
        self.create_timer(0.2, self.timer_callback)

    def listener_callback(self, msg: PointCloud2):
        points = self.extract_xyzi(msg)
        self.frame_buffer.append(points)
        self.timestamp_buffer.append(msg.header.stamp)

    def timer_callback(self):
        if len(self.frame_buffer) < 5:
            return
        latest_frames = list(self.frame_buffer)[-5:]
        latest_timestamps = list(self.timestamp_buffer)[-5:]
        # 保存用キューに追加
        self.save_queue.put((latest_frames, latest_timestamps))

    def _save_worker(self):
        while True:
            frames, timestamps = self.save_queue.get()
            with threading.Lock():  # sequence_id 更新をスレッド安全に
                sequence_id = self.sequence_id
                self.sequence_id += 1
            # self.get_logger().info(f"Saving seq {sequence_id} started")  # 開始ログ
            combined_points = np.vstack(frames)
            filtered = self.filter_frames(combined_points)
            timestamp = timestamps[0]  # 最も古いフレームのタイムスタンプ
            self.save_all_files(filtered, timestamp, sequence_id)
            # self.get_logger().info(f"Saving seq {sequence_id} finished")  # 終了ログ

    def save_all_files(self, points, timestamp, sequence_id):
        self.save_bin(points, sequence_id)
        self.save_metadata(points, timestamp, sequence_id)

    def extract_xyzi(self, pc2_msg: PointCloud2) -> np.ndarray:
        pts = []
        for x, y, z, i in read_points(pc2_msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
            pts.append([x, y, z, i])
        return np.array(pts, dtype=np.float32)

    def filter_frames(self, combined_pts: np.ndarray) -> np.ndarray:
        # さらに高速化：NumPy ベクトル演算、unique 集約
        voxel_indices = np.floor(combined_pts[:, :3] / self.voxel_size).astype(np.int32)
        keys, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)

        # 各ボクセルごとに平均を取る
        sums = np.zeros((keys.shape[0], 4), dtype=np.float64)
        np.add.at(sums, inverse, combined_pts)
        counts = np.bincount(inverse)
        means = sums / counts[:, None]

        return means.astype(np.float32)

    def save_bin(self, points: np.ndarray, sequence_id: int):
        try:
            filename = os.path.join(self.bin_output_dir, f"{sequence_id:06d}.bin")
            points.astype(np.float32).tofile(filename)
            self.get_logger().info(f"Saved BIN: {filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to save BIN: {e}")

    def save_metadata(self, points: np.ndarray, timestamp, sequence_id: int):
        try:
            filename = os.path.join(self.txt_output_dir, f"{sequence_id:06d}.txt")
            metadata = {
                "file_name_bin": f"{sequence_id:06d}.bin",
                "sequence_id": sequence_id,
                "num_points": points.shape[0],
                "timestamp_sec": timestamp.sec,
                "timestamp_nanosec": timestamp.nanosec
            }
            with open(filename, 'w') as f:
                json.dump(metadata, f, indent=4)
        except Exception as e:
            self.get_logger().error(f"Failed to save metadata: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = LidarFiveFrameSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
