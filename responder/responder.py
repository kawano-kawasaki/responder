import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
import numpy as np
import os
import open3d as o3d
from datetime import datetime
import json
from collections import defaultdict

class LidarTwoFrameSaver(Node):
    def __init__(self):
        super().__init__('lidar_twoframe_saver')

        # --- パス設定 ここから ---
        # 保存するデータのベースディレクトリの絶対パスを指定
        self.ABSOLUTE_BASE_PATH = "/tmp"
        
        # 日付と時刻に基づくユニークなディレクトリ名を生成
        dt = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        self.session_output_dir = os.path.join(self.ABSOLUTE_BASE_PATH, f'lidar_data_{dt}')
        
        # 各ファイルタイプごとのサブディレクトリ
        self.pcd_output_dir = os.path.join(self.session_output_dir, 'pcd_data')
        self.bin_output_dir = os.path.join(self.session_output_dir, 'bin_data')
        self.txt_output_dir = os.path.join(self.session_output_dir, 'txt_data')

        # ディレクトリの作成
        os.makedirs(self.pcd_output_dir, exist_ok=True)
        os.makedirs(self.bin_output_dir, exist_ok=True)
        os.makedirs(self.txt_output_dir, exist_ok=True)

        self.get_logger().info(f'Base directory "{self.session_output_dir}" created.')
        self.get_logger().info(f'PCD files will be saved in "{self.pcd_output_dir}".')
        self.get_logger().info(f'BIN files will be saved in "{self.bin_output_dir}".')
        self.get_logger().info(f'TXT metadata files will be saved in "{self.txt_output_dir}".')
        # --- パス設定 ここまで ---

        self.subscription = self.create_subscription(
            PointCloud2,
            '/livox/lidar',
            self.listener_callback,
            10
        )
        
        # フレームバッファ
        self.prev_points = None
        self.frame_counter = 0
        self.sequence_id = 0
        # ボクセル設定
        self.voxel_size = 0.05

    def listener_callback(self, msg: PointCloud2):
        # ポイント抽出
        points = self.extract_xyzi(msg)
        self.frame_counter += 1

        # 奇数フレームはバッファに保存
        if self.frame_counter % 2 == 1:
            self.prev_points = points
            return

        # 偶数フレーム: 直前のフレームと合わせて処理
        if self.prev_points is None:
            # 万一prevがない場合はスキップ
            self.prev_points = points
            return

        # 2つのフレームを結合し、ボクセル単位で重心を計算
        combined_points = np.vstack((self.prev_points, points)) # 2フレームの点を結合
        filtered = self.filter_two_frames(combined_points) # 結合した点を渡す
        
        # 3種類のファイルを保存
        self.save_pcd(filtered)
        self.save_bin(filtered)
        self.save_metadata(filtered, msg.header.stamp)
        
        self.sequence_id += 1
        # 次のバッファをクリア or 更新
        self.prev_points = None

    def extract_xyzi(self, pc2_msg: PointCloud2) -> np.ndarray:
        """PointCloud2から[x, y, z, intensity]配列を抽出"""
        pts = []
        for x, y, z, i in read_points(pc2_msg, field_names=("x", "y", "z", "intensity"), skip_nans=True):
            pts.append([x, y, z, i])
        return np.array(pts, dtype=np.float32)

    def filter_two_frames(self, combined_pts: np.ndarray) -> np.ndarray:
        """
        結合された点群 (2フレーム分) のボクセル単位で重心 (XYzI) を計算して返す
        """
        voxel_points_data = defaultdict(list)

        # ボクセルキー生成と全ての点 (強度を含む) のボクセルへの格納
        for pt in combined_pts:
            key = tuple(np.floor(pt[:3] / self.voxel_size).astype(np.int32))
            voxel_points_data[key].append(pt)

        # 各ボクセルの重心 (平均) を計算（強度を含む）
        filtered = []
        for key in voxel_points_data:
            pts_in_voxel = np.array(voxel_points_data[key])
            avg_x = np.mean(pts_in_voxel[:, 0])
            avg_y = np.mean(pts_in_voxel[:, 1])
            avg_z = np.mean(pts_in_voxel[:, 2])
            avg_i = np.mean(pts_in_voxel[:, 3]) # 強度を平均
            filtered.append([avg_x, avg_y, avg_z, avg_i])
        
        return np.array(filtered, dtype=np.float32)

    def save_pcd(self, points: np.ndarray):
        """PCD形式で保存 (点のみ)"""
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3]) 
            filename = os.path.join(self.pcd_output_dir, f"{self.sequence_id:06d}.pcd")
            o3d.io.write_point_cloud(filename, pcd, write_ascii=True)
            self.get_logger().info(f"Saved PCD (ASCII用にXYZのみ): {filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to save PCD: {e}")

    def save_bin(self, points: np.ndarray):
        """バイナリ形式 (.bin) で保存 (xyzi)"""
        try:
            filename = os.path.join(self.bin_output_dir, f"{self.sequence_id:06d}.bin")
            points.astype(np.float32).tofile(filename) # XYZIを保存
            self.get_logger().info(f"Saved BIN (XYZI): {filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to save BIN: {e}")

    def save_metadata(self, points: np.ndarray, timestamp):
        """メタデータをテキスト形式 (.txt) で保存"""
        try:
            filename = os.path.join(self.txt_output_dir, f"{self.sequence_id:06d}.txt")
            metadata = {
                "file_name_pcd": f"{self.sequence_id:06d}.pcd",
                "file_name_bin": f"{self.sequence_id:06d}.bin",
                "sequence_id": self.sequence_id,
                "num_points": points.shape[0],
                "timestamp_sec": timestamp.sec,
                "timestamp_nanosec": timestamp.nanosec
            }
            with open(filename, 'w') as f:
                json.dump(metadata, f, indent=4)
            self.get_logger().info(f"Saved metadata: {filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to save metadata: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = LidarTwoFrameSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()