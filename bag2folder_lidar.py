import os
import rosbag
import numpy as np
import open3d as o3d
from sensor_msgs import point_cloud2

# ================= 配置区 =================
BAG_FILE = './basalt_26_3_big.bag'          # 你的 bag 文件路径
LIDAR_TOPIC = '/livox/lidar'     # 你的雷达 Topic
OUTPUT_DIR = './lidar_frames'       # 输出目录
# ==========================================

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def save_pcd_from_bag():
    print(f"正在处理: {BAG_FILE}")
    bag = rosbag.Bag(BAG_FILE, "r")
    count = 0

    for topic, msg, t in bag.read_messages(topics=[LIDAR_TOPIC]):
        # 1. 解析 PointCloud2 消息中的点坐标 (x, y, z) 和反射强度 (intensity)
        # 注意：如果你的雷达只有 (x,y,z)，请移除 'intensity'
        field_names = ("x", "y", "z", "intensity")
        cloud_data = list(point_cloud2.read_points(msg, field_names=field_names, skip_nans=True))
        
        if not cloud_data:
            continue
            
        points = np.array(cloud_data)
        
        # 2. 转换为 Open3D 点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        # 3. 如果有强度信息，可以将其映射到颜色或单独保存（MATLAB可以识别）
        # 这里简单保存为包含坐标的 PCD
        file_path = os.path.join(OUTPUT_DIR, f"frame_{count:06d}_{t.to_nsec()}.pcd")
        
        # 4. 保存文件 (使用二进制压缩格式，MATLAB 读取更快)
        o3d.io.write_point_cloud(file_path, pcd, write_ascii=False)
        
        count += 1
        if count % 10 == 0:
            print(f"已提取 {count} 帧...")

    bag.close()
    print(f"完成！共提取 {count} 帧，保存在 {OUTPUT_DIR}")

if __name__ == "__main__":
    save_pcd_from_bag()
