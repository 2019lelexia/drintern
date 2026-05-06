import rosbag
import cv2
import numpy as np
import os
import sys
from cv_bridge import CvBridge

# ==========================================
# 话题配置区 (请确保与你的 bag 包内一致)
# ==========================================
TOPIC_CAM0 = '/fisheye/left/image_raw'
TOPIC_CAM1 = '/fisheye/right/image_raw'

class OfflineStereoRectifier:
    def __init__(self):
        # 1. 原始 Kalibr 鱼眼参数
        self.xi_0 = np.array([1.01554299])
        self.D_0 = np.array([-0.0426741,  -0.04054315,  0.00067815, -0.00046529])
        self.K_0 = np.array([
            [899.78068828, 0.0, 971.0864345],
            [0.0, 899.96302634, 586.29007694],
            [0.0, 0.0, 1.0]
        ])

        self.xi_1 = np.array([0.87395489])
        self.D_1 = np.array([-0.10477637, -0.00593549,  0.00017271,  0.00005546])
        self.K_1 = np.array([
            [835.46195977, 0.0, 967.72247886],
            [0.0, 835.26719106, 613.92519162],
            [0.0, 0.0, 1.0]
        ])

        # ==========================================
        # 🌟 核心替换：使用新标定的外参 (Baseline T_1_0)
        # ==========================================
        # 根据提示，将提取到的旋转矩阵直接进行转置 (.T)
        R_new = np.array([
            [0.014805, -0.393210, -0.919357],
            [0.389393,  0.849100, -0.356895],
            [0.920957, -0.352657,  0.165705]
        ])
        self.R_10 = R_new.T  
        
        # 替换为最新的平移向量
        self.t_10 = np.array([-0.050396, -0.018390, -0.044998])

        # 2. 计算校正矩阵
        self.calculate_rectification()

        # 目标针孔相机内参和分辨率
        self.new_W, self.new_H = 960, 540
        self.new_fx, self.new_fy = 500.0, 500.0
        self.new_cx, self.new_cy = self.new_W / 2.0, self.new_H / 2.0
        self.K_new = np.array([
            [self.new_fx, 0.0, self.new_cx],
            [0.0, self.new_fy, self.new_cy],
            [0.0, 0.0, 1.0]
        ])

        print("🔄 正在预计算双目 Omni 彩色映射表...")
        self.map1_l, self.map2_l = cv2.omnidir.initUndistortRectifyMap(
            self.K_0, self.D_0, self.xi_0, self.R_rect0, self.K_new, 
            (self.new_W, self.new_H), cv2.CV_32FC1, cv2.omnidir.RECTIFY_PERSPECTIVE
        )
        self.map1_r, self.map2_r = cv2.omnidir.initUndistortRectifyMap(
            self.K_1, self.D_1, self.xi_1, self.R_rect1, self.K_new, 
            (self.new_W, self.new_H), cv2.CV_32FC1, cv2.omnidir.RECTIFY_PERSPECTIVE
        )
        print("✅ 映射表计算完毕！")

    def calculate_rectification(self):
        t_01 = -self.t_10
        self.baseline = np.linalg.norm(t_01)
        X_new = t_01 / self.baseline
        Z_0 = np.array([0, 0, 1])
        Z_1 = self.R_10.T @ np.array([0, 0, 1])
        Z_bisect = (Z_0 + Z_1) / 2
        Z_bisect /= np.linalg.norm(Z_bisect)
        Y_new = np.cross(Z_bisect, X_new)
        Y_new /= np.linalg.norm(Y_new)
        Z_new = np.cross(X_new, Y_new)
        Z_new /= np.linalg.norm(Z_new)
        self.R_rect0 = np.vstack((X_new, Y_new, Z_new)).T
        self.R_rect1 = (self.R_rect0 @ self.R_10)

def extract_and_rectify(bag_path, output_dir):
    rectifier = OfflineStereoRectifier()
    
    # 构建 EuRoC 目录
    cam0_dir = os.path.join(output_dir, 'mav0', 'cam0', 'data')
    cam1_dir = os.path.join(output_dir, 'mav0', 'cam1', 'data')
    os.makedirs(cam0_dir, exist_ok=True)
    os.makedirs(cam1_dir, exist_ok=True)

    cam0_csv = os.path.join(output_dir, 'mav0', 'cam0', 'data.csv')
    cam1_csv = os.path.join(output_dir, 'mav0', 'cam1', 'data.csv')

    bridge = CvBridge()
    msg_count = 0

    print(f"\n📦 开始边解包边去畸变 (RGB模式): {bag_path}")
    
    with rosbag.Bag(bag_path, 'r') as bag, \
         open(cam0_csv, 'w', newline='') as f_cam0, \
         open(cam1_csv, 'w', newline='') as f_cam1:

        f_cam0.write('#timestamp [ns],filename\n')
        f_cam1.write('#timestamp [ns],filename\n')

        for topic, msg, t in bag.read_messages(topics=[TOPIC_CAM0, TOPIC_CAM1]):
            timestamp_ns = str(msg.header.stamp.to_nsec()) if hasattr(msg, 'header') else str(t.to_nsec())

            if topic == TOPIC_CAM0:
                cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
                rect_img = cv2.remap(cv_img, rectifier.map1_l, rectifier.map2_l, cv2.INTER_LINEAR)
                filename = f"{timestamp_ns}.png"
                cv2.imwrite(os.path.join(cam0_dir, filename), rect_img)
                f_cam0.write(f"{timestamp_ns},{filename}\n")
                msg_count += 1

            elif topic == TOPIC_CAM1:
                cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
                rect_img = cv2.remap(cv_img, rectifier.map1_r, rectifier.map2_r, cv2.INTER_LINEAR)
                filename = f"{timestamp_ns}.png"
                cv2.imwrite(os.path.join(cam1_dir, filename), rect_img)
                f_cam1.write(f"{timestamp_ns},{filename}\n")
                msg_count += 1

            if msg_count % 1000 == 0:
                print(f"⏳ 已处理并校正 {msg_count} 条数据...")

    print("✅ 完美收工！所有彩色图像已转换为无畸变的针孔立体对。")

    print("\n=============================================")
    print("📋 VI-Stereo-DSO / 深度估计模型 标准参数:")
    print("=============================================")
    print(f"1. 针孔内参矩阵 (K_new):")
    print(f"fx={rectifier.new_fx}, fy={rectifier.new_fy}, cx={rectifier.new_cx}, cy={rectifier.new_cy}")
    print(f"分辨率: {rectifier.new_W}x{rectifier.new_H}")
    print(f"\n2. 立体基线 (Baseline):")
    print(f"B = {rectifier.baseline:.6f} 米")
    print("=============================================\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python3 extract_color_stereo.py <修复好时间戳的bag包> <输出目录>")
    else:
        extract_and_rectify(sys.argv[1], sys.argv[2])