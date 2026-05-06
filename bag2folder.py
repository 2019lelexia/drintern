# !/usr/bin/env python3
# -*- coding: utf-8 -*-

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
TOPIC_IMU  = '/livox/imu'

class OfflineStereoRectifier:
    def __init__(self):
        # 1. 原始 Kalibr 鱼眼参数
        self.xi_0 = np.array([1.12982056])
        self.D_0 = np.array([0.02107552, -0.07657528, 0.00036545, -0.00053907])
        self.K_0 = np.array([
            [949.42753588, 0.0, 972.33687702],
            [0.0, 949.66755126, 587.26231267],
            [0.0, 0.0, 1.0]
        ])

        self.xi_1 = np.array([1.12437983])
        self.D_1 = np.array([0.01863795, -0.07427101, -0.00013423, 0.00010909])
        self.K_1 = np.array([
            [945.95142923, 0.0, 966.77167353],
            [0.0, 946.05602896, 615.18759191],
            [0.0, 0.0, 1.0]
        ])

        # Baseline T_1_0
        x, y, z, w = -0.00176472, 0.64489424, -0.27399369, 0.71346742
        self.R_10 = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
        ])
        self.t_10 = np.array([-0.05120867, -0.01938352, -0.04599073])

        # 2. 计算校正矩阵
        self.calculate_rectification()

        # 目标针孔相机内参和分辨率
        self.new_W, self.new_H = 960, 540
        self.new_fx, self.new_fy = 300.0, 300.0
        self.new_cx, self.new_cy = self.new_W / 2.0, self.new_H / 2.0
        self.K_new = np.array([
            [self.new_fx, 0.0, self.new_cx],
            [0.0, self.new_fy, self.new_cy],
            [0.0, 0.0, 1.0]
        ])

        print("🔄 正在预计算双目 Omni 静态映射表...")
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
    imu_dir = os.path.join(output_dir, 'mav0', 'imu0')
    os.makedirs(cam0_dir, exist_ok=True)
    os.makedirs(cam1_dir, exist_ok=True)
    os.makedirs(imu_dir, exist_ok=True)

    cam0_csv = os.path.join(output_dir, 'mav0', 'cam0', 'data.csv')
    cam1_csv = os.path.join(output_dir, 'mav0', 'cam1', 'data.csv')
    imu_csv = os.path.join(output_dir, 'mav0', 'imu0', 'data.csv')

    bridge = CvBridge()
    msg_count = 0

    print(f"\n📦 开始边解包边去畸变: {bag_path}")
    
    with rosbag.Bag(bag_path, 'r') as bag, \
         open(cam0_csv, 'w', newline='') as f_cam0, \
         open(cam1_csv, 'w', newline='') as f_cam1, \
         open(imu_csv, 'w', newline='') as f_imu:

        f_cam0.write('#timestamp [ns],filename\n')
        f_cam1.write('#timestamp [ns],filename\n')
        f_imu.write('#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n')

        for topic, msg, t in bag.read_messages(topics=[TOPIC_CAM0, TOPIC_CAM1, TOPIC_IMU]):
            timestamp_ns = str(msg.header.stamp.to_nsec()) if hasattr(msg, 'header') else str(t.to_nsec())

            if topic == TOPIC_CAM0:
                cv_img = bridge.imgmsg_to_cv2(msg, "mono8")
                # 核心：直接在这里使用预计算好的映射表进行插值校正
                rect_img = cv2.remap(cv_img, rectifier.map1_l, rectifier.map2_l, cv2.INTER_LINEAR)
                filename = f"{timestamp_ns}.png"
                cv2.imwrite(os.path.join(cam0_dir, filename), rect_img)
                f_cam0.write(f"{timestamp_ns},{filename}\n")
                msg_count += 1

            elif topic == TOPIC_CAM1:
                cv_img = bridge.imgmsg_to_cv2(msg, "mono8")
                rect_img = cv2.remap(cv_img, rectifier.map1_r, rectifier.map2_r, cv2.INTER_LINEAR)
                filename = f"{timestamp_ns}.png"
                cv2.imwrite(os.path.join(cam1_dir, filename), rect_img)
                f_cam1.write(f"{timestamp_ns},{filename}\n")
                msg_count += 1

            elif topic == TOPIC_IMU:
                f_imu.write(f"{timestamp_ns},{msg.angular_velocity.x},{msg.angular_velocity.y},{msg.angular_velocity.z},{msg.linear_acceleration.x},{msg.linear_acceleration.y},{msg.linear_acceleration.z}\n")
                msg_count += 1

            if msg_count % 1000 == 0:
                print(f"⏳ 已处理并校正 {msg_count} 条数据...")

    print("✅ 完美收工！所有图像已转换为无畸变的针孔立体对。")

    # 打印给你直接用的 DSO 配置
    print("\n=============================================")
    print("📋 VI-Stereo-DSO 标定文件参数 (直接复制使用):")
    print("=============================================")
    print(f"1. cam0.txt 和 cam1.txt 内容完全相同:")
    print(f"Pinhole {rectifier.new_fx} {rectifier.new_fy} {rectifier.new_cx} {rectifier.new_cy} 0")
    print(f"{rectifier.new_W} {rectifier.new_H}")
    print("none")
    print(f"{rectifier.new_W} {rectifier.new_H}")
    print(f"\n2. T_C0C1.txt (已通过极线校正完全平行化，R为单位阵):")
    print(f"{rectifier.baseline:.6f} 0.0 0.0")
    print("0.0 0.0 0.0")
    print("=============================================\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python3 extract_and_rectify.py <修复好时间戳的bag包> <输出目录>")
    else:
        extract_and_rectify(sys.argv[1], sys.argv[2])
