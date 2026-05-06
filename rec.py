#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class StereoRectifier:
    def __init__(self):
        rospy.init_node('fisheye_stereo_rectifier', anonymous=True)
        self.bridge = CvBridge()

        # ==========================================
        # 1. 填入你的相机参数
        # ==========================================
        # cam0 (Left)
        self.xi_0 = np.array([1.12982056])
        self.D_0 = np.array([0.02107552, -0.07657528, 0.00036545, -0.00053907])
        self.K_0 = np.array([
            [949.42753588, 0.0, 972.33687702],
            [0.0, 949.66755126, 587.26231267],
            [0.0, 0.0, 1.0]
        ])

        # cam1 (Right)
        self.xi_1 = np.array([1.12437983])
        self.D_1 = np.array([0.01863795, -0.07427101, -0.00013423, 0.00010909])
        self.K_1 = np.array([
            [945.95142923, 0.0, 966.77167353],
            [0.0, 946.05602896, 615.18759191],
            [0.0, 0.0, 1.0]
        ])

        # Baseline T_1_0 (q: [x, y, z, w])
        x, y, z, w = -0.00176472, 0.64489424, -0.27399369, 0.71346742
        # 由四元数构造旋转矩阵 R10
        self.R_10 = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
        ])
        self.t_10 = np.array([-0.05120867, -0.01938352, -0.04599073])

        # ==========================================
        # 2. 计算校正矩阵与目标参数
        # ==========================================
        self.calculate_rectification()

        # 定义去畸变后的目标针孔相机内参和分辨率 (你可以根据需要调整以防FOV丢失)
        self.new_W, self.new_H = 960, 540
        self.new_fx, self.new_fy = 300.0, 300.0
        self.new_cx, self.new_cy = self.new_W / 2.0, self.new_H / 2.0
        
        self.K_new = np.array([
            [self.new_fx, 0.0, self.new_cx],
            [0.0, self.new_fy, self.new_cy],
            [0.0, 0.0, 1.0]
        ])

        # 生成映射表 (cv2.omnidir 需要 opencv-contrib-python)
        rospy.loginfo("正在计算 Omni 映射表，这可能需要几秒钟...")
        self.map1_l, self.map2_l = cv2.omnidir.initUndistortRectifyMap(
            self.K_0, self.D_0, self.xi_0, self.R_rect0, self.K_new, 
            (self.new_W, self.new_H), cv2.CV_32FC1, cv2.omnidir.RECTIFY_PERSPECTIVE
        )
        self.map1_r, self.map2_r = cv2.omnidir.initUndistortRectifyMap(
            self.K_1, self.D_1, self.xi_1, self.R_rect1, self.K_new, 
            (self.new_W, self.new_H), cv2.CV_32FC1, cv2.omnidir.RECTIFY_PERSPECTIVE
        )
        rospy.loginfo("映射表计算完毕！")

        self.print_dso_config()

        # ==========================================
        # 3. ROS 发布与订阅
        # ==========================================
        self.pub_left = rospy.Publisher('/cam0/image_raw', Image, queue_size=10)
        self.pub_right = rospy.Publisher('/cam1/image_raw', Image, queue_size=10)

        # 确保双目时间戳同步
        sub_left = message_filters.Subscriber('/fisheye/left/image_raw', Image)
        sub_right = message_filters.Subscriber('/fisheye/right/image_raw', Image)
        
        self.ts = message_filters.ApproximateTimeSynchronizer([sub_left, sub_right], 10, 0.05)
        self.ts.registerCallback(self.image_callback)
        rospy.loginfo("节点已启动，等待图像输入...")

    def calculate_rectification(self):
        # 计算 cam1 在 cam0 下的平移向量 (即基线向量)
        t_01 = -self.t_10
        self.baseline = np.linalg.norm(t_01)

        # 新 X 轴：两相机光心的连线方向
        X_new = t_01 / self.baseline

        # 光轴平分线计算
        Z_0 = np.array([0, 0, 1])
        Z_1 = self.R_10.T @ np.array([0, 0, 1])
        Z_bisect = (Z_0 + Z_1) / 2
        Z_bisect /= np.linalg.norm(Z_bisect)

        # 确立正交的新 Y 轴与 Z 轴
        Y_new = np.cross(Z_bisect, X_new)
        Y_new /= np.linalg.norm(Y_new)
        Z_new = np.cross(X_new, Y_new)
        Z_new /= np.linalg.norm(Z_new)

        # 构建校正旋转矩阵
        self.R_rect0 = np.vstack((X_new, Y_new, Z_new)).T
        self.R_rect1 = (self.R_rect0 @ self.R_10)

    def print_dso_config(self):
        # 打印可以直接贴入 stereo-dso 的参数
        config_text = f"""
=============================================
[stereo_dso] Configuration Parameters:
=============================================
{self.new_fx:.4f} {self.new_fy:.4f} {self.new_cx:.4f} {self.new_cy:.4f} 0.0 0.0 0.0 0.0
{self.new_H} {self.new_W}
crop
{self.new_H} {self.new_W}
{self.baseline:.6f}
=============================================
"""
        rospy.loginfo(config_text)

    def image_callback(self, msg_left, msg_right):
        try:
            cv_img_left = self.bridge.imgmsg_to_cv2(msg_left, "mono8")
            cv_img_right = self.bridge.imgmsg_to_cv2(msg_right, "mono8")

            # Remap 去畸变与校正
            rect_left = cv2.remap(cv_img_left, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
            rect_right = cv2.remap(cv_img_right, self.map1_r, self.map2_r, cv2.INTER_LINEAR)

            # 转换为ROS消息并发布 (保持原时间戳)
            msg_rect_left = self.bridge.cv2_to_imgmsg(rect_left, "mono8")
            msg_rect_left.header = msg_left.header
            self.pub_left.publish(msg_rect_left)

            msg_rect_right = self.bridge.cv2_to_imgmsg(rect_right, "mono8")
            msg_rect_right.header = msg_right.header
            self.pub_right.publish(msg_rect_right)

        except Exception as e:
            rospy.logerr(f"Image processing error: {e}")

if __name__ == '__main__':
    try:
        node = StereoRectifier()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
