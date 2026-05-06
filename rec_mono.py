#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class MonoFisheyeUndistorter:
    def __init__(self):
        rospy.init_node('mono_fisheye_undistorter', anonymous=True)
        self.bridge = CvBridge()

        # ==========================================
        # 1. 左目相机 (cam0) 原始 Kalibr 鱼眼参数
        # ==========================================
        self.xi_0 = np.array([1.12982056])
        self.D_0 = np.array([0.02107552, -0.07657528, 0.00036545, -0.00053907])
        self.K_0 = np.array([
            [949.42753588, 0.0, 972.33687702],
            [0.0, 949.66755126, 587.26231267],
            [0.0, 0.0, 1.0]
        ])

        # ==========================================
        # 2. 设定目标针孔相机参数
        # ==========================================
        self.new_W, self.new_H = 960, 540
        self.new_fx, self.new_fy = 300.0, 300.0
        self.new_cx, self.new_cy = self.new_W / 2.0, self.new_H / 2.0
        
        self.K_new = np.array([
            [self.new_fx, 0.0, self.new_cx],
            [0.0, self.new_fy, self.new_cy],
            [0.0, 0.0, 1.0]
        ])

        # 【核心修改】：单目去畸变，保持原光轴方向不变，因此旋转矩阵为单位阵
        self.R_rect0 = np.eye(3)

        # 生成映射表
        rospy.loginfo("正在计算 Omni 单目映射表...")
        self.map1_l, self.map2_l = cv2.omnidir.initUndistortRectifyMap(
            self.K_0, self.D_0, self.xi_0, self.R_rect0, self.K_new, 
            (self.new_W, self.new_H), cv2.CV_32FC1, cv2.omnidir.RECTIFY_PERSPECTIVE
        )
        rospy.loginfo("映射表计算完毕！")

        self.print_config()

        # ==========================================
        # 3. ROS 发布与订阅 (去除双目同步，直接单话题回调，极致低延迟)
        # ==========================================
        self.pub_left = rospy.Publisher('/cam0/image_raw', Image, queue_size=1)
        
        rospy.Subscriber('/fisheye/left/image_raw', Image, self.image_callback, queue_size=2)
        rospy.loginfo("单目去畸变节点已启动，等待 /fisheye/left/image_raw 输入...")

    def print_config(self):
        # 打印给 DM-VIO 用的单目内参配置
        config_text = f"""
=============================================
DM-VIO Camera Intrinsics (pinhole, radtan):
=============================================
{self.new_fx:.4f} {self.new_fy:.4f} {self.new_cx:.4f} {self.new_cy:.4f} 0.0 0.0 0.0 0.0
{self.new_W} {self.new_H}
crop
{self.new_W} {self.new_H}
=============================================
"""
        rospy.loginfo(config_text)

    def image_callback(self, msg_left):
        try:
            cv_img_left = self.bridge.imgmsg_to_cv2(msg_left, "mono8")

            # Remap 去畸变
            rect_left = cv2.remap(cv_img_left, self.map1_l, self.map2_l, cv2.INTER_LINEAR)

            # 转换为ROS消息并发布 (严格保持原时间戳)
            msg_rect_left = self.bridge.cv2_to_imgmsg(rect_left, "mono8")
            msg_rect_left.header = msg_left.header
            self.pub_left.publish(msg_rect_left)

        except Exception as e:
            rospy.logerr(f"Image processing error: {e}")

if __name__ == '__main__':
    try:
        node = MonoFisheyeUndistorter()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass