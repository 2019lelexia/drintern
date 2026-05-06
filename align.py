#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
import os
import json  # 引入 json 库

class TrajectoryInterpolator:
    def __init__(self, tum_trajectory_file):
        self.timestamps = []
        self.translations = []
        self.rotations = []
        
        print(f"📂 正在加载 SLAM 轨迹: {tum_trajectory_file}")
        with open(tum_trajectory_file, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                data = list(map(float, line.strip().split()))
                if len(data) >= 8:
                    self.timestamps.append(data[0])
                    self.translations.append(data[1:4])
                    self.rotations.append(data[4:8]) # qx, qy, qz, qw
                    
        self.timestamps = np.array(self.timestamps)
        self.translations = np.array(self.translations)
        self.rot_obj = R.from_quat(self.rotations)
        self.slerp = Slerp(self.timestamps, self.rot_obj)
        print(f"✅ 轨迹加载成功，共 {len(self.timestamps)} 个位姿节点。")

    def get_pose_at_time(self, target_time):
        if target_time < self.timestamps[0] or target_time > self.timestamps[-1]:
            raise ValueError(f"❌ 目标时间 {target_time} 超出了轨迹覆盖的时间范围！")

        # 线性插值平移
        interp_t = np.array([
            np.interp(target_time, self.timestamps, self.translations[:, 0]),
            np.interp(target_time, self.timestamps, self.translations[:, 1]),
            np.interp(target_time, self.timestamps, self.translations[:, 2])
        ])

        # 球面线性插值旋转
        interp_r = self.slerp([target_time])[0]
        rot_matrix = interp_r.as_matrix()

        T_map_lidar = np.eye(4)
        T_map_lidar[:3, :3] = rot_matrix
        T_map_lidar[:3, 3] = interp_t
        return T_map_lidar

def main():
    # ==========================================
    # 1. 配置文件路径 (请确保这些文件在当前目录下，或填写绝对路径)
    # ==========================================
    TRAJ_FILE = "fastlio_traj.txt"   # FAST-LIO 导出的 TUM 格式轨迹
    MAP_PCD_FILE = "scans.pcd"       # FAST-LIO 的全局点云地图
    CAM_PCD_FILE = "c1.pcd"      # 你生成的双目深度点云
    
    # ==========================================
    # 2. 时间戳换算与对齐
    # ==========================================
    raw_time_ns = 1735690730009560219
    raw_time_s = raw_time_ns / 1e9
    time_offset = -0.171 # 相机与雷达的硬件时间差补偿
    sync_time = raw_time_s + time_offset
    
    print(f"\n⏱️ 原始相机时间: {raw_time_s:.6f} 秒")
    print(f"⏱️ 补偿后同步时间: {sync_time:.6f} 秒")

    try:
        # ==========================================
        # 3. 获取雷达在 Map 中的位姿 (T_map_lidar)
        # ==========================================
        traj = TrajectoryInterpolator(TRAJ_FILE)
        T_map_lidar = traj.get_pose_at_time(sync_time)

        # ==========================================
        # 4. 计算立体校正的“脖子扭转矩阵” (R_rect0)
        # ==========================================
        qx, qy, qz, qw = -0.00176472,  0.64489424, -0.27399369,  0.71346742
        R_10 = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)]
        ])
        t_10 = np.array([-0.05120867, -0.01938352, -0.04599073])

        t_01 = -t_10
        baseline = np.linalg.norm(t_01)
        X_new = t_01 / baseline
        Z_0 = np.array([0, 0, 1])
        Z_1 = R_10.T @ np.array([0, 0, 1])
        Z_bisect = (Z_0 + Z_1) / 2
        Z_bisect /= np.linalg.norm(Z_bisect)
        Y_new = np.cross(Z_bisect, X_new)
        Y_new /= np.linalg.norm(Y_new)
        Z_new = np.cross(X_new, Y_new)
        Z_new /= np.linalg.norm(Z_new)
        
        R_rect0 = np.vstack((X_new, Y_new, Z_new)).T

        # 构造把双目点云“扭回”原始左目光轴的 4x4 矩阵
        T_rect_to_cam0 = np.eye(4)
        T_rect_to_cam0[:3, :3] = R_rect0.T  # 注意这里是转置(即求逆)

        # ==========================================
        # 5. 你标定出的最终物理外参 (T_cl: Lidar -> Cam0)
        # ==========================================
        # ⚠️ 注意：这里填入的是你之前平移跑到 4m 的那个失效外参，
        # 如果你重新标定成功了，请务必更新这里的数值！
          
        
        # ours by step-by-step
        # T_cl = np.array([
        #     [ 0.03962, -0.99427,  0.09933, -0.05041],
        #     [ 0.29251, -0.08351, -0.95261, -0.01213],
        #     [ 0.95544,  0.0668,   0.28752, -0.02877],
        #     [ 0.,       0.,       0.,       1.     ],
        # ])
        

        # ours by ransac+ceres
        # T_cl = np.array([
        #     [ 0.691264,-0.722374,-0.0181549,0.0238257],
        #     [ 0.274638,0.285883,-0.918066,-0.0349469],
        #     [ 0.668377,0.62964,0.396012,-0.00742667],
        #     [ 0.,       0.,       0.,       1.     ],
        # ])


        # ours by ceres
        # T_cl = np.array([
        #     [ 0.693731,-0.720165,0.00997824,-0.0266551],
        #     [ 0.29985,0.276192,-0.913131,-0.0627643],
        #     [ 0.654849,0.636459,0.407545,-0.0217571],
        #     [ 0.,       0.,       0.,       1.     ],
        # ])
        
        # calib by zt
        T_cl = np.array([
            [ 0.692095,  -0.721586,   0.017848,-0.017761],
            [ 0.299952,   0.265027,  -0.916401,-0.049350],
            [ 0.656531,   0.639590,   0.399864,-0.029783],
            [ 0.,       0.,       0.,       1.     ],
        ])
        
        T_lc = np.linalg.inv(T_cl) # 转换为 Lidar <- Cam0

        # ==========================================
        # 6. 终极坐标系连乘打通！
        # ==========================================
        # 公式: P_map = T_map_lidar * T_lc * T_rect_to_cam0 * P_stereo_cloud
        T_map_stereo_camera = T_map_lidar @ T_lc @ T_rect_to_cam0
        
        # 📍 提取当前相机在地图中的绝对 XYZ 坐标
        camera_pos = T_map_stereo_camera[:3, 3]

        # ==========================================
        # 7. 加载点云并执行“空间局部裁剪” (Spatial Cropping)
        # ==========================================
        print("\n🖼️ 正在加载并裁剪局部雷达点云...")
        if not os.path.exists(MAP_PCD_FILE) or not os.path.exists(CAM_PCD_FILE):
            print("❌ 找不到 PCD 文件！")
            return

        map_pcd = o3d.io.read_point_cloud(MAP_PCD_FILE)
        cam_pcd = o3d.io.read_point_cloud(CAM_PCD_FILE)

        # 🚀 魔法：以相机为中心，生成一个边长 30 米的空间包围盒
        # 这完美代替了“按时间找帧”，且瞬间过滤掉几百万个无用的远景雷达点！
        radius = 5.0  # 视距半径 (15米，你可以自由调小到 5米 或 10米)
        min_bound = camera_pos - radius
        max_bound = camera_pos + radius
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        
        # 执行一键裁剪，得到局部地图！
        local_map_pcd = map_pcd.crop(bbox)
        print(f"✂️ 裁剪完成！雷达点云从 {len(map_pcd.points)} 点精简至 {len(local_map_pcd.points)} 点。")

        # 🌟 对局部地图进行视觉处理 (降采样 + 纯白发光无阴影)
        local_map_pcd = local_map_pcd.voxel_down_sample(voxel_size=0.05) 
        local_map_pcd.normals = o3d.utility.Vector3dVector() 
        local_map_pcd.paint_uniform_color([1.0, 1.0, 1.0]) 

        # 🌟 相机点云变换 (飞迁到全局坐标系，保留彩色)
        cam_pcd.transform(T_map_stereo_camera)

        # 生成辅助观察的轨迹红线
        traj_points = o3d.geometry.PointCloud()
        traj_points.points = o3d.utility.Vector3dVector(traj.translations[::5])
        traj_points.paint_uniform_color([1, 0, 0])

        # ==========================================
        # 8. 启动高级无光照渲染器
        # ==========================================
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="标定", width=1280, height=720)
        
        # ⚠️ 注意这里添加的是 local_map_pcd
        vis.add_geometry(local_map_pcd) 
        vis.add_geometry(cam_pcd)
        vis.add_geometry(traj_points)
        
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.05, 0.05, 0.05]) 
        opt.point_size = 1.0  
        opt.light_on = False

        # ==========================================
        # 🌟 新增：固定相机视角
        # ==========================================
        

        # 1. 获取视角控制器
        ctr = vis.get_view_control()
        
        # 2. 读取 viewpoint.json
        param_path = "viewer2.json"
        if os.path.exists(param_path):
            try:
                with open(param_path, 'r', encoding='utf-8') as f:
                    view_data = json.load(f)
                
                # 提取 trajectory 列表中的第一个视角配置
                if "trajectory" in view_data and len(view_data["trajectory"]) > 0:
                    cam_params = view_data["trajectory"][0]
                    
                    # 强行设置控制器的四个核心参数
                    ctr.set_front(cam_params["front"])
                    ctr.set_lookat(cam_params["lookat"])
                    ctr.set_up(cam_params["up"])
                    ctr.set_zoom(cam_params["zoom"])
                    
                    print(f"✅ 成功加载固定视角: {param_path}")
                else:
                    print("⚠️ viewpoint.json 格式不符合预期，未能找到 trajectory。")
            except Exception as e:
                print(f"⚠️ 读取视角文件失败: {e}")
        else:
            print("⚠️ 未找到 viewpoint.json，将使用默认视角。请在窗口中按 Ctrl+C 提取视角并保存。")

        print("✅ 准备就绪！弹出 3D 视窗...")
        vis.run()
        vis.destroy_window()

    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
