#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy import interpolate
import matplotlib.pyplot as plt
import sys
import os

def load_tum_trajectory(filepath):
    if not os.path.exists(filepath):
        print(f"❌ 找不到文件: {filepath}")
        sys.exit(1)
    data = np.loadtxt(filepath)
    return data[:, 0], data[:, 1:4], data[:, 4:8]

def get_relative_motions(times, positions, quats, sample_times):
    interp_pos = interpolate.interp1d(times, positions, axis=0, kind='linear')
    pos_sampled = interp_pos(sample_times)
    
    slerp = Slerp(times, R.from_quat(quats))
    rot_sampled = slerp(sample_times)
    
    R_matrices, t_vectors = [], []
    for i in range(len(sample_times) - 1):
        R_i = rot_sampled[i].as_matrix()
        t_i = pos_sampled[i]
        
        R_next = rot_sampled[i+1].as_matrix()
        t_next = pos_sampled[i+1]
        
        R_matrices.append(R_i.T @ R_next)
        t_vectors.append((R_i.T @ (t_next - t_i)).reshape(3, 1))
        
    # 修改点：把绝对位姿也返回，留给最后画图用
    return R_matrices, t_vectors, pos_sampled, rot_sampled

def main(fastlio_file, dso_file):
    print("📂 正在读取原始轨迹数据...")
    t_lio, p_lio, q_lio = load_tum_trajectory(fastlio_file)
    t_dso, p_dso, q_dso = load_tum_trajectory(dso_file)
    
    # ⏱️ 强制时间对齐
    time_offset = -0.1710  
    t_dso = t_dso + time_offset 

    # ✂️ 黄金区间截取
    t_absolute_start = max(t_lio[0], t_dso[0])
    t_absolute_end = min(t_lio[-1], t_dso[-1])
    
    t_start_cropped = t_absolute_start + 5
    t_end_cropped = min(t_absolute_start + 270.0, t_absolute_end)
    
    step_sec = 0.5 
    sample_times = np.arange(t_start_cropped, t_end_cropped, step_sec)
    
    R_L_list, t_L_list, p_L_abs, rot_L_abs = get_relative_motions(t_lio, p_lio, q_lio, sample_times)
    R_C_list, t_C_list, p_C_abs, rot_C_abs = get_relative_motions(t_dso, p_dso, q_dso, sample_times)
    
    # =========================================================
    # 步骤 1：在线计算最佳尺度 s
    # =========================================================
    scale_list = []
    for i in range(len(t_L_list)):
        norm_L = np.linalg.norm(t_L_list[i])
        norm_C = np.linalg.norm(t_C_list[i])
        if norm_L > 0.2 and norm_C > 0.05:
            scale_list.append(norm_L / norm_C)
            
    optimal_scale = np.median(scale_list) if len(scale_list) > 0 else 1.0
    print(f"⚖️ 算出的最优视觉轨迹尺度系数 s = {optimal_scale:.5f}")
    
    t_C_scaled_list = [t * optimal_scale for t in t_C_list]

    # =========================================================
    # 步骤 2：使用 SVD 求解并“锁死”绝对纯净的旋转外参 R_CL
    # =========================================================
    axes_L, axes_C = [], []
    for i in range(len(R_L_list)):
        rot_L = R.from_matrix(R_L_list[i])
        rot_C = R.from_matrix(R_C_list[i])
        angle_L, angle_C = rot_L.magnitude(), rot_C.magnitude()
        if angle_L > 0.02 and angle_C > 0.02:
            axes_L.append(rot_L.as_rotvec() / angle_L)
            axes_C.append(rot_C.as_rotvec() / angle_C)
            
    axes_L = np.array(axes_L).T
    axes_C = np.array(axes_C).T
    
    U, _, Vt = np.linalg.svd(axes_L @ axes_C.T)
    R_CL = Vt.T @ U.T
    if np.linalg.det(R_CL) < 0:
        Vt[2, :] *= -1
        R_CL = Vt.T @ U.T

    euler = R.from_matrix(R_CL).as_euler('xyz', degrees=True)
    print(f"🔒 旋转外参 R_CL 已被锁死 (Roll, Pitch, Yaw): [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}] 度")

    # =========================================================
    # 步骤 3：构建超定方程组 Ax = b 优化平移 t_CL
    # =========================================================
    print("🔬 正在构建最小二乘方程组以求解极致平移...")
    
    A_stack = []
    b_stack = []
    I_3x3 = np.eye(3)
    
    for i in range(len(R_L_list)):
        if R.from_matrix(R_L_list[i]).magnitude() > 0.02:
            A_i = R_C_list[i] - I_3x3
            b_i = R_CL @ t_L_list[i] - t_C_scaled_list[i]
            
            A_stack.append(A_i)
            b_stack.append(b_i)
            
    A = np.vstack(A_stack)
    b = np.vstack(b_stack)
    
    cond_num = np.linalg.cond(A.T @ A)
    print(f"📈 优化矩阵条件数 (Condition Number) = {cond_num:.2f}")
    if cond_num < 50:
        print("   ✅ 评价: 条件数极佳！你手持产生的 3D 激励非常充分，Z 轴求解高度可信。")
    elif cond_num < 500:
        print("   ⚠️ 评价: 条件数一般。有一定的高低起伏激励，平移结果基本可用。")
    else:
        print("   ❌ 评价: 条件数过高！方程呈现严重退化，你走得太平稳了，Z 轴可能会算飞。")

    t_CL, residuals, rank, s_vals = np.linalg.lstsq(A, b, rcond=None)
    
    T_CL = np.eye(4)
    T_CL[:3, :3] = R_CL
    T_CL[:3, 3] = t_CL.flatten()
    
    print("\n=======================================================")
    print("🎯 分步解耦优化完成！终极外参矩阵 T_CL:")
    print("=======================================================")
    print(np.array_str(T_CL, precision=5, suppress_small=True))
    print("-------------------------------------------------------")
    print(f"📐 极优平移 (X, Y, Z) 米: [{t_CL[0][0]:.4f}, {t_CL[1][0]:.4f}, {t_CL[2][0]:.4f}]")
    print(f"🔄 锁死旋转 (R, P, Y) 度: [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}]")
    print("=======================================================\n")

    # =========================================================
    # 步骤 4：绝对轨迹投影与可视化 (完美融合新求出的 T_CL)
    # =========================================================
    print("📊 正在执行绝对坐标系投影可视化...")
    
    # 给相机的绝对平移轨迹套上刚刚算出的最优尺度 s
    p_C_scaled = p_C_abs[0] + optimal_scale * (p_C_abs - p_C_abs[0])
    
    # 构造第一帧的绝对位姿
    T_L_0 = np.eye(4)
    T_L_0[:3, :3] = rot_L_abs[0].as_matrix()
    T_L_0[:3, 3] = p_L_abs[0]
    
    T_C_0 = np.eye(4)
    T_C_0[:3, :3] = rot_C_abs[0].as_matrix()
    T_C_0[:3, 3] = p_C_scaled[0]
    
    # 计算两个 World 坐标系的绝对对齐矩阵
    # 魔法公式：T_L_0 * T_CL^(-1) = T_W_align * T_C_0
    T_W_align = T_L_0 @ np.linalg.inv(T_CL) @ np.linalg.inv(T_C_0)
    
    p_L_estimated_by_C = []
    
    for i in range(len(p_C_scaled)):
        T_C_i = np.eye(4)
        T_C_i[:3, :3] = rot_C_abs[i].as_matrix()
        T_C_i[:3, 3] = p_C_scaled[i]
        
        # 将相机的绝对位姿投影回 LiDAR 的绝对坐标系下
        T_L_est = T_W_align @ T_C_i @ T_CL
        p_L_estimated_by_C.append(T_L_est[:3, 3])
        
    p_L_estimated_by_C = np.array(p_L_estimated_by_C)
    
    # 起点归零，方便同框对比
    shift_vector = p_L_abs[0]
    p_L_viz = p_L_abs - shift_vector
    p_C_viz = p_L_estimated_by_C - shift_vector

    # --- 开始绘图 ---
    fig = plt.figure(figsize=(14, 6))
    
    # 3D 图
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(p_L_viz[:, 0], p_L_viz[:, 1], p_L_viz[:, 2], label='FAST-LIO (Ground Truth)', color='blue', linewidth=2)
    ax1.plot(p_C_viz[:, 0], p_C_viz[:, 1], p_C_viz[:, 2], label='Stereo-DSO (Optimized $T_{CL}$)', color='green', linewidth=2)
    ax1.scatter(0, 0, 0, color='red', s=50, label='Start')
    
    ax1.set_title('Absolute Trajectory Alignment (Optimized Extrinsic)', fontsize=12)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.legend()

    # 2D 鸟瞰图
    ax2 = fig.add_subplot(122)
    ax2.plot(p_L_viz[:, 0], p_L_viz[:, 1], label='FAST-LIO', color='blue', linewidth=2)
    ax2.plot(p_C_viz[:, 0], p_C_viz[:, 1], label='Stereo-DSO (Mapped)', color='green', linewidth=2, linestyle='--')
    ax2.scatter(0, 0, color='red', s=50, label='Start', zorder=5)
    
    ax2.set_title('Bird\'s-Eye View (X-Y Plane)', fontsize=12)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    ax2.axis('equal') 

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python3 translation_optimizer.py <fastlio_traj.txt> <dso_traj.txt>")
    else:
        main(sys.argv[1], sys.argv[2])