#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy import interpolate
from scipy import signal
import sys
import os

def load_tum_trajectory(filepath):
    if not os.path.exists(filepath):
        print(f"❌ 找不到文件: {filepath}")
        sys.exit(1)
    data = np.loadtxt(filepath)
    return data[:, 0], data[:, 1:4], data[:, 4:8]

def compute_raw_angular_velocity(times, quats):
    """只计算最原始的角速度，不再做粗糙的滑动平均"""
    rotations = R.from_quat(quats)
    dt = np.diff(times)
    valid = dt > 1e-6
    dt = dt[valid]
    
    rot_diff = rotations[:-1][valid].inv() * rotations[1:][valid]
    omega = rot_diff.magnitude() / dt
    
    t_mid = times[:-1][valid] + dt / 2.0
    return t_mid, omega

def butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    巴特沃斯重型低通滤波器
    使用 filtfilt 保证零相位偏移，不改变波峰的物理时间！
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data) 
    return y

def main(fastlio_file, dso_file, output_file):
    print("📂 正在读取轨迹数据...")
    t_lio, _, q_lio = load_tum_trajectory(fastlio_file)
    t_dso, _, q_dso = load_tum_trajectory(dso_file)
    
    print("⚙️ 正在提取原始角速度...")
    t_lio_mid, w_lio = compute_raw_angular_velocity(t_lio, q_lio)
    t_dso_mid, w_dso = compute_raw_angular_velocity(t_dso, q_dso)
    
    # 构建 1ms 精度 (1000Hz 采样率) 的公共时间轴
    t_start = max(t_lio_mid[0], t_dso_mid[0])
    t_end = min(t_lio_mid[-1], t_dso_mid[-1])
    if t_start >= t_end:
        print("❌ 严重错误：两条轨迹在时间上没有任何重叠！")
        sys.exit(1)
        
    dt_res = 0.0001
    fs = 1.0 / dt_res  # 采样频率 1000 Hz
    t_common = np.arange(t_start, t_end, dt_res)
    
    # 线性插值
    interp_lio = interpolate.interp1d(t_lio_mid, w_lio, kind='linear', bounds_error=False, fill_value=0)
    interp_dso = interpolate.interp1d(t_dso_mid, w_dso, kind='linear', bounds_error=False, fill_value=0)
    w_lio_resampled = interp_lio(t_common)
    w_dso_resampled = interp_dso(t_common)
    
    # ---------------------------------------------------------
    # 🛡️ 核心火力：零相移巴特沃斯重型滤波
    # ---------------------------------------------------------
    print("🛡️ 正在应用零相移低通滤波器 (Cutoff: 1.0 Hz)...")
    # 截止频率设为 1.0 Hz (意味着只保留周期大于 1 秒的缓慢转弯动作，滤掉所有高频计算毛刺)
    cutoff_freq = 1.0 
    w_lio_filtered = butter_lowpass_filter(w_lio_resampled, cutoff_freq, fs, order=4)
    w_dso_filtered = butter_lowpass_filter(w_dso_resampled, cutoff_freq, fs, order=4)
    
    # 去除直流分量 (均值)
    w_lio_filtered -= np.mean(w_lio_filtered)
    w_dso_filtered -= np.mean(w_dso_filtered)
    
    # ---------------------------------------------------------
    # 🔬 互相关对齐
    # ---------------------------------------------------------
    print("🔬 正在进行高精度互相关对齐...")
    correlation = signal.correlate(w_lio_filtered, w_dso_filtered, mode='full')
    lags = signal.correlation_lags(len(w_lio_filtered), len(w_dso_filtered), mode='full')
    
    best_lag_idx = np.argmax(correlation)
    time_offset = lags[best_lag_idx] * dt_res
    
    print("\n=======================================================")
    print(f"🎯 重型滤波标定结果：精准时间差 t_offset = {time_offset:.4f} 秒")
    print("=======================================================\n")
    
    # ---------------------------------------------------------
    # 💾 生成新轨迹并绘图验证
    # ---------------------------------------------------------
    data_dso_raw = np.loadtxt(dso_file)
    data_dso_aligned = np.copy(data_dso_raw)
    data_dso_aligned[:, 0] += time_offset
    np.savetxt(output_file, data_dso_aligned, fmt="%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f")
    print(f"✅ 时间对齐后的新相机轨迹已保存至: {output_file}")
    
    # 绘制滤波 + 平移后的信号验证图
    plt.figure(figsize=(12, 6))
    plt.plot(t_common - t_common[0], w_lio_filtered, label='FAST-LIO (Filtered Base)', color='royalblue', linewidth=2, alpha=0.8)
    
    # 对加上 offset 的原始 dso 信号重新走一遍插值和滤波流程
    interp_dso_aligned = interpolate.interp1d(t_dso_mid + time_offset, w_dso, kind='linear', bounds_error=False, fill_value=0)
    w_dso_aligned_resampled = interp_dso_aligned(t_common)
    w_dso_aligned_filtered = butter_lowpass_filter(w_dso_aligned_resampled, cutoff_freq, fs, order=4)
    w_dso_aligned_filtered -= np.mean(w_dso_aligned_filtered)
    
    plt.plot(t_common - t_common[0], w_dso_aligned_filtered, label=f'Stereo-DSO (Filtered & Shifted {time_offset:.4f}s)', color='darkorange', linestyle='--', linewidth=2, alpha=0.9)
    
    plt.title(f'Verification: Butterworth Low-pass Filtered Alignment (Offset: {time_offset:.4f}s)', fontsize=14)
    plt.xlabel('Common Time [s]', fontsize=12)
    plt.ylabel('Filtered Angular Velocity (Zero-mean)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python3 exact_time_align_butterworth.py <fastlio轨迹.txt> <dso轨迹.txt>")
    else:
        out_file = sys.argv[2].replace(".txt", "_time_aligned.txt")
        main(sys.argv[1], sys.argv[2], out_file)