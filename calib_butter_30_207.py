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
    rotations = R.from_quat(quats)
    dt = np.diff(times)
    valid = dt > 1e-6
    dt = dt[valid]
    
    rot_diff = rotations[:-1][valid].inv() * rotations[1:][valid]
    omega = rot_diff.magnitude() / dt
    
    t_mid = times[:-1][valid] + dt / 2.0
    return t_mid, omega

def butter_lowpass_filter(data, cutoff, fs, order=4):
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
    
    # 获取两个传感器的绝对公共起始时间
    t_absolute_start = max(t_lio_mid[0], t_dso_mid[0])
    t_absolute_end = min(t_lio_mid[-1], t_dso_mid[-1])
    
    # =========================================================
    # ✂️ 核心修改：只截取 30s 到 207s 的黄金动态区间
    # =========================================================
    crop_start_sec = 30.0
    crop_end_sec = 207.0
    
    # 换算为绝对时间戳
    t_start_cropped = t_absolute_start + crop_start_sec
    t_end_cropped = t_absolute_start + crop_end_sec
    
    # 防止你填写的结束时间超出了实际数据长度
    t_end_cropped = min(t_end_cropped, t_absolute_end)
    
    print(f"✂️ 已裁剪数据：仅保留相对时间 {crop_start_sec}s 到 {t_end_cropped - t_absolute_start:.1f}s 之间的部分进行互相关。")

    if t_start_cropped >= t_end_cropped:
        print("❌ 错误：裁剪区间无效，请检查你设定的时间窗口！")
        sys.exit(1)
        
    dt_res = 0.0001
    fs = 1.0 / dt_res 
    # 现在的公共时间轴只包含 30s ~ 207s 这段黄金区间！
    t_common = np.arange(t_start_cropped, t_end_cropped, dt_res)
    
    # 线性插值
    interp_lio = interpolate.interp1d(t_lio_mid, w_lio, kind='linear', bounds_error=False, fill_value=0)
    interp_dso = interpolate.interp1d(t_dso_mid, w_dso, kind='linear', bounds_error=False, fill_value=0)
    w_lio_resampled = interp_lio(t_common)
    w_dso_resampled = interp_dso(t_common)
    
    # ---------------------------------------------------------
    # 🛡️ 零相移巴特沃斯重型滤波 (Cutoff: 1.0 Hz)
    # ---------------------------------------------------------
    print("🛡️ 正在对黄金区间应用重型低通滤波器...")
    cutoff_freq = 1.0 
    w_lio_filtered = butter_lowpass_filter(w_lio_resampled, cutoff_freq, fs, order=4)
    w_dso_filtered = butter_lowpass_filter(w_dso_resampled, cutoff_freq, fs, order=4)
    
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
    print(f"🎯 黄金区间标定结果：精准时间差 t_offset = {time_offset:.4f} 秒")
    print("=======================================================\n")
    
    # ---------------------------------------------------------
    # 💾 将算出的时间差应用到【整条】DSO 轨迹上并保存
    # ---------------------------------------------------------
    data_dso_raw = np.loadtxt(dso_file)
    data_dso_aligned = np.copy(data_dso_raw)
    data_dso_aligned[:, 0] += time_offset  # 全局漂移补偿
    np.savetxt(output_file, data_dso_aligned, fmt="%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f")
    print(f"✅ 整条相机轨迹已完成时间补偿，并保存至: {output_file}")
    
    # 绘制验证图 (只画截取的部分，看的更清楚)
    plt.figure(figsize=(12, 6))
    
    # 横坐标显示为相对于 0 的相对时间，更符合人的直觉
    plot_time_axis = t_common - t_absolute_start
    
    plt.plot(plot_time_axis, w_lio_filtered, label='FAST-LIO (Filtered Base)', color='royalblue', linewidth=2, alpha=0.8)
    
    # 对 DSO 信号进行平移验证
    interp_dso_aligned = interpolate.interp1d(t_dso_mid + time_offset, w_dso, kind='linear', bounds_error=False, fill_value=0)
    w_dso_aligned_resampled = interp_dso_aligned(t_common)
    w_dso_aligned_filtered = butter_lowpass_filter(w_dso_aligned_resampled, cutoff_freq, fs, order=4)
    w_dso_aligned_filtered -= np.mean(w_dso_aligned_filtered)
    
    plt.plot(plot_time_axis, w_dso_aligned_filtered, label=f'Stereo-DSO (Shifted {time_offset:.4f}s)', color='darkorange', linestyle='--', linewidth=2, alpha=0.9)
    
    plt.title(f'Verification (Window: 30s-207s) | Offset: {time_offset:.4f}s', fontsize=14)
    plt.xlabel('Relative Time from Bag Start [s]', fontsize=12)
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