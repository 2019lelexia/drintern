import cv2
import numpy as np

def convert_to_sgm_format(image_path, save_path):
    # 1. 直接以灰度模式读取 (IMREAD_GRAYSCALE 自动返回 CV_8U)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print("错误：无法加载图片")
        return

    # 2. 如果图像已经是浮点型或其他高位格式，进行归一化并转换
    if img.dtype != np.uint8:
        # 线性拉伸到 0-255 并转为 uint8
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)

    # 3. 确保内存连续 (libSGM 的 CUDA 模块通常要求连续内存)
    if not img.flags['C_CONTIGUOUS']:
        img = np.ascontiguousarray(img)

    cv2.imwrite(save_path, img)
    print(f"转换完成！格式: {img.dtype}, 形状: {img.shape}")

# 使用示例
convert_to_sgm_format('right.png', 'right_sgm.png')
