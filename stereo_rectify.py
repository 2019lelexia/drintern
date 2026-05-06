import cv2
import numpy as np
import os

class OfflineStereoRectifier:
    def __init__(self):
        # ==========================================
        # 1. 传感器内参 (根据你的标定结果)
        # ==========================================
        # Cam0 (Left)
        self.xi_0 = np.array([1.12982056])
        self.D_0 = np.array([0.02107552, -0.07657528, 0.00036545, -0.00053907])
        self.K_0 = np.array([[949.42753588, 0.0, 972.33687702],
                             [0.0, 949.66755126, 587.26231267],
                             [0.0, 0.0, 1.0]])

        # Cam1 (Right)
        self.xi_1 = np.array([1.12437983])
        self.D_1 = np.array([0.01863795, -0.07427101, -0.00013423, 0.00010909])
        self.K_1 = np.array([[945.95142923, 0.0, 966.77167353],
                             [0.0, 946.05602896, 615.18759191],
                             [0.0, 0.0, 1.0]])

        # 相对外参 (由四元数 [x, y, z, w] 转换)
        qx, qy, qz, qw = -0.00176472, 0.64489424, -0.27399369, 0.71346742
        self.R_10 = np.array([
            [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),       1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)]
        ])
        self.t_10 = np.array([-0.05120867, -0.01938352, -0.04599073])

        # ==========================================
        # 2. 计算校正参数
        # ==========================================
        # 计算基线
        t_01 = -self.t_10
        self.baseline = np.linalg.norm(t_01)
        
        # 计算校正旋转矩阵 (让 X 轴指向另一个相机，光轴对齐)
        X_new = t_01 / self.baseline
        Z_bisect = (np.array([0, 0, 1]) + self.R_10.T @ np.array([0, 0, 1])) / 2
        Z_bisect /= np.linalg.norm(Z_bisect)
        Y_new = np.cross(Z_bisect, X_new)
        Y_new /= np.linalg.norm(Y_new)
        Z_new = np.cross(X_new, Y_new)
        Z_new /= np.linalg.norm(Z_new)

        self.R_rect0 = np.vstack((X_new, Y_new, Z_new)).T
        self.R_rect1 = (self.R_rect0 @ self.R_10)

        # 设置校正后的虚拟相机参数 (建议调大分辨率以保留FOV)
        self.new_W, self.new_H = 1280, 720
        self.new_fx, self.new_fy = 400.0, 400.0  # 增大焦距会变窄视角，减小会包含更多内容
        self.new_cx, self.new_cy = self.new_W / 2.0, self.new_H / 2.0
        self.K_new = np.array([[self.new_fx, 0.0, self.new_cx],
                               [0.0, self.new_fy, self.new_cy],
                               [0.0, 0.0, 1.0]])

        # 预计算 Remap 映射表
        print("💡 正在生成校正映射表...")
        self.map_l = cv2.omnidir.initUndistortRectifyMap(
            self.K_0, self.D_0, self.xi_0, self.R_rect0, self.K_new, 
            (self.new_W, self.new_H), cv2.CV_32FC1, cv2.omnidir.RECTIFY_PERSPECTIVE)
        self.map_r = cv2.omnidir.initUndistortRectifyMap(
            self.K_1, self.D_1, self.xi_1, self.R_rect1, self.K_new, 
            (self.new_W, self.new_H), cv2.CV_32FC1, cv2.omnidir.RECTIFY_PERSPECTIVE)

    def process_single_frame(self, img_l_path, img_r_path, save_dir):
        img_l = cv2.imread(img_l_path)
        img_r = cv2.imread(img_r_path)
        
        if img_l is None or img_r is None:
            print("Error: 无法读取图像")
            return

        # 执行校正
        rect_l = cv2.remap(img_l, self.map_l[0], self.map_l[1], cv2.INTER_LINEAR)
        rect_r = cv2.remap(img_r, self.map_r[0], self.map_r[1], cv2.INTER_LINEAR)

        # 保存结果
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        cv2.imwrite(os.path.join(save_dir, "rect_left.png"), rect_l)
        cv2.imwrite(os.path.join(save_dir, "rect_right.png"), rect_r)
        
        # 可视化检查：画极线
        canvas = np.hstack((rect_l, rect_r))
        for i in range(0, canvas.shape[0], 40):
            cv2.line(canvas, (0, i), (canvas.shape[1], i), (0, 255, 0), 1)
        cv2.imshow("Stereo Rectification Check (Epipolar Lines)", canvas)
        cv2.waitKey(0)

if __name__ == "__main__":
    rectifier = OfflineStereoRectifier()
    # 替换为你本地的测试图片路径
    rectifier.process_single_frame("left.png", "right.png", "./output")
