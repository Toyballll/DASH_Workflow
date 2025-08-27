import numpy as np
from typing import Callable
import time

class DASHCorrection:
    def __init__(self, N_pixels: int, N_modes: int, f: float = 0.3, P: int = 5):
        """
        初始化DASH矫正系统

        Parameters:
        -----------
        N_pixels : int
            SLM像素数 (总共N x N)
        N_modes : int
            可矫正模式数量 (通常等于像素数)
        f : float
            功率分配比例 (典型值0.3)
        P : int
            相位步进数 (文章中使用5)
        """
        self.N = int(np.sqrt(N_pixels))  # SLM边长
        self.N_pixels = N_pixels
        self.N_modes = N_modes
        self.f = f
        self.P = P

        # 相位步进值 (等间隔分布在0到2π)
        self.theta = np.linspace(0, 2 * np.pi, P, endpoint=False)

        # 初始化校正场 (开始时为平面波)
        self.C = np.ones((self.N, self.N), dtype=complex)

        # 生成模式基底 (平面波光栅)
        self._generate_modes()

    def _generate_modes(self):
        """生成平面波光栅模式基底"""
        self.modes = np.zeros((self.N, self.N, self.N_modes))

        # 创建像素坐标
        k = np.fft.fftfreq(self.N, 1 / self.N)
        Kx, Ky = np.meshgrid(k, k)

        idx = 0
        for m in range(self.N):
            for n in range(self.N):
                # 光栅矢量
                gx = -np.pi + (n + 1) * 2 * np.pi / self.N
                gy = -np.pi + (m + 1) * 2 * np.pi / self.N
                self.modes[:, :, idx] = gx * Kx + gy * Ky
                idx += 1

    def generate_slm_pattern(self, mode_idx: int, phase_step: int) -> np.ndarray:
        """
        生成SLM相位图案

        Parameters:
        -----------
        mode_idx : int
            当前测试的模式索引
        phase_step : int
            当前相位步进索引 (0 到 P-1)

        Returns:
        --------
        np.ndarray
            SLM上要显示的相位图案
        """
        Mn = self.modes[:, :, mode_idx]
        theta_p = self.theta[phase_step]

        # DASH的相位图案生成 (方程1)
        # 将测试模式和参考场全息组合
        modulated_field = np.sqrt(self.f) * np.exp(1j * (Mn + theta_p))
        reference_field = np.sqrt(1 - self.f) * np.exp(1j * np.angle(self.C))

        # 组合场的相位
        combined_field = modulated_field + reference_field
        slm_pattern = np.angle(combined_field)

        return slm_pattern

    def update_correction(self, mode_idx: int, signals: np.ndarray):
        """
        根据测量信号更新校正

        Parameters:
        -----------
        mode_idx : int
            刚测量完的模式索引
        signals : np.ndarray
            P个相位步进的双光子信号 [S0, S1, ..., SP-1]
        """
        # 相位步进干涉测量 (补充材料方程8)
        complex_sum = np.sum(np.sqrt(signals) * np.exp(-1j * self.theta))

        # 提取相位和幅度 (补充材料方程8和9)
        phi_n = np.angle(complex_sum)

        # 计算参考波幅度 (补充材料方程11)
        r_n = np.sqrt(np.sum(signals))

        # 归一化幅度 (补充材料方程9)
        a_n = (1 / self.P) * np.abs(complex_sum) / r_n if r_n > 0 else 0

        # 立即更新校正场 (方程2)
        # 这是DASH的核心创新：立即更新而不是等待完整迭代
        Mn = self.modes[:, :, mode_idx]
        self.C += a_n * np.exp(1j * (Mn - phi_n))

    def run_correction(self, measure_signal: Callable, iterations: int = 10):
        """
        运行完整的DASH矫正

        Parameters:
        -----------
        measure_signal : Callable
            测量双光子信号的函数，输入SLM图案，返回信号强度
        iterations : int
            迭代次数
        """
        measurements_per_iteration = self.N_modes * self.P

        for i in range(iterations):
            print(f"Iteration {i + 1}/{iterations}")

            for mode_idx in range(self.N_modes):
                # 对每个模式进行P次相位步进测量
                signals = np.zeros(self.P)

                for p in range(self.P):
                    # 生成SLM图案
                    slm_pattern = self.generate_slm_pattern(mode_idx, p)

                    # 写入SLM并测量信号
                    # 实际实验中这里需要：
                    # 1. 将slm_pattern写入SLM硬件
                    # 2. 等待SLM稳定（取决于SLM类型）
                    # 3. 从PMT读取双光子荧光信号
                    signals[p] = measure_signal(slm_pattern)

                # 立即更新校正（DASH的关键特征）
                self.update_correction(mode_idx, signals)

                # 可选：显示进度
                if (mode_idx + 1) % 10 == 0:
                    print(f"  Mode {mode_idx + 1}/{self.N_modes} completed")

        # 返回最终校正相位
        return np.angle(self.C)


# 使用示例
def example_usage():
    """演示如何使用DASH矫正"""

    # 初始化DASH系统
    # 15x15 = 225个模式（与文章一致）
    dash = DASHCorrection(N_pixels=225, N_modes=225, f=0.3, P=5)

    # 定义测量函数（实际实验中需要替换为真实的硬件接口）
    def measure_signal(slm_pattern):
        """
        实际实验中这个函数应该：
        1. 将slm_pattern写入SLM
        2. 触发PMT测量
        3. 积分1ms（或5ms用于深层组织）
        4. 返回光子计数
        """
        # 这里只是占位符
        return np.random.poisson(1000)  # 模拟泊松噪声

    # 运行矫正
    final_correction = dash.run_correction(measure_signal, iterations=10)

    return final_correction


# 实际硬件接口示例框架
class HardwareInterface:
    """实际实验的硬件接口"""

    def __init__(self, slm_device, pmt_device):
        self.slm = slm_device
        self.pmt = pmt_device
        self.accumulation_time = 1.0  # ms

    def write_to_slm(self, phase_pattern):
        """写入相位图案到SLM"""
        # 转换为SLM需要的格式（通常是0-255的uint8）
        gray_levels = ((phase_pattern + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
        self.slm.write(gray_levels)

        # 等待SLM稳定
        # 液晶SLM可能需要3-50ms
        # MEMS可能只需要50μs
        time.sleep(0.003)  # 3ms for fast nematic LC-SLM

    def measure_tpef_signal(self):
        """测量双光子荧光信号"""
        # 触发PMT采集
        self.pmt.start_acquisition()

        # 积分指定时间
        time.sleep(self.accumulation_time / 1000)  # 转换为秒

        # 读取光子计数
        photon_count = self.pmt.read_counts()

        return photon_count