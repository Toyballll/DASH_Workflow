import numpy as np
from ctypes import *
import time
from typing import Optional, Callable
import threading
import queue


class DASH_Meadowlark_System:
    """
    DASH矫正系统 - 集成Meadowlark SLM
    """

    def __init__(self,
                 slm_sdk_path: str = "C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\SDK\\",
                 pmt_device=None,
                 N_modes: int = 225,
                 f: float = 0.3,
                 P: int = 5):
        """
        初始化DASH系统

        Parameters:
        -----------
        slm_sdk_path : str
            Blink SDK路径
        pmt_device : object
            PMT设备接口
        N_modes : int
            矫正模式数量
        f : float
            功率分配比例
        P : int
            相位步进数
        """
        self.N_modes = N_modes
        self.N = int(np.sqrt(N_modes))  # 假设方形区域
        self.f = f
        self.P = P
        self.pmt = pmt_device

        # 相位步进值
        self.theta = np.linspace(0, 2 * np.pi, P, endpoint=False)

        # 初始化SLM
        self._init_slm(slm_sdk_path)

        # 预计算所有模式
        self._precompute_modes()

        # 初始化校正场
        self.C = np.ones((self.slm_height, self.slm_width), dtype=complex)

        # 性能优化：预分配缓冲区
        self.slm_buffer = np.zeros(self.slm_height * self.slm_width * self.bytes_per_pixel,
                                   dtype=np.uint8, order='C')

    def _init_slm(self, sdk_path):
        """初始化Meadowlark SLM"""
        # 加载DLL
        cdll.LoadLibrary(sdk_path + "Blink_C_wrapper")
        self.slm_lib = CDLL("Blink_C_wrapper")

        # 初始化参数
        bit_depth = c_uint(12)
        num_boards_found = c_uint(0)
        constructed_okay = c_uint(-1)
        is_nematic_type = c_bool(1)
        RAM_write_enable = c_bool(1)
        use_GPU = c_bool(1)
        max_transients = c_uint(20)
        self.board_number = c_uint(1)

        # 创建SDK实例
        self.slm_lib.Create_SDK(
            bit_depth,
            byref(num_boards_found),
            byref(constructed_okay),
            is_nematic_type,
            RAM_write_enable,
            use_GPU,
            max_transients,
            0
        )

        if constructed_okay.value != 0:
            raise RuntimeError("Failed to initialize Blink SDK")

        # 获取SLM参数
        self.slm_height = self.slm_lib.Get_image_height(self.board_number)
        self.slm_width = self.slm_lib.Get_image_width(self.board_number)
        self.bit_depth = self.slm_lib.Get_image_depth(self.board_number)
        self.bytes_per_pixel = self.bit_depth // 8

        print(f"SLM initialized: {self.slm_width}x{self.slm_height}, {self.bit_depth} bits")

        # 加载线性LUT（您应该使用校准后的相位LUT）
        lut_file = self._get_lut_file()
        self.slm_lib.Load_LUT_file(self.board_number, lut_file.encode())

        # 设置SLM写入参数
        self.wait_for_trigger = c_uint(0)
        self.flip_immediate = c_uint(0)
        self.output_pulse_flip = c_uint(0)
        self.output_pulse_refresh = c_uint(0)
        self.timeout_ms = c_uint(5000)

    def _get_lut_file(self):
        """根据SLM尺寸选择LUT文件"""
        base_path = "C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\"

        if self.slm_width == 512:
            if self.bit_depth == 8:
                return base_path + "512x512_linearVoltage.LUT"
            else:
                return base_path + "512x512_16bit_linearVoltage.LUT"
        elif self.slm_width == 1920:
            return base_path + "1920x1152_linearVoltage.LUT"
        elif self.slm_width == 1024:
            return base_path + "1024x1024_linearVoltage.LUT"
        else:
            raise ValueError(f"Unsupported SLM size: {self.slm_width}x{self.slm_height}")

    def _precompute_modes(self):
        """预计算所有平面波光栅模式"""
        # 在矫正区域内生成模式
        self.modes = np.zeros((self.N, self.N, self.N_modes))

        k = np.fft.fftfreq(self.N, 1 / self.N)
        Kx, Ky = np.meshgrid(k, k)

        idx = 0
        for m in range(self.N):
            for n in range(self.N):
                gx = -np.pi + (n + 1) * 2 * np.pi / self.N
                gy = -np.pi + (m + 1) * 2 * np.pi / self.N
                self.modes[:, :, idx] = gx * Kx + gy * Ky
                idx += 1

        print(f"Precomputed {self.N_modes} correction modes")

    def _generate_slm_pattern(self, mode_idx: int, phase_step: int) -> np.ndarray:
        """
        生成SLM相位图案（DASH算法核心）
        """
        # 获取当前模式
        Mn = self.modes[:, :, mode_idx]
        theta_p = self.theta[phase_step]

        # 生成DASH组合场（在矫正区域内）
        modulated = np.sqrt(self.f) * np.exp(1j * (Mn + theta_p))
        reference = np.sqrt(1 - self.f) * np.exp(1j * np.angle(self.C[:self.N, :self.N]))
        combined = modulated + reference

        # 提取相位
        phase_pattern = np.angle(combined)

        # 创建完整SLM图案（如果SLM大于矫正区域）
        full_pattern = np.zeros((self.slm_height, self.slm_width))

        # 将矫正图案放在SLM中心
        y_start = (self.slm_height - self.N) // 2
        x_start = (self.slm_width - self.N) // 2
        full_pattern[y_start:y_start + self.N, x_start:x_start + self.N] = phase_pattern

        return full_pattern

    def _phase_to_gray(self, phase: np.ndarray) -> np.ndarray:
        """
        将相位转换为SLM灰度值
        """
        # 归一化到[0, 1]
        normalized = (phase + np.pi) / (2 * np.pi)

        # 转换为灰度值
        if self.bit_depth == 8:
            gray = (normalized * 255).astype(np.uint8)
        elif self.bit_depth == 12:
            # 12位深度，但通常以16位存储
            gray = (normalized * 4095).astype(np.uint16)
        elif self.bit_depth == 16:
            gray = (normalized * 65535).astype(np.uint16)
        else:
            raise ValueError(f"Unsupported bit depth: {self.bit_depth}")

        return gray

    def _write_to_slm(self, phase_pattern: np.ndarray):
        """
        写入相位图案到SLM
        """
        # 转换为灰度值
        gray = self._phase_to_gray(phase_pattern)

        # 展平为一维数组（C顺序）
        if self.bytes_per_pixel == 1:
            self.slm_buffer[:] = gray.flatten()
        else:  # 16位
            gray_bytes = gray.astype(np.uint16).tobytes()
            self.slm_buffer[:] = np.frombuffer(gray_bytes, dtype=np.uint8)

        # 写入SLM
        ret = self.slm_lib.Write_image(
            self.board_number,
            self.slm_buffer.ctypes.data_as(POINTER(c_ubyte)),
            len(self.slm_buffer),
            self.wait_for_trigger,
            self.flip_immediate,
            self.output_pulse_flip,
            self.output_pulse_refresh,
            self.timeout_ms
        )

        if ret == -1:
            raise RuntimeError("SLM DMA write failed")

        # 等待图像写入完成
        ret = self.slm_lib.ImageWriteComplete(self.board_number, self.timeout_ms)
        if ret == -1:
            raise RuntimeError("SLM ImageWriteComplete failed")

    def _measure_signal(self, accumulation_time_ms: float = 1.0) -> float:
        """
        测量双光子荧光信号
        """
        if self.pmt is None:
            # 模拟信号（用于测试）
            return np.random.poisson(1000)
        else:
            # 实际PMT测量
            return self.pmt.measure(accumulation_time_ms)

    def _update_correction(self, mode_idx: int, signals: np.ndarray):
        """
        根据测量信号更新校正（DASH核心算法）
        """
        # 相位步进干涉测量
        complex_sum = np.sum(np.sqrt(signals) * np.exp(-1j * self.theta))

        # 提取相位和幅度
        phi_n = np.angle(complex_sum)
        r_n = np.sqrt(np.sum(signals))
        a_n = (1 / self.P) * np.abs(complex_sum) / r_n if r_n > 0 else 0

        # 更新校正场（立即更新是DASH的关键）
        Mn = self.modes[:, :, mode_idx]

        # 更新矫正区域
        self.C[:self.N, :self.N] += a_n * np.exp(1j * (Mn - phi_n))

    def run_correction(self,
                       iterations: int = 10,
                       accumulation_time_ms: float = 1.0,
                       save_progress: bool = True):
        """
        运行DASH矫正

        Parameters:
        -----------
        iterations : int
            迭代次数
        accumulation_time_ms : float
            每次测量的信号积累时间（毫秒）
        save_progress : bool
            是否保存进度数据
        """
        print(f"Starting DASH correction: {iterations} iterations, {self.N_modes} modes")

        # 记录增强因子
        enhancement_history = []

        try:
            for iteration in range(iterations):
                print(f"\nIteration {iteration + 1}/{iterations}")
                iteration_start = time.time()

                for mode_idx in range(self.N_modes):
                    # 对每个模式进行P次相位步进
                    signals = np.zeros(self.P)

                    for p in range(self.P):
                        # 生成并写入SLM图案
                        pattern = self._generate_slm_pattern(mode_idx, p)
                        self._write_to_slm(pattern)

                        # 测量信号
                        signals[p] = self._measure_signal(accumulation_time_ms)

                    # 立即更新校正
                    self._update_correction(mode_idx, signals)

                    # 计算当前增强因子（可选）
                    if mode_idx % 10 == 0:
                        current_enhancement = np.mean(signals) / 1000  # 相对于初始信号
                        enhancement_history.append(current_enhancement)
                        print(f"  Mode {mode_idx + 1}/{self.N_modes}, Enhancement: {current_enhancement:.2f}")

                iteration_time = time.time() - iteration_start
                print(f"  Iteration completed in {iteration_time:.1f} seconds")

                # 保存中间结果
                if save_progress:
                    np.save(f'correction_iter_{iteration + 1}.npy', np.angle(self.C))

        except KeyboardInterrupt:
            print("\nCorrection interrupted by user")
        finally:
            # 返回最终校正相位
            final_correction = np.angle(self.C)

            # 保存最终结果
            if save_progress:
                np.save('final_correction.npy', final_correction)
                np.save('enhancement_history.npy', enhancement_history)

            return final_correction, enhancement_history

    def apply_correction_only(self):
        """
        只应用校正（不进行模式测试）
        """
        correction_phase = np.angle(self.C)
        self._write_to_slm(correction_phase)
        print("Correction applied to SLM")

    def cleanup(self):
        """
        清理资源
        """
        if hasattr(self, 'slm_lib'):
            self.slm_lib.Delete_SDK()
            print("SLM SDK cleaned up")


# PMT接口示例（需要根据实际硬件实现）
class PMT_Interface:
    """
    PMT接口示例
    """

    def __init__(self, device_name: str = "Dev1/ai0"):
        # 这里使用NI DAQ作为示例
        # 实际使用时需要安装nidaqmx: pip install nidaqmx
        try:
            import nidaqmx
            self.device = device_name
            self.task = None
        except ImportError:
            print("Warning: nidaqmx not installed, using simulated signals")
            self.device = None

    def measure(self, accumulation_time_ms: float) -> float:
        """
        测量光子计数
        """
        if self.device is None:
            # 模拟信号
            return np.random.poisson(1000 * accumulation_time_ms)
        else:
            # 实际DAQ测量
            import nidaqmx
            with nidaqmx.Task() as task:
                task.ai_channels.add_ai_voltage_chan(self.device)
                task.timing.cfg_samp_clk_timing(
                    rate=10000,  # 10kHz采样率
                    sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                    samps_per_chan=int(10 * accumulation_time_ms)
                )
                data = task.read(number_of_samples_per_channel=int(10 * accumulation_time_ms))

                # 转换为光子计数（需要根据PMT校准）
                photon_count = np.sum(data > 0.5)  # 简单阈值
                return photon_count


# 使用示例
def main():
    """
    主程序
    """
    # 初始化PMT
    pmt = PMT_Interface()

    # 初始化DASH系统
    dash = DASH_Meadowlark_System(
        pmt_device=pmt,
        N_modes=225,  # 15x15模式
        f=0.3,
        P=5
    )

    try:
        # 运行矫正
        correction, history = dash.run_correction(
            iterations=10,
            accumulation_time_ms=1.0,  # 1ms积累时间
            save_progress=True
        )

        # 应用最终校正
        dash.apply_correction_only()

        print("\nCorrection completed!")
        print(f"Final enhancement: {history[-1]:.2f}x")

    finally:
        # 清理
        dash.cleanup()


if __name__ == "__main__":
    main()