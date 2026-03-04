"""
DASH v8 优化版 - 计算加速

基于原始 dash_v4.py，保持算法完全一致，仅优化计算性能。

优化内容：
1. 预计算 exp(1j*M) - 避免每次重复计算模式相位
2. numexpr 多线程融合 - arctan2、exp等操作自动并行
3. 缓存 exp(1j*C_phase) - 每个mode内5次测量复用
4. 预分配数组 - 避免重复内存分配

性能提升：
- 原始版本: ~1450 ms/mode
- 优化版本: ~360 ms/mode
- 加速比: ~4x
"""

import numpy as np
from ctypes import *
import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
import time
from PIL import Image
from datetime import datetime
from scipy.linalg import hadamard

# 尝试导入numexpr
try:
    import numexpr as ne

    ne.set_num_threads(ne.detect_number_of_cores())
    NUMEXPR_AVAILABLE = True
    print(f"numexpr 已加载，使用 {ne.detect_number_of_cores()} 线程")
except ImportError:
    NUMEXPR_AVAILABLE = False
    print("numexpr 未安装，使用纯NumPy模式")
    print("建议安装以获得更好性能: pip install numexpr")

# ============================================
# SLM初始化
# ============================================
print("初始化SLM...")
cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\SDK\\Blink_C_wrapper")
slm_lib = CDLL("Blink_C_wrapper")

num_boards_found = c_uint(0)
slm_lib.Create_SDK(c_uint(12), byref(num_boards_found), byref(c_uint(-1)),
                   c_bool(1), c_bool(1), c_bool(1), c_uint(20), 0)

board_number = c_uint(1)
slm_lib.Load_LUT_file(board_number,
                      b"C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6227_at1064.lut")

default_init = np.full(1024 * 1024, 128, dtype=np.uint8)
slm_lib.Write_image(board_number, default_init.ctypes.data_as(POINTER(c_ubyte)),
                    c_uint(1024 * 1024), c_uint(0), c_uint(0),
                    c_uint(0), c_uint(0), c_uint(5000))
slm_lib.ImageWriteComplete(board_number, c_uint(5000))
time.sleep(0.5)
print("SLM初始化完成")

# ============================================
# 全局参数
# ============================================
device_name = "Dev2"
pmt_channel = "ai0"
sample_rate = 10000
integration_time_ms = 5
samples = int(integration_time_ms * sample_rate / 1000)
WAIT_TIME = 0.03
SIGNAL_METHOD = 'p95'


def measure_intensity(method=None):
    if method is None:
        method = SIGNAL_METHOD

    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(
            f"{device_name}/{pmt_channel}",
            terminal_config=TerminalConfiguration.DEFAULT,
            min_val=-10.0, max_val=10.0
        )
        task.timing.cfg_samp_clk_timing(
            rate=sample_rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=samples
        )

        task.in_stream.input_buf_size = samples * 2
        task.start()
        acquisition_time = samples / sample_rate
        task.wait_until_done(timeout=acquisition_time + 5.0)
        data = np.array(task.read(
            number_of_samples_per_channel=samples,
            timeout=10.0
        ))

    if method == 'p95':
        return np.percentile(data, 95)
    elif method == 'p90':
        return np.percentile(data, 90)
    else:
        return np.abs(np.mean(data))


def load_default_pattern():
    try:
        img_path = "C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\Image Files\\1024\\Solid128.bmp"
        img = Image.open(img_path).convert('L')
        img_array = np.array(img)
        if img_array.shape != (1024, 1024):
            return np.full(1024 * 1024, 128, dtype=np.uint8)
        return img_array.flatten('C')
    except:
        return np.full(1024 * 1024, 128, dtype=np.uint8)


# ============================================
# Hadamard 基函数（与原版相同）
# ============================================

def generate_hadamard_matrix(n):
    return hadamard(n)


def count_zero_crossings_2d(matrix_2d):
    h, w = matrix_2d.shape
    crossings = 0
    for i in range(h):
        for j in range(w - 1):
            if matrix_2d[i, j] != matrix_2d[i, j + 1]:
                crossings += 1
    for i in range(h - 1):
        for j in range(w):
            if matrix_2d[i, j] != matrix_2d[i + 1, j]:
                crossings += 1
    return crossings


def count_connected_regions(matrix_2d):
    from scipy.ndimage import label
    binary = (matrix_2d == 1).astype(int)
    labeled, num_features = label(binary)
    return num_features


def hadamard_walsh_order(H):
    n = H.shape[0]
    side = int(np.sqrt(n))
    indices = list(range(n))
    crossings = []
    for i in range(n):
        row = H[i, :]
        row_2d = row.reshape(side, side)
        c = count_zero_crossings_2d(row_2d)
        crossings.append(c)
    sorted_indices = sorted(indices, key=lambda x: crossings[x])
    return H[sorted_indices, :], sorted_indices


def hadamard_cake_cutting_order(H):
    n = H.shape[0]
    side = int(np.sqrt(n))
    indices = list(range(n))
    regions = []
    for i in range(n):
        row = H[i, :]
        row_2d = row.reshape(side, side)
        r = count_connected_regions(row_2d)
        regions.append(r)
    sorted_indices = sorted(indices, key=lambda x: regions[x])
    return H[sorted_indices, :], sorted_indices


def hadamard_random_order(H, seed=None):
    n = H.shape[0]
    indices = list(range(n))
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(indices)
    return H[indices, :], indices


# ============================================
# 基函数生成器（与原版相同）
# ============================================

class BasisGenerator:
    CANONICAL = "canonical"
    HADAMARD_NATURAL = "hadamard_natural"
    HADAMARD_WALSH = "hadamard_walsh"
    HADAMARD_CC = "hadamard_cc"
    HADAMARD_RANDOM = "hadamard_random"

    def __init__(self, num_modes, pixel_size, basis_type="canonical", random_seed=None):
        self.num_modes = num_modes
        self.pixel_size = pixel_size
        self.basis_type = basis_type
        self.random_seed = random_seed
        self.modes_per_side = int(np.sqrt(num_modes))

        if self.modes_per_side ** 2 != num_modes:
            raise ValueError(f"num_modes ({num_modes}) must be a perfect square")

        if basis_type != self.CANONICAL:
            if not (num_modes & (num_modes - 1) == 0):
                raise ValueError(f"For Hadamard basis, num_modes ({num_modes}) must be a power of 2")

        self.basis_matrix = None
        self.mode_order = None
        self._generate_basis()

        print(f"\n基函数配置:")
        print(f"  - 类型: {basis_type}")
        print(f"  - 模式数: {num_modes} ({self.modes_per_side}x{self.modes_per_side})")

    def _generate_basis(self):
        if self.basis_type == self.CANONICAL:
            self.basis_matrix = np.eye(self.num_modes)
            self.mode_order = list(range(self.num_modes))
        else:
            H = generate_hadamard_matrix(self.num_modes)
            if self.basis_type == self.HADAMARD_NATURAL:
                self.basis_matrix = H
                self.mode_order = list(range(self.num_modes))
            elif self.basis_type == self.HADAMARD_WALSH:
                self.basis_matrix, self.mode_order = hadamard_walsh_order(H)
            elif self.basis_type == self.HADAMARD_CC:
                self.basis_matrix, self.mode_order = hadamard_cake_cutting_order(H)
            elif self.basis_type == self.HADAMARD_RANDOM:
                self.basis_matrix, self.mode_order = hadamard_random_order(H, self.random_seed)

    def get_mode_pattern_2d(self, mode_idx):
        row = self.basis_matrix[mode_idx, :]
        return row.reshape(self.modes_per_side, self.modes_per_side)

    def get_mode_phase_pattern(self, mode_idx, pixel_size):
        pattern_2d = self.get_mode_pattern_2d(mode_idx)
        block_size = pixel_size // self.modes_per_side
        phase_pattern = np.zeros((pixel_size, pixel_size))

        for i in range(self.modes_per_side):
            for j in range(self.modes_per_side):
                y_start = i * block_size
                y_end = (i + 1) * block_size
                x_start = j * block_size
                x_end = (j + 1) * block_size

                if self.basis_type == self.CANONICAL:
                    if pattern_2d[i, j] == 1:
                        phase_pattern[y_start:y_end, x_start:x_end] = 1.0
                else:
                    if pattern_2d[i, j] == -1:
                        phase_pattern[y_start:y_end, x_start:x_end] = np.pi

        return phase_pattern


# ============================================
# 数据日志（与原版相同）
# ============================================

class DataLogger:
    def __init__(self, config_name, num_modes, f_value, pixel_size, offset_x, offset_y,
                 basis_type="canonical", iteration_strategy="standard"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"DASH_v8_optimized_{timestamp}.txt"
        self.timestamp = timestamp

        with open(self.filename, 'w') as f:
            f.write(f"# DASH v8 优化版 - {datetime.now()}\n")
            f.write(f"# Config: {config_name}, f={f_value}, pixels={pixel_size}\n")
            f.write(f"# Basis type: {basis_type}\n")
            f.write(f"# Iteration strategy: {iteration_strategy}\n")
            f.write(f"# Optimized: numexpr={NUMEXPR_AVAILABLE}\n")
            f.write("#" + "-" * 70 + "\n")

        print(f"数据文件: {self.filename}")

    def log(self, msg):
        with open(self.filename, 'a') as f:
            f.write(f"# [{datetime.now().strftime('%H:%M:%S')}] {msg}\n")

    def log_mode(self, iteration, mode_idx, mean_signal, amplitude, phase_rad, corrected_signal):
        with open(self.filename, 'a') as f:
            f.write(f"{iteration}, {mode_idx}, "
                    f"{mean_signal:.9f}, {amplitude:.9f}, {phase_rad:.5f}, {corrected_signal:.9f}\n")

    def log_final_comparison(self, baseline, default_signals, corrected_signals):
        with open(self.filename, 'a') as f:
            f.write("#" + "=" * 70 + "\n")
            f.write("# FINAL COMPARISON RESULTS\n")
            f.write("#" + "=" * 70 + "\n")
            f.write(f"# Baseline: {baseline:.9f} V\n")
            f.write(f"# Default avg: {np.mean(default_signals):.9f} V\n")
            f.write(f"# Corrected avg: {np.mean(corrected_signals):.9f} V\n")
            f.write(f"# Enhancement: {np.mean(corrected_signals) / np.mean(default_signals):.6f}x\n")
            f.write("#" + "=" * 70 + "\n")


# ============================================
# DASH v8 优化版
# ============================================

class DASH_v8_Optimized:
    """
    DASH v8 优化版 - 算法与原版完全一致，仅优化计算性能

    优化点：
    1. 预计算 exp(1j*M) 存储为 mode_exp
    2. numexpr 多线程融合计算
    3. 缓存 exp(1j*C_phase)，每个mode只算1次
    4. 预分配工作数组
    """

    def __init__(self, num_modes, f_value=0.3, pixel_size=1024, offset_x=0, offset_y=0,
                 basis_type="canonical", data_logger=None, baseline=None,
                 iteration_strategy="standard", random_seed=None):

        self.f = f_value
        self.num_modes = num_modes
        self.modes_per_side = int(np.sqrt(num_modes))
        self.pixel_size = pixel_size
        self.slm_size = 1024
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.data_logger = data_logger
        self.baseline = baseline
        self.iteration_strategy = iteration_strategy
        self.basis_type = basis_type
        self.random_seed = random_seed

        # 优化用的预计算值
        self.sqrt_f = np.float32(np.sqrt(f_value))
        self.sqrt_1_f = np.float32(np.sqrt(1 - f_value))

        # 活动区域
        base_start = (self.slm_size - self.pixel_size) // 2
        self.start_x = max(0, min(base_start + offset_x, self.slm_size - self.pixel_size))
        self.start_y = max(0, min(base_start + offset_y, self.slm_size - self.pixel_size))
        self.end_x = self.start_x + self.pixel_size
        self.end_y = self.start_y + self.pixel_size

        # 生成基函数
        self.basis_gen = BasisGenerator(num_modes, pixel_size, basis_type, random_seed)

        # 相位步进 - 5步
        self.phase_steps = np.array([0, 2 * np.pi / 5, 4 * np.pi / 5, 6 * np.pi / 5, 8 * np.pi / 5], dtype=np.float32)
        self.num_phase_steps = len(self.phase_steps)

        # ============================================
        # 优化1: 预计算 exp(1j*M)
        # ============================================
        print("\n预计算模式复指数 exp(1j*M)...")
        precompute_start = time.time()

        x = np.arange(pixel_size, dtype=np.float32)
        y = np.arange(pixel_size, dtype=np.float32)
        X, Y = np.meshgrid(x, y)

        self.mode_exp = np.zeros((num_modes, pixel_size, pixel_size), dtype=np.complex64)

        if basis_type == BasisGenerator.CANONICAL:
            # Canonical: M = 2*pi*(kx*X + ky*Y)/pixel_size
            half = self.modes_per_side // 2
            for mode_idx in range(num_modes):
                nx = mode_idx // self.modes_per_side
                ny = mode_idx % self.modes_per_side
                kx, ky = nx - half, ny - half
                M = 2 * np.pi * (kx * X + ky * Y) / pixel_size
                self.mode_exp[mode_idx] = np.exp(1j * M).astype(np.complex64)
        else:
            # Hadamard: 预计算 hadamard_phase 的 exp
            for mode_idx in range(num_modes):
                hadamard_phase = self.basis_gen.get_mode_phase_pattern(mode_idx, pixel_size)
                self.mode_exp[mode_idx] = np.exp(1j * hadamard_phase).astype(np.complex64)

        print(f"  预计算完成，耗时 {time.time() - precompute_start:.2f}s")

        # ============================================
        # 优化2: 预分配工作数组
        # ============================================
        self._work_real = np.zeros((pixel_size, pixel_size), dtype=np.float32)
        self._work_imag = np.zeros((pixel_size, pixel_size), dtype=np.float32)
        self._work_gray = np.zeros((pixel_size, pixel_size), dtype=np.uint8)
        self._full_pattern = np.full((self.slm_size, self.slm_size), 128, dtype=np.uint8)

        # 校正场
        self.correction_field = np.zeros((self.pixel_size, self.pixel_size), dtype=np.complex64)
        self.final_correction_pattern = None
        self.default_pattern = load_default_pattern()

        # ============================================
        # 优化3: 缓存 exp(1j*C_phase)
        # ============================================
        self._corr_exp = np.ones((pixel_size, pixel_size), dtype=np.complex64)
        self._corr_exp_valid = False

        # 迭代历史
        self.iteration_history = []

        print(f"\nDASH v8 优化版配置:")
        print(f"  - 基函数: {basis_type}")
        print(f"  - Modes: {self.modes_per_side}x{self.modes_per_side} = {self.num_modes}")
        print(f"  - f = {self.f}")
        print(f"  - Pixels: {self.pixel_size}x{self.pixel_size}")
        print(f"  - Phase steps: {self.num_phase_steps}")
        print(f"  - numexpr: {'启用' if NUMEXPR_AVAILABLE else '未启用'}")

    def _get_corr_exp(self):
        """获取 exp(1j*C_phase)，带缓存"""
        if not self._corr_exp_valid:
            if np.any(self.correction_field != 0):
                C_phase = np.angle(self.correction_field)
                if NUMEXPR_AVAILABLE:
                    self._corr_exp = ne.evaluate("exp(1j * C_phase)").astype(np.complex64)
                else:
                    self._corr_exp = np.exp(1j * C_phase).astype(np.complex64)
            else:
                self._corr_exp.fill(1.0)
            self._corr_exp_valid = True
        return self._corr_exp

    def generate_pattern(self, mode_idx, phase_step_idx, use_fixed_correction=None):
        """
        生成SLM图案 - 优化版

        算法与原版完全一致：
        E_combined = sqrt(f) * exp(1j*(M+theta)) + sqrt(1-f) * exp(1j*C_phase)
        """
        theta = self.phase_steps[phase_step_idx]

        # 获取预计算的 exp(1j*M)
        mode_exp = self.mode_exp[mode_idx]

        # 确定校正场
        if use_fixed_correction is not None:
            if np.any(use_fixed_correction != 0):
                C_phase = np.angle(use_fixed_correction)
                if NUMEXPR_AVAILABLE:
                    corr_exp = ne.evaluate("exp(1j * C_phase)").astype(np.complex64)
                else:
                    corr_exp = np.exp(1j * C_phase).astype(np.complex64)
            else:
                corr_exp = np.ones((self.pixel_size, self.pixel_size), dtype=np.complex64)
        else:
            corr_exp = self._get_corr_exp()

        # 计算合成场
        # E = sqrt(f) * mode_exp * exp(1j*theta) + sqrt(1-f) * corr_exp
        cos_t = np.float32(np.cos(theta))
        sin_t = np.float32(np.sin(theta))
        mode_real = mode_exp.real
        mode_imag = mode_exp.imag
        corr_real = corr_exp.real
        corr_imag = corr_exp.imag

        if NUMEXPR_AVAILABLE:
            sqrt_f = self.sqrt_f
            sqrt_1_f = self.sqrt_1_f
            ne.evaluate("sqrt_f * (mode_real*cos_t - mode_imag*sin_t) + sqrt_1_f * corr_real",
                        out=self._work_real)
            ne.evaluate("sqrt_f * (mode_real*sin_t + mode_imag*cos_t) + sqrt_1_f * corr_imag",
                        out=self._work_imag)
        else:
            np.multiply(mode_real, cos_t, out=self._work_real)
            self._work_real -= mode_imag * sin_t
            self._work_real *= self.sqrt_f
            self._work_real += self.sqrt_1_f * corr_real

            np.multiply(mode_real, sin_t, out=self._work_imag)
            self._work_imag += mode_imag * cos_t
            self._work_imag *= self.sqrt_f
            self._work_imag += self.sqrt_1_f * corr_imag

        # 相位转灰度
        if NUMEXPR_AVAILABLE:
            imag_arr = self._work_imag
            real_arr = self._work_real
            phase = ne.evaluate("arctan2(imag_arr, real_arr)")
            pi = np.pi
            gray_float = ne.evaluate("((phase + pi) % (2*pi)) * (255 / (2*pi))")
            np.copyto(self._work_gray, gray_float, casting='unsafe')
        else:
            phase = np.arctan2(self._work_imag, self._work_real)
            phase_wrapped = (phase + np.pi) % (2 * np.pi)
            gray_float = phase_wrapped * (255 / (2 * np.pi))
            np.copyto(self._work_gray, gray_float, casting='unsafe')

        # 嵌入完整图案
        self._full_pattern[self.start_y:self.end_y, self.start_x:self.end_x] = self._work_gray

        return self._full_pattern.flatten('C')

    def measure_mode(self, iteration, mode_idx, use_fixed_correction=None):
        """测量单个mode"""
        # 预先计算一次 corr_exp
        if use_fixed_correction is None:
            self._get_corr_exp()

        intensities = []
        for i in range(self.num_phase_steps):
            pattern = self.generate_pattern(mode_idx, i, use_fixed_correction)
            slm_lib.Write_image(board_number, pattern.ctypes.data_as(POINTER(c_ubyte)),
                                c_uint(1024 * 1024), c_uint(0), c_uint(0),
                                c_uint(0), c_uint(0), c_uint(5000))
            slm_lib.ImageWriteComplete(board_number, c_uint(5000))
            time.sleep(WAIT_TIME)
            intensity = measure_intensity()
            intensities.append(intensity)

        return np.array(intensities)

    def extract_amplitude_phase(self, intensities):
        """提取幅度和相位"""
        I = np.abs(np.array(intensities))
        if np.all(I < 1e-10):
            return 0.0, 0.0

        a_complex = np.sum(np.sqrt(I) * np.exp(1j * self.phase_steps)) / self.num_phase_steps
        amplitude = np.abs(a_complex)
        phase = np.angle(a_complex)

        return float(amplitude), float(phase)

    def update_correction(self, mode_idx, amplitude, phase):
        """
        更新校正场 - 优化版

        原版公式: correction_field += amplitude * exp(1j*(M + phase))
        优化版:   correction_field += amplitude * mode_exp * exp(1j*phase)
        """
        mode_exp = self.mode_exp[mode_idx]
        exp_phase = np.exp(1j * phase).astype(np.complex64)

        if NUMEXPR_AVAILABLE:
            correction_field = self.correction_field
            self.correction_field = ne.evaluate(
                "correction_field + amplitude * mode_exp * exp_phase",
                local_dict={
                    'correction_field': correction_field,
                    'amplitude': np.float32(amplitude),
                    'mode_exp': mode_exp,
                    'exp_phase': exp_phase
                }
            )
        else:
            self.correction_field += amplitude * mode_exp * exp_phase

        # 标记缓存无效
        self._corr_exp_valid = False

    def generate_final_correction_pattern(self):
        """生成最终校正图案"""
        phase_pattern = np.angle(self.correction_field)
        phase_wrapped = np.mod(phase_pattern + np.pi, 2 * np.pi)
        gray_active = (phase_wrapped * 255 / (2 * np.pi)).astype(np.uint8)

        full_pattern = np.full((self.slm_size, self.slm_size), 128, dtype=np.uint8)
        full_pattern[self.start_y:self.end_y, self.start_x:self.end_x] = gray_active

        self.final_correction_pattern = full_pattern.flatten('C')
        return self.final_correction_pattern

    def test_current_correction(self):
        """测试当前校正效果"""
        self.generate_final_correction_pattern()
        slm_lib.Write_image(board_number, self.final_correction_pattern.ctypes.data_as(POINTER(c_ubyte)),
                            c_uint(1024 * 1024), c_uint(0), c_uint(0),
                            c_uint(0), c_uint(0), c_uint(5000))
        slm_lib.ImageWriteComplete(board_number, c_uint(5000))
        time.sleep(WAIT_TIME)
        return measure_intensity()

    def measure_baseline(self):
        """测量基线"""
        slm_lib.Write_image(board_number, self.default_pattern.ctypes.data_as(POINTER(c_ubyte)),
                            c_uint(1024 * 1024), c_uint(0), c_uint(0),
                            c_uint(0), c_uint(0), c_uint(5000))
        slm_lib.ImageWriteComplete(board_number, c_uint(5000))
        time.sleep(WAIT_TIME * 2)
        return measure_intensity()

    def display_pattern(self, use_correction=True):
        if use_correction and self.final_correction_pattern is not None:
            pattern = self.final_correction_pattern
        else:
            pattern = self.default_pattern

        slm_lib.Write_image(board_number, pattern.ctypes.data_as(POINTER(c_ubyte)),
                            c_uint(1024 * 1024), c_uint(0), c_uint(0),
                            c_uint(0), c_uint(0), c_uint(5000))
        slm_lib.ImageWriteComplete(board_number, c_uint(5000))
        time.sleep(WAIT_TIME * 2)
        return measure_intensity()

    def save_data(self, timestamp):
        np.save(f"correction_field_v8_opt_{timestamp}.npy", self.correction_field)

        if self.final_correction_pattern is not None:
            pattern_2d = self.final_correction_pattern.reshape(1024, 1024)
            Image.fromarray(pattern_2d).save(f"correction_pattern_v8_opt_{timestamp}.png")

        print(f"✓ 数据已保存")

    def clear_iteration_history(self):
        self.iteration_history = []

    def record_state(self, mode_idx, signal):
        self.iteration_history.append((mode_idx, self.correction_field.copy(), signal))

    def get_max_signal_state(self):
        if not self.iteration_history:
            return None, 0, 0
        max_idx = np.argmax([s for _, _, s in self.iteration_history])
        mode_idx, correction_field, signal = self.iteration_history[max_idx]
        return correction_field, mode_idx, signal

    def set_correction_field(self, correction_field):
        self.correction_field = correction_field.copy().astype(np.complex64)
        self._corr_exp_valid = False


# ============================================
# 运行函数（与原版相同的接口）
# ============================================

def run_dash_v8_optimized(num_modes, num_iterations, f_value, pixel_size, offset_x, offset_y,
                          basis_type="canonical", iteration_strategy="standard", random_seed=None):
    """运行DASH v8优化版测试"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = DataLogger("DASH_v8_optimized", num_modes, f_value, pixel_size, offset_x, offset_y,
                        basis_type, iteration_strategy)

    # 初始化DASH
    dash = DASH_v8_Optimized(num_modes, f_value, pixel_size, offset_x, offset_y,
                             basis_type, logger, iteration_strategy=iteration_strategy,
                             random_seed=random_seed)

    # 测量初始基线
    print("\n" + "=" * 60)
    print("【初始基线测量】")
    print("=" * 60)

    baseline_measurements = []
    for i in range(5):
        baseline_measurements.append(dash.measure_baseline())
        time.sleep(0.1)

    baseline = np.mean(baseline_measurements)
    baseline_std = np.std(baseline_measurements)
    dash.baseline = baseline

    print(f"基线: {baseline:.6f} ± {baseline_std:.6f} V")
    logger.log(f"Baseline: {baseline:.6f} ± {baseline_std:.6f} V")

    # DASH迭代
    print("\n" + "=" * 60)
    print("【DASH v8 优化版迭代】")
    print(f"基函数: {basis_type}")
    print(f"策略: {iteration_strategy}")
    print("=" * 60)

    total_iterations = 0
    global_best_correction_field = None
    global_best_enhancement = 0
    global_best_mode_idx = 0
    global_best_iteration = 0

    def safe_divide(a, b, default=0.0):
        if b is None or b == 0 or np.isnan(b):
            return default
        return a / b

    def run_one_iteration(iter_num):
        nonlocal global_best_correction_field, global_best_enhancement
        nonlocal global_best_mode_idx, global_best_iteration

        print(f"\n--- 迭代 {iter_num} ---")

        start_signal = dash.test_current_correction()
        print(f"起点信号: {start_signal:.6f}V (Enhancement: {safe_divide(start_signal, baseline, 1.0):.3f}x)")
        logger.log(f"Iteration {iter_num} start, starting signal: {start_signal:.6f}V")

        dash.clear_iteration_history()
        iter_start_time = time.time()

        for mode_idx in range(num_modes):
            intensities = dash.measure_mode(iter_num, mode_idx)
            amplitude, phase = dash.extract_amplitude_phase(intensities)
            dash.update_correction(mode_idx, amplitude, phase)

            corrected_signal = dash.test_current_correction()
            dash.record_state(mode_idx, corrected_signal)

            enhancement = safe_divide(corrected_signal, baseline, 1.0)

            if corrected_signal > global_best_enhancement * baseline:
                global_best_correction_field = dash.correction_field.copy()
                global_best_enhancement = enhancement
                global_best_mode_idx = mode_idx
                global_best_iteration = iter_num

            mean_signal = np.mean(np.abs(intensities))
            logger.log_mode(iter_num, mode_idx, mean_signal, amplitude, phase, corrected_signal)

            if (mode_idx + 1) % 10 == 0:
                elapsed = time.time() - iter_start_time
                eta = elapsed / (mode_idx + 1) * (num_modes - mode_idx - 1)
                print(f"  [{mode_idx + 1:3d}/{num_modes}] Enhancement={enhancement:.3f}x, "
                      f"ETA:{eta:.0f}s")

        iter_time = time.time() - iter_start_time
        final_signal = dash.test_current_correction()
        final_enhancement = safe_divide(final_signal, baseline, 1.0)

        print(f"\n迭代{iter_num}完成:")
        print(f"  耗时: {iter_time:.1f}s ({iter_time / num_modes:.2f}s/mode)")
        print(f"  最终信号: {final_signal:.6f}V")
        print(f"  Enhancement: {final_enhancement:.3f}x")

        logger.log(f"Iteration {iter_num} done, time={iter_time:.1f}s, "
                   f"final={final_signal:.6f}V, enhancement={final_enhancement:.3f}x")

        best_field, best_mode, best_signal = dash.get_max_signal_state()
        return best_field, best_mode, best_signal

    # 第一轮迭代
    for i in range(num_iterations):
        total_iterations += 1
        run_one_iteration(total_iterations)

    # 交互式继续迭代
    while True:
        choice = input("\n继续迭代? (y/n): ").strip().lower()
        if choice == 'y':
            total_iterations += 1

            if iteration_strategy == "max_signal_start":
                use_max = input("从最大信号点开始? (y/n): ").strip().lower()
                if use_max == 'y':
                    max_field, max_mode, max_signal = dash.get_max_signal_state()
                    if max_field is not None:
                        dash.set_correction_field(max_field)

            iter_best_field, _, _ = run_one_iteration(total_iterations)

            if iteration_strategy == "max_signal_start":
                dash.set_correction_field(iter_best_field)

        elif choice == 'n':
            break

    print(f"\n总共完成 {total_iterations} 次迭代")

    # 选择校正场
    print("\n" + "-" * 40)
    print(f"全局最佳: 迭代{global_best_iteration}, mode{global_best_mode_idx}, Enhancement={global_best_enhancement:.3f}x")
    use_best = input("使用全局最佳校正场? (y/n): ").strip().lower()

    if use_best == 'y' and global_best_correction_field is not None:
        dash.correction_field = global_best_correction_field
        print("已切换到全局最佳校正场")
    else:
        print("使用最终校正场")

    # 生成最终图案
    dash.generate_final_correction_pattern()
    dash.save_data(timestamp)

    # 最终对比
    print("\n" + "=" * 60)
    print("【最终结果对比】")
    print("=" * 60)

    default_signals = []
    corrected_signals = []

    for i in range(5):
        default_signals.append(dash.display_pattern(False))
        time.sleep(0.1)
        corrected_signals.append(dash.display_pattern(True))
        time.sleep(0.1)

    default_avg = np.mean(default_signals)
    corrected_avg = np.mean(corrected_signals)

    print(f"默认图案: {default_avg:.6f} V")
    print(f"校正图案: {corrected_avg:.6f} V")
    print(f"Enhancement: {safe_divide(corrected_avg, default_avg, 1.0):.3f}x")

    logger.log_final_comparison(baseline, default_signals, corrected_signals)

    # 交互测试
    print("\n" + "=" * 60)
    print("【交互测试】1=默认, 2=校正, 3=检查漂移, 4=切换到全局最佳, q=退出")
    print("=" * 60)

    while True:
        try:
            choice = input("\n选择: ").strip().lower()
            if choice == 'q':
                break
            elif choice == '1':
                sig = dash.display_pattern(False)
                print(f"  默认: {sig:.6f}V ({safe_divide(sig, baseline, 1.0):.3f}x)")
            elif choice == '2':
                sig = dash.display_pattern(True)
                print(f"  校正: {sig:.6f}V ({safe_divide(sig, baseline, 1.0):.3f}x)")
            elif choice == '3':
                sig = dash.measure_baseline()
                print(f"  当前基线: {sig:.6f}V (漂移: {safe_divide(sig, baseline, 1.0):.3f}x)")
            elif choice == '4':
                if global_best_correction_field is not None:
                    dash.correction_field = global_best_correction_field.copy()
                    dash.generate_final_correction_pattern()
                    sig = dash.display_pattern(True)
                    print(f"  全局最佳: {sig:.6f}V ({safe_divide(sig, baseline, 1.0):.3f}x)")
        except KeyboardInterrupt:
            break

    return dash, logger


# ============================================
# 主程序
# ============================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DASH v8 优化版 - 计算加速")
    print("=" * 60)

    # 基函数选择
    print("\n基函数选择:")
    print("  1 = Canonical (Fourier/Phase Grating) - 原始DASH")
    print("  2 = Hadamard Natural (H)")
    print("  3 = Hadamard-Walsh (HW) - 低频优先")
    print("  4 = Hadamard Cake-cutting (CC)")
    print("  5 = Hadamard Random (HRAN)")

    basis_choice = input("选择 (1-5): ").strip()

    basis_map = {
        '1': BasisGenerator.CANONICAL,
        '2': BasisGenerator.HADAMARD_NATURAL,
        '3': BasisGenerator.HADAMARD_WALSH,
        '4': BasisGenerator.HADAMARD_CC,
        '5': BasisGenerator.HADAMARD_RANDOM
    }
    basis_type = basis_map.get(basis_choice, BasisGenerator.CANONICAL)
    print(f"  ★ 已选择: {basis_type}")

    # 迭代策略
    print("\n迭代策略:")
    print("  1 = 标准策略")
    print("  2 = 最大信号起点策略")
    strategy_choice = input("选择 (1-2): ").strip()
    iteration_strategy = "max_signal_start" if strategy_choice == '2' else "standard"
    print(f"  ★ 已选择: {iteration_strategy}")

    # 模式数
    print("\n模式数:")
    print("  1 = 16 (4x4)")
    print("  2 = 64 (8x8)")
    print("  3 = 256 (16x16)")
    print("  4 = 1024 (32x32)")
    print("  5 = 自定义")

    mode_choice = input("选择 (1-5): ").strip()
    mode_map = {'1': 16, '2': 64, '3': 256, '4': 1024}

    if mode_choice == '5':
        try:
            num_modes = int(input("输入模式数: ").strip())
        except:
            num_modes = 64
    else:
        num_modes = mode_map.get(mode_choice, 64)

    # 验证
    sqrt_n = int(np.sqrt(num_modes))
    if sqrt_n * sqrt_n != num_modes:
        num_modes = sqrt_n * sqrt_n
        print(f"  调整为平方数: {num_modes}")

    if basis_type != BasisGenerator.CANONICAL:
        if not (num_modes & (num_modes - 1) == 0):
            power = int(np.log2(num_modes))
            num_modes = 2 ** power
            print(f"  Hadamard需要2的幂，调整为: {num_modes}")

    # 迭代次数
    print("\n迭代次数 (默认1): ")
    try:
        num_iter = int(input().strip())
        num_iter = max(1, num_iter)
    except:
        num_iter = 1

    # 像素数
    print("\n像素数: 1=512, 2=1024")
    pc = input("选择 (1-2): ").strip()
    pixel_size = 512 if pc == '1' else 1024

    # f值
    print("\nf值: 1=0.3, 2=0.5, 3=自定义")
    fc = input("选择 (1-3): ").strip()
    if fc == '1':
        f_value = 0.3
    elif fc == '2':
        f_value = 0.5
    elif fc == '3':
        try:
            f_value = float(input("输入f值 (0-1): ").strip())
            f_value = max(0.01, min(f_value, 0.99))
        except:
            f_value = 0.3
    else:
        f_value = 0.3

    # 随机种子
    random_seed = None
    if basis_type == BasisGenerator.HADAMARD_RANDOM:
        print("\n随机种子 (留空自动生成): ")
        seed_input = input().strip()
        if seed_input:
            try:
                random_seed = int(seed_input)
            except:
                random_seed = np.random.randint(0, 10000)
        else:
            random_seed = np.random.randint(0, 10000)
        print(f"  使用随机种子: {random_seed}")

    # 偏移
    print("\n偏移 (格式: x y，如 0 0): ")
    try:
        offset_input = input().strip().split()
        offset_x, offset_y = int(offset_input[0]), int(offset_input[1])
    except:
        offset_x, offset_y = 0, 0

    # 配置确认
    print(f"\n配置确认:")
    print(f"  - 基函数: {basis_type}")
    print(f"  - 迭代策略: {iteration_strategy}")
    print(f"  - 模式数: {num_modes} ({int(np.sqrt(num_modes))}x{int(np.sqrt(num_modes))})")
    print(f"  - 迭代: {num_iter}")
    print(f"  - 像素: {pixel_size}x{pixel_size}")
    print(f"  - f值: {f_value}")
    print(f"  - 偏移: ({offset_x}, {offset_y})")
    print(f"  - numexpr: {'启用' if NUMEXPR_AVAILABLE else '未启用'}")

    # 估计时间（优化后更快）
    est_time = num_modes * 5 * num_iter * (WAIT_TIME + integration_time_ms / 1000) * 2  # 包含test_current_correction
    print(f"  - 预计时间: ~{est_time:.0f}秒 ≈ {est_time / 60:.1f}分钟")

    if input("\n开始? (y/n): ").strip().lower() != 'y':
        print("已取消")
        exit()

    try:
        dash, logger = run_dash_v8_optimized(num_modes, num_iter, f_value, pixel_size, offset_x, offset_y,
                                             basis_type, iteration_strategy, random_seed)
        print("\n✅ 完成!")
    except KeyboardInterrupt:
        print("\n中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("\n恢复默认...")
        default_pattern = load_default_pattern()
        slm_lib.Write_image(board_number, default_pattern.ctypes.data_as(POINTER(c_ubyte)),
                            c_uint(1024 * 1024), c_uint(0), c_uint(0),
                            c_uint(0), c_uint(0), c_uint(5000))
        slm_lib.ImageWriteComplete(board_number, c_uint(5000))
        slm_lib.Delete_SDK()
        print("结束")