"""
DASH v8 - 添加多种基函数选择

新增功能：
1. ✅ 基函数选择：
   - Canonical (Fourier/Phase Grating) - 原始DASH使用的基函数
   - Hadamard Natural (H) - 自然排序的Hadamard基
   - Hadamard-Walsh (HW) - Walsh排序，按零交叉数排序（低通滤波效果）
   - Hadamard Cake-cutting (CC) - 按2D连通区域排序
   - Hadamard Random (HRAN) - 随机排序

2. ✅ 迭代策略选择：
   - 标准策略：从上一迭代的最终状态继续
   - 最大信号起点策略：从上一迭代中信号最大时的状态开始

原有功能：
- 可选随机/顺序测量模式
- 趋势验证
- f值、mode数量、像素数支持自定义输入
"""

import numpy as np
from ctypes import *
import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
import time
from PIL import Image
from datetime import datetime
from scipy.linalg import hadamard

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
# Hadamard 基函数生成和排序
# ============================================

def generate_hadamard_matrix(n):
    """生成 n x n Hadamard 矩阵 (n 必须是2的幂)"""
    return hadamard(n)


def count_zero_crossings_1d(row):
    """计算1D向量的零交叉数"""
    crossings = 0
    for i in range(len(row) - 1):
        if row[i] != row[i + 1]:
            crossings += 1
    return crossings


def count_zero_crossings_2d(matrix_2d):
    """计算2D矩阵的零交叉数（水平+垂直）"""
    h, w = matrix_2d.shape
    crossings = 0
    # 水平方向
    for i in range(h):
        for j in range(w - 1):
            if matrix_2d[i, j] != matrix_2d[i, j + 1]:
                crossings += 1
    # 垂直方向
    for i in range(h - 1):
        for j in range(w):
            if matrix_2d[i, j] != matrix_2d[i + 1, j]:
                crossings += 1
    return crossings


def count_connected_regions(matrix_2d):
    """计算2D矩阵中+1连通区域的数量（用于cake-cutting排序）"""
    from scipy.ndimage import label
    binary = (matrix_2d == 1).astype(int)
    labeled, num_features = label(binary)
    return num_features


def hadamard_walsh_order(H):
    """
    Hadamard-Walsh 排序：按零交叉数（sequency）排序
    零交叉数少的排在前面，相当于低频优先
    """
    n = H.shape[0]
    side = int(np.sqrt(n))

    indices = list(range(n))

    # 计算每一行的零交叉数
    crossings = []
    for i in range(n):
        row = H[i, :]
        # 转换为2D来计算
        row_2d = row.reshape(side, side)
        c = count_zero_crossings_2d(row_2d)
        crossings.append(c)

    # 按零交叉数排序
    sorted_indices = sorted(indices, key=lambda x: crossings[x])

    return H[sorted_indices, :], sorted_indices


def hadamard_cake_cutting_order(H):
    """
    Cake-cutting 排序：按2D表示中+1连通区域数量排序
    连通区域少的排在前面
    """
    n = H.shape[0]
    side = int(np.sqrt(n))

    indices = list(range(n))

    # 计算每一行的连通区域数
    regions = []
    for i in range(n):
        row = H[i, :]
        row_2d = row.reshape(side, side)
        r = count_connected_regions(row_2d)
        regions.append(r)

    # 按连通区域数排序
    sorted_indices = sorted(indices, key=lambda x: regions[x])

    return H[sorted_indices, :], sorted_indices


def hadamard_random_order(H, seed=None):
    """随机排序"""
    n = H.shape[0]
    indices = list(range(n))

    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(indices)

    return H[indices, :], indices


class BasisGenerator:
    """基函数生成器"""

    CANONICAL = "canonical"
    HADAMARD_NATURAL = "hadamard_natural"
    HADAMARD_WALSH = "hadamard_walsh"
    HADAMARD_CC = "hadamard_cc"
    HADAMARD_RANDOM = "hadamard_random"

    def __init__(self, num_modes, pixel_size, basis_type="canonical", random_seed=None):
        """
        Args:
            num_modes: 模式数量 (必须是平方数，对于Hadamard还需要是2的幂)
            pixel_size: SLM上的像素大小
            basis_type: 基函数类型
            random_seed: 随机种子（用于HADAMARD_RANDOM）
        """
        self.num_modes = num_modes
        self.pixel_size = pixel_size
        self.basis_type = basis_type
        self.random_seed = random_seed
        self.modes_per_side = int(np.sqrt(num_modes))

        # 验证
        if self.modes_per_side ** 2 != num_modes:
            raise ValueError(f"num_modes ({num_modes}) must be a perfect square")

        # 对于Hadamard，还需要是2的幂
        if basis_type != self.CANONICAL:
            if not (num_modes & (num_modes - 1) == 0):
                raise ValueError(f"For Hadamard basis, num_modes ({num_modes}) must be a power of 2")

        # 生成基函数
        self.basis_matrix = None  # N x N 矩阵
        self.mode_order = None  # 模式顺序
        self._generate_basis()

        print(f"\n基函数配置:")
        print(f"  - 类型: {basis_type}")
        print(f"  - 模式数: {num_modes} ({self.modes_per_side}x{self.modes_per_side})")

    def _generate_basis(self):
        """生成基函数矩阵"""
        if self.basis_type == self.CANONICAL:
            # Canonical basis: 单位矩阵（每个元素只激活一个像素块）
            self.basis_matrix = np.eye(self.num_modes)
            self.mode_order = list(range(self.num_modes))

        else:
            # Hadamard 基函数
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
        """
        获取第 mode_idx 个模式的2D图案

        对于 Canonical: 返回一个只有一个块为1，其他为0的图案
        对于 Hadamard: 返回 ±1 的图案

        Returns:
            2D numpy array of shape (modes_per_side, modes_per_side)
        """
        row = self.basis_matrix[mode_idx, :]
        return row.reshape(self.modes_per_side, self.modes_per_side)

    def get_mode_phase_pattern(self, mode_idx, pixel_size):
        """
        获取模式的相位图案（用于SLM显示）

        对于 Canonical: 返回相位光栅（只在激活区域）
        对于 Hadamard: +1 -> 0相位, -1 -> π相位

        Returns:
            2D numpy array of shape (pixel_size, pixel_size)
        """
        pattern_2d = self.get_mode_pattern_2d(mode_idx)

        # 上采样到 pixel_size x pixel_size
        block_size = pixel_size // self.modes_per_side

        phase_pattern = np.zeros((pixel_size, pixel_size))

        for i in range(self.modes_per_side):
            for j in range(self.modes_per_side):
                y_start = i * block_size
                y_end = (i + 1) * block_size
                x_start = j * block_size
                x_end = (j + 1) * block_size

                if self.basis_type == self.CANONICAL:
                    # Canonical: 只有值为1的块有相位
                    if pattern_2d[i, j] == 1:
                        phase_pattern[y_start:y_end, x_start:x_end] = 1.0  # 标记为激活
                else:
                    # Hadamard: -1 对应 π 相位
                    if pattern_2d[i, j] == -1:
                        phase_pattern[y_start:y_end, x_start:x_end] = np.pi
                    # +1 对应 0 相位，已经是0了

        return phase_pattern


class DataLogger:
    def __init__(self, config_name, num_modes, f_value, pixel_size, offset_x, offset_y,
                 basis_type="canonical", iteration_strategy="standard"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"DASH_v8_{timestamp}.txt"
        self.timestamp = timestamp

        with open(self.filename, 'w') as f:
            f.write(f"# DASH v8 - {datetime.now()}\n")
            f.write(f"# Config: {config_name}, f={f_value}, pixels={pixel_size}\n")
            f.write(f"# Basis type: {basis_type}\n")
            f.write(f"# Iteration strategy: {iteration_strategy}\n")
            f.write(f"# [v8] Multiple basis functions support\n")
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


class DASH_v8:
    """
    DASH v8 - 支持多种基函数
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

        # 活动区域
        base_start = (self.slm_size - self.pixel_size) // 2
        self.start_x = max(0, min(base_start + offset_x, self.slm_size - self.pixel_size))
        self.start_y = max(0, min(base_start + offset_y, self.slm_size - self.pixel_size))
        self.end_x = self.start_x + self.pixel_size
        self.end_y = self.start_y + self.pixel_size

        # 生成基函数
        self.basis_gen = BasisGenerator(num_modes, pixel_size, basis_type, random_seed)

        # 使用像素索引坐标 (用于Canonical/Fourier基)
        x = np.arange(self.pixel_size)
        y = np.arange(self.pixel_size)
        self.X, self.Y = np.meshgrid(x, y)

        # 相位步进 - 5步
        self.phase_steps = np.array([0, 2 * np.pi / 5, 4 * np.pi / 5, 6 * np.pi / 5, 8 * np.pi / 5])
        self.num_phase_steps = len(self.phase_steps)

        # 校正场初始化
        self.correction_field = np.zeros((self.pixel_size, self.pixel_size), dtype=complex)
        self.final_correction_pattern = None
        self.default_pattern = load_default_pattern()

        # 迭代历史
        self.iteration_history = []

        print(f"\nDASH v8 配置:")
        print(f"  - 基函数: {basis_type}")
        print(f"  - Modes: {self.modes_per_side}x{self.modes_per_side} = {self.num_modes}")
        print(f"  - f = {self.f}")
        print(f"  - Pixels: {self.pixel_size}x{self.pixel_size}")
        print(f"  - Phase steps: {self.num_phase_steps}")
        print(f"  - Iteration strategy: {iteration_strategy}")

    def _get_centered_k(self, mode_idx):
        """对于Canonical基，获取中心化的k向量"""
        nx = mode_idx // self.modes_per_side
        ny = mode_idx % self.modes_per_side
        half = self.modes_per_side // 2
        return nx - half, ny - half

    def _compute_canonical_mode_phase(self, kx, ky):
        """计算Canonical（Fourier）模式相位"""
        return 2 * np.pi * (kx * self.X + ky * self.Y) / self.pixel_size

    def generate_pattern(self, mode_idx, phase_step_idx, use_fixed_correction=None):
        """
        生成SLM图案

        对于Canonical基：使用相位光栅
        对于Hadamard基：使用Hadamard模式（±1 -> 0/π相位）
        """
        theta = self.phase_steps[phase_step_idx]

        # 确定使用哪个校正场
        if use_fixed_correction is not None:
            correction_field = use_fixed_correction
        else:
            correction_field = self.correction_field

        if np.any(correction_field != 0):
            C_phase = np.angle(correction_field)
        else:
            C_phase = np.zeros((self.pixel_size, self.pixel_size))

        if self.basis_type == BasisGenerator.CANONICAL:
            # Canonical: 使用相位光栅
            kx, ky = self._get_centered_k(mode_idx)
            M = self._compute_canonical_mode_phase(kx, ky)

            # DASH的phase-only实现
            E_combined = np.sqrt(self.f) * np.exp(1j * (M + theta)) + \
                         np.sqrt(1 - self.f) * np.exp(1j * C_phase)

        else:
            # Hadamard: 使用Hadamard模式
            hadamard_phase = self.basis_gen.get_mode_phase_pattern(mode_idx, self.pixel_size)

            # Hadamard模式作为调制波前
            # +1 区域: 相位 = theta
            # -1 区域: 相位 = theta + π
            M = hadamard_phase + theta

            E_combined = np.sqrt(self.f) * np.exp(1j * M) + \
                         np.sqrt(1 - self.f) * np.exp(1j * C_phase)

        phase_pattern = np.angle(E_combined)

        # 转换为灰度 [0, 255]
        phase_wrapped = np.mod(phase_pattern + np.pi, 2 * np.pi)
        gray_active = (phase_wrapped * 255 / (2 * np.pi)).astype(np.uint8)

        # 嵌入完整图案
        full_pattern = np.full((self.slm_size, self.slm_size), 128, dtype=np.uint8)
        full_pattern[self.start_y:self.end_y, self.start_x:self.end_x] = gray_active

        return full_pattern.flatten('C')

    def measure_mode(self, iteration, mode_idx, use_fixed_correction=None):
        """测量单个mode的相位步进响应"""
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

        a_complex = np.sum(np.sqrt(I) * np.exp(+1j * self.phase_steps)) / self.num_phase_steps

        amplitude = np.abs(a_complex)
        phase = np.angle(a_complex)

        return amplitude, phase

    def update_correction(self, mode_idx, amplitude, phase):
        """更新校正场"""
        if self.basis_type == BasisGenerator.CANONICAL:
            # Canonical: 使用相位光栅
            kx, ky = self._get_centered_k(mode_idx)
            M = self._compute_canonical_mode_phase(kx, ky)
            self.correction_field += amplitude * np.exp(1j * (M + phase))
        else:
            # Hadamard: 使用Hadamard模式
            hadamard_phase = self.basis_gen.get_mode_phase_pattern(mode_idx, self.pixel_size)
            self.correction_field += amplitude * np.exp(1j * (hadamard_phase + phase))

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
        np.save(f"correction_field_v8_{timestamp}.npy", self.correction_field)

        if self.final_correction_pattern is not None:
            pattern_2d = self.final_correction_pattern.reshape(1024, 1024)
            Image.fromarray(pattern_2d).save(f"correction_pattern_v8_{timestamp}.png")

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
        self.correction_field = correction_field.copy()


def run_dash_v8(num_modes, num_iterations, f_value, pixel_size, offset_x, offset_y,
                basis_type="canonical", iteration_strategy="standard", random_seed=None):
    """运行DASH v8测试"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = DataLogger("DASH_v8", num_modes, f_value, pixel_size, offset_x, offset_y,
                        basis_type, iteration_strategy)

    # 初始化DASH
    dash = DASH_v8(num_modes, f_value, pixel_size, offset_x, offset_y,
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
    print("【DASH v8 迭代】")
    print(f"基函数: {basis_type}")
    print(f"策略: {iteration_strategy}")
    print("=" * 60)

    total_iterations = 0

    # 全局最佳记录
    global_best_correction_field = None
    global_best_enhancement = 0
    global_best_mode_idx = 0
    global_best_iteration = 0

    def run_one_iteration(iter_num):
        nonlocal global_best_correction_field, global_best_enhancement
        nonlocal global_best_mode_idx, global_best_iteration

        print(f"\n--- 迭代 {iter_num} ---")

        start_signal = dash.test_current_correction()
        print(f"起点信号: {start_signal:.6f}V (Enhancement: {start_signal / baseline:.3f}x)")
        logger.log(f"Iteration {iter_num} start, starting signal: {start_signal:.6f}V")

        dash.clear_iteration_history()

        iter_best_enhancement = start_signal / baseline
        iter_best_mode_idx = 0
        iter_best_correction_field = dash.correction_field.copy()

        for mode_idx in range(num_modes):
            # 测量模式
            intensities = dash.measure_mode(iter_num - 1, mode_idx)
            mean_signal = np.mean(intensities)

            # 提取幅度和相位
            amplitude, phase = dash.extract_amplitude_phase(intensities)

            # 更新校正
            dash.update_correction(mode_idx, amplitude, phase)

            # 测试校正效果
            corrected_signal = dash.test_current_correction()
            enhancement = corrected_signal / baseline if baseline > 0 else 0

            # 记录状态
            dash.record_state(mode_idx, corrected_signal)

            # 记录到日志
            logger.log_mode(iter_num - 1, mode_idx, mean_signal, amplitude, phase, corrected_signal)

            # 更新最佳记录
            if enhancement > iter_best_enhancement:
                iter_best_enhancement = enhancement
                iter_best_mode_idx = mode_idx
                iter_best_correction_field = dash.correction_field.copy()

            if enhancement > global_best_enhancement:
                global_best_enhancement = enhancement
                global_best_correction_field = dash.correction_field.copy()
                global_best_mode_idx = mode_idx
                global_best_iteration = iter_num

            # 每25%输出进度
            if (mode_idx + 1) % max(1, num_modes // 4) == 0:
                print(f"  [{(mode_idx + 1) * 100 / num_modes:5.1f}%] mode={mode_idx:3d} | "
                      f"Enh: {enhancement:.3f}x | iter_best: {iter_best_enhancement:.3f}x")

        final_enh = dash.test_current_correction() / baseline
        print(f"\n  迭代{iter_num}完成:")
        print(f"    最终信号: Enh={final_enh:.3f}x")
        print(f"    本迭代最佳: Enh={iter_best_enhancement:.3f}x (mode {iter_best_mode_idx})")
        print(f"    全局最佳: Enh={global_best_enhancement:.3f}x (iter{global_best_iteration}, mode{global_best_mode_idx})")

        logger.log(f"Iteration {iter_num} done: final={final_enh:.3f}x, "
                   f"iter_best={iter_best_enhancement:.3f}x, global_best={global_best_enhancement:.3f}x")

        return iter_best_correction_field, iter_best_enhancement, iter_best_mode_idx

    # 执行迭代
    for iteration in range(num_iterations):
        total_iterations += 1
        iter_best_field, iter_best_enh, iter_best_mode = run_one_iteration(total_iterations)

        # 根据策略设置下一次迭代起点
        if iteration < num_iterations - 1:
            if iteration_strategy == "max_signal_start":
                print(f"\n  [策略: max_signal_start] 从本迭代最佳点开始下一迭代")
                dash.set_correction_field(iter_best_field)
            else:
                print(f"\n  [策略: standard] 从当前最终状态继续")

    # 询问是否继续
    while True:
        print("\n" + "-" * 40)
        choice = input("继续迭代? (y=继续, n=结束): ").strip().lower()
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
    print(f"Enhancement: {corrected_avg / default_avg:.3f}x")

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
                print(f"  默认: {sig:.6f}V ({sig / baseline:.3f}x)")
            elif choice == '2':
                sig = dash.display_pattern(True)
                print(f"  校正: {sig:.6f}V ({sig / baseline:.3f}x)")
            elif choice == '3':
                sig = dash.measure_baseline()
                print(f"  当前基线: {sig:.6f}V (漂移: {sig / baseline:.3f}x)")
            elif choice == '4':
                if global_best_correction_field is not None:
                    dash.correction_field = global_best_correction_field.copy()
                    dash.generate_final_correction_pattern()
                    sig = dash.display_pattern(True)
                    print(f"  全局最佳: {sig:.6f}V ({sig / baseline:.3f}x)")
        except KeyboardInterrupt:
            break

    return dash, logger


# ============================================
# 主程序
# ============================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DASH v8 - 多基函数支持")
    print("=" * 60)

    # 基函数选择
    print("\n基函数选择:")
    print("  1 = Canonical (Fourier/Phase Grating) - 原始DASH")
    print("  2 = Hadamard Natural (H)")
    print("  3 = Hadamard-Walsh (HW) - 低频优先 [推荐用于稀疏像差]")
    print("  4 = Hadamard Cake-cutting (CC) - 连通区域优先")
    print("  5 = Hadamard Random (HRAN)")

    basis_choice = input("选择 (1-5): ").strip()

    if basis_choice == '1':
        basis_type = BasisGenerator.CANONICAL
    elif basis_choice == '2':
        basis_type = BasisGenerator.HADAMARD_NATURAL
    elif basis_choice == '3':
        basis_type = BasisGenerator.HADAMARD_WALSH
    elif basis_choice == '4':
        basis_type = BasisGenerator.HADAMARD_CC
    elif basis_choice == '5':
        basis_type = BasisGenerator.HADAMARD_RANDOM
    else:
        basis_type = BasisGenerator.CANONICAL

    print(f"  ★ 已选择: {basis_type}")

    # 迭代策略
    print("\n迭代策略:")
    print("  1 = 标准策略")
    print("  2 = 最大信号起点策略")
    strategy_choice = input("选择 (1-2): ").strip()

    if strategy_choice == '2':
        iteration_strategy = "max_signal_start"
    else:
        iteration_strategy = "standard"

    print(f"  ★ 已选择: {iteration_strategy}")

    # 模式数
    print("\n模式数 (对于Hadamard必须是2的幂的平方数):")
    print("  1 = 16 (4x4)")
    print("  2 = 64 (8x8) [推荐]")
    print("  3 = 256 (16x16)")
    print("  4 = 1024 (32x32)")
    print("  5 = 自定义")

    mode_choice = input("选择 (1-5): ").strip()

    if mode_choice == '1':
        num_modes = 16
    elif mode_choice == '2':
        num_modes = 64
    elif mode_choice == '3':
        num_modes = 256
    elif mode_choice == '4':
        num_modes = 1024
    elif mode_choice == '5':
        try:
            num_modes = int(input("输入模式数: ").strip())
        except:
            num_modes = 64
    else:
        num_modes = 64

    # 验证
    sqrt_n = int(np.sqrt(num_modes))
    if sqrt_n * sqrt_n != num_modes:
        num_modes = sqrt_n * sqrt_n
        print(f"  调整为平方数: {num_modes}")

    if basis_type != BasisGenerator.CANONICAL:
        # 检查是否是2的幂
        if not (num_modes & (num_modes - 1) == 0):
            # 找到最近的2的幂
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
    if pc == '1':
        pixel_size = 512
    else:
        pixel_size = 1024

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

    # 随机种子（用于Hadamard Random）
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

    print(f"\n配置确认:")
    print(f"  - 基函数: {basis_type}")
    print(f"  - 迭代策略: {iteration_strategy}")
    print(f"  - 模式数: {num_modes} ({int(np.sqrt(num_modes))}x{int(np.sqrt(num_modes))})")
    print(f"  - 迭代: {num_iter}")
    print(f"  - 像素: {pixel_size}x{pixel_size}")
    print(f"  - f值: {f_value}")
    print(f"  - 偏移: ({offset_x}, {offset_y})")

    # 估计时间
    est_time = num_modes * 5 * num_iter * (WAIT_TIME + integration_time_ms / 1000)
    print(f"  - 预计时间: {est_time:.0f}秒 ≈ {est_time / 60:.1f}分钟")

    if input("\n开始? (y/n): ").strip().lower() != 'y':
        print("已取消")
        exit()

    try:
        dash, logger = run_dash_v8(num_modes, num_iter, f_value, pixel_size, offset_x, offset_y,
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