"""
Zernike模式扫描像差校正q


经典方法：对每个Zernike模式扫描系数，选择信号最大值对应的系数

功能：
1. 逐个Zernike模式扫描
2. 记录每个模式的扫描信号曲线
3. 累积最佳系数形成校正相位
4. 交互式切换default/correction
5. 可选择校正的模式数量和扫描范围
"""

import numpy as np
from ctypes import *
import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
import time
from PIL import Image
from datetime import datetime
import math

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

# DAQ配置
device_name = "Dev2"
pmt_channel = "ai0"
sample_rate = 10000
integration_time_ms = 30
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
            min_val=-10.0, max_val=10.0)
        task.timing.cfg_samp_clk_timing(
            rate=sample_rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=samples)
        data = np.array(task.read(number_of_samples_per_channel=samples))
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
# Zernike多项式
# ============================================

# Zernike模式定义 (n, m, name)
# 按照Noll索引顺序
ZERNIKE_MODES = [
    (0, 0, "Piston"),  # Z1 - 通常跳过
    (1, -1, "Tilt Y"),  # Z2
    (1, 1, "Tilt X"),  # Z3
    (2, -2, "Astigmatism 45°"),  # Z4
    (2, 0, "Defocus"),  # Z5
    (2, 2, "Astigmatism 0°"),  # Z6
    (3, -3, "Trefoil Y"),  # Z7
    (3, -1, "Coma Y"),  # Z8
    (3, 1, "Coma X"),  # Z9
    (3, 3, "Trefoil X"),  # Z10
    (4, -4, "Tetrafoil Y"),  # Z11
    (4, -2, "2nd Astig Y"),  # Z12
    (4, 0, "Spherical"),  # Z13
    (4, 2, "2nd Astig X"),  # Z14
    (4, 4, "Tetrafoil X"),  # Z15
    (5, -5, "Pentafoil Y"),  # Z16
    (5, -3, "2nd Trefoil Y"),  # Z17
    (5, -1, "2nd Coma Y"),  # Z18
    (5, 1, "2nd Coma X"),  # Z19
    (5, 3, "2nd Trefoil X"),  # Z20
    (5, 5, "Pentafoil X"),  # Z21
]


def zernike_radial(n, m, rho):
    """计算Zernike径向多项式 R_n^m(rho)"""
    m_abs = abs(m)
    result = np.zeros_like(rho)

    for k in range((n - m_abs) // 2 + 1):
        coef = ((-1) ** k * math.factorial(n - k) /
                (math.factorial(k) *
                 math.factorial((n + m_abs) // 2 - k) *
                 math.factorial((n - m_abs) // 2 - k)))
        result += coef * rho ** (n - 2 * k)

    return result


def zernike_polynomial(n, m, rho, theta):
    """计算Zernike多项式 Z_n^m(rho, theta)"""
    R = zernike_radial(n, m, rho)

    if m >= 0:
        return R * np.cos(m * theta)
    else:
        return R * np.sin(-m * theta)


def generate_zernike_basis(size, num_modes):
    """生成Zernike基函数"""
    # 创建归一化坐标
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)

    rho = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan2(Y, X)

    # 圆形孔径掩膜
    mask = rho <= 1.0

    # 生成Zernike基
    basis = []
    for i in range(min(num_modes, len(ZERNIKE_MODES))):
        n, m, name = ZERNIKE_MODES[i]
        Z = zernike_polynomial(n, m, rho, theta)
        Z[~mask] = 0
        basis.append(Z)

    return basis, mask


class ZernikeCorrector:
    """Zernike模式扫描像差校正器"""

    def __init__(self, num_modes=15, pixel_size=1024, offset_x=0, offset_y=0,
                 scan_range=2.0, scan_steps=21, start_mode=1):
        """
        参数:
            num_modes: 校正的Zernike模式数量（从start_mode开始）
            pixel_size: 活动区域像素数
            scan_range: 扫描范围 [-scan_range*pi, +scan_range*pi]
            scan_steps: 每个模式扫描的步数
            start_mode: 起始模式索引（0=piston, 1=tilt Y, 通常从1或4开始）
        """
        self.num_modes = num_modes
        self.pixel_size = pixel_size
        self.slm_size = 1024
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.scan_range = scan_range
        self.scan_steps = scan_steps
        self.start_mode = start_mode

        # 活动区域
        base_start = (self.slm_size - self.pixel_size) // 2
        self.start_x = max(0, min(base_start + offset_x, self.slm_size - self.pixel_size))
        self.start_y = max(0, min(base_start + offset_y, self.slm_size - self.pixel_size))
        self.end_x = self.start_x + self.pixel_size
        self.end_y = self.start_y + self.pixel_size

        # 生成Zernike基
        total_modes = start_mode + num_modes
        self.zernike_basis, self.mask = generate_zernike_basis(pixel_size, total_modes)

        # 扫描系数范围
        self.scan_coeffs = np.linspace(-scan_range * np.pi, scan_range * np.pi, scan_steps)

        # 最佳系数（初始为0）
        self.best_coeffs = np.zeros(total_modes)

        # 校正相位
        self.correction_phase = np.zeros((pixel_size, pixel_size))
        self.final_correction_pattern = None
        self.default_pattern = load_default_pattern()

        # 扫描历史记录
        self.scan_history = {}  # mode_idx -> (coeffs, signals)

        print(f"\nZernike校正器配置:")
        print(f"  - 模式数: {num_modes} (从Z{start_mode + 1}到Z{start_mode + num_modes})")
        print(f"  - 像素: {pixel_size}x{pixel_size}")
        print(f"  - 扫描范围: [{-scan_range:.1f}π, +{scan_range:.1f}π]")
        print(f"  - 扫描步数: {scan_steps}")

    def compute_phase_pattern(self, coeffs=None):
        """根据系数计算相位图案"""
        if coeffs is None:
            coeffs = self.best_coeffs

        phase = np.zeros((self.pixel_size, self.pixel_size))
        for i, coeff in enumerate(coeffs):
            if i < len(self.zernike_basis):
                phase += coeff * self.zernike_basis[i]

        return phase

    def generate_pattern(self, phase):
        """将相位转换为SLM灰度图案"""
        # 转换为灰度 [0, 255]
        phase_wrapped = np.mod(phase + np.pi, 2 * np.pi)
        gray_active = (phase_wrapped * 255 / (2 * np.pi)).astype(np.uint8)

        # 嵌入完整图案
        full_pattern = np.full((self.slm_size, self.slm_size), 128, dtype=np.uint8)
        full_pattern[self.start_y:self.end_y, self.start_x:self.end_x] = gray_active

        return full_pattern.flatten('C')

    def display_pattern(self, pattern):
        """显示图案并测量信号"""
        slm_lib.Write_image(board_number, pattern.ctypes.data_as(POINTER(c_ubyte)),
                            c_uint(1024 * 1024), c_uint(0), c_uint(0),
                            c_uint(0), c_uint(0), c_uint(5000))
        slm_lib.ImageWriteComplete(board_number, c_uint(5000))
        time.sleep(WAIT_TIME)
        return measure_intensity()

    def measure_baseline(self):
        """测量基线（默认图案）"""
        slm_lib.Write_image(board_number, self.default_pattern.ctypes.data_as(POINTER(c_ubyte)),
                            c_uint(1024 * 1024), c_uint(0), c_uint(0),
                            c_uint(0), c_uint(0), c_uint(5000))
        slm_lib.ImageWriteComplete(board_number, c_uint(5000))
        time.sleep(WAIT_TIME * 2)
        return measure_intensity()

    def scan_single_mode(self, mode_idx):
        """扫描单个Zernike模式"""
        n, m, name = ZERNIKE_MODES[mode_idx]
        print(f"  扫描 Z{mode_idx + 1} ({name})...")

        signals = []

        # 保存当前系数
        temp_coeffs = self.best_coeffs.copy()

        for coeff in self.scan_coeffs:
            # 设置当前模式的系数
            temp_coeffs[mode_idx] = coeff

            # 计算相位并显示
            phase = self.compute_phase_pattern(temp_coeffs)
            pattern = self.generate_pattern(phase)
            signal = self.display_pattern(pattern)
            signals.append(signal)

        signals = np.array(signals)

        # 找最佳系数
        best_idx = np.argmax(signals)
        best_coeff = self.scan_coeffs[best_idx]
        best_signal = signals[best_idx]

        # 更新最佳系数
        self.best_coeffs[mode_idx] = best_coeff

        # 保存扫描历史
        self.scan_history[mode_idx] = (self.scan_coeffs.copy(), signals.copy())

        print(f"    最佳系数: {best_coeff / np.pi:.3f}π, 信号: {best_signal:.6f}V")

        return best_coeff, best_signal, signals

    def run_correction(self, logger):
        """运行完整的像差校正"""
        print("\n" + "=" * 60)
        print("【Zernike模式扫描校正】")
        print("=" * 60)

        # 测量初始基线
        baseline = self.measure_baseline()
        print(f"初始基线: {baseline:.6f}V")
        logger.log(f"Initial baseline: {baseline:.6f}V")

        # 逐个模式扫描
        for i in range(self.num_modes):
            mode_idx = self.start_mode + i
            if mode_idx >= len(ZERNIKE_MODES):
                break

            best_coeff, best_signal, signals = self.scan_single_mode(mode_idx)

            n, m, name = ZERNIKE_MODES[mode_idx]
            logger.log_mode_scan(mode_idx, n, m, name, self.scan_coeffs, signals, best_coeff)

            # 显示当前累积校正效果
            self.correction_phase = self.compute_phase_pattern()
            pattern = self.generate_pattern(self.correction_phase)
            current_signal = self.display_pattern(pattern)
            enhancement = current_signal / baseline
            print(f"    累积Enhancement: {enhancement:.3f}x")

        # 生成最终校正图案
        self.correction_phase = self.compute_phase_pattern()
        self.final_correction_pattern = self.generate_pattern(self.correction_phase)

        return baseline

    def display_default(self):
        """显示默认图案"""
        return self.display_pattern(self.default_pattern)

    def display_correction(self):
        """显示校正图案"""
        if self.final_correction_pattern is not None:
            return self.display_pattern(self.final_correction_pattern)
        else:
            return self.display_default()

    def save_data(self, timestamp):
        """保存数据"""
        # 保存系数
        np.save(f"zernike_coeffs_{timestamp}.npy", self.best_coeffs)

        # 保存校正相位
        np.save(f"zernike_phase_{timestamp}.npy", self.correction_phase)

        # 保存校正图案
        if self.final_correction_pattern is not None:
            pattern_2d = self.final_correction_pattern.reshape(1024, 1024)
            Image.fromarray(pattern_2d).save(f"zernike_pattern_{timestamp}.png")

        print(f"✓ 数据已保存")


class ZernikeLogger:
    """Zernike校正数据记录器"""

    def __init__(self, num_modes, scan_range, scan_steps, start_mode, pixel_size):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"Zernike_{timestamp}.txt"
        self.timestamp = timestamp

        with open(self.filename, 'w') as f:
            f.write(f"# Zernike Mode Scanning - {datetime.now()}\n")
            f.write(f"# Modes: Z{start_mode + 1} to Z{start_mode + num_modes}\n")
            f.write(f"# Scan range: [-{scan_range}π, +{scan_range}π], {scan_steps} steps\n")
            f.write(f"# Pixel size: {pixel_size}x{pixel_size}\n")
            f.write("#" + "-" * 70 + "\n")

        print(f"数据文件: {self.filename}")

    def log(self, msg):
        with open(self.filename, 'a') as f:
            f.write(f"# [{datetime.now().strftime('%H:%M:%S')}] {msg}\n")

    def log_mode_scan(self, mode_idx, n, m, name, coeffs, signals, best_coeff):
        """记录单个模式的扫描结果"""
        with open(self.filename, 'a') as f:
            f.write(f"# Mode Z{mode_idx + 1} (n={n}, m={m}, {name})\n")
            f.write(f"# Best coefficient: {best_coeff / np.pi:.6f}π\n")
            f.write(f"# Coeff(π), Signal(V)\n")
            for coeff, signal in zip(coeffs, signals):
                f.write(f"{coeff / np.pi:.6f}, {signal:.9f}\n")
            f.write("#" + "-" * 40 + "\n")

    def log_final_comparison(self, baseline, default_signals, corrected_signals):
        """保存最终对比结果"""
        with open(self.filename, 'a') as f:
            f.write("#" + "=" * 70 + "\n")
            f.write("# FINAL COMPARISON RESULTS\n")
            f.write("#" + "=" * 70 + "\n")
            f.write(f"# Baseline: {baseline:.9f} V\n")
            f.write("#\n")
            f.write("# Default pattern measurements:\n")
            for i, sig in enumerate(default_signals):
                f.write(f"#   Default[{i}]: {sig:.9f} V\n")
            f.write(f"#   Default avg: {np.mean(default_signals):.9f} V\n")
            f.write("#\n")
            f.write("# Correction pattern measurements:\n")
            for i, sig in enumerate(corrected_signals):
                f.write(f"#   Correction[{i}]: {sig:.9f} V\n")
            f.write(f"#   Correction avg: {np.mean(corrected_signals):.9f} V\n")
            f.write("#\n")
            f.write(f"# Enhancement: {np.mean(corrected_signals) / np.mean(default_signals):.6f}x\n")
            f.write("#" + "=" * 70 + "\n")


def run_zernike_correction(num_modes, pixel_size, offset_x, offset_y,
                           scan_range, scan_steps, start_mode):
    """运行Zernike校正"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = ZernikeLogger(num_modes, scan_range, scan_steps, start_mode, pixel_size)

    # 初始化校正器
    corrector = ZernikeCorrector(num_modes, pixel_size, offset_x, offset_y,
                                 scan_range, scan_steps, start_mode)

    # 运行校正
    baseline = corrector.run_correction(logger)

    # 保存数据
    corrector.save_data(timestamp)

    # 最终对比测试
    print("\n" + "=" * 60)
    print("【最终结果对比】")
    print("=" * 60)

    default_signals = []
    corrected_signals = []

    for i in range(5):
        default_signals.append(corrector.display_default())
        time.sleep(0.1)
        corrected_signals.append(corrector.display_correction())
        time.sleep(0.1)

    default_avg = np.mean(default_signals)
    corrected_avg = np.mean(corrected_signals)

    print(f"默认图案: {default_avg:.6f} V")
    for i, sig in enumerate(default_signals):
        print(f"  [{i}] {sig:.6f} V")

    print(f"校正图案: {corrected_avg:.6f} V")
    for i, sig in enumerate(corrected_signals):
        print(f"  [{i}] {sig:.6f} V")

    print(f"Enhancement: {corrected_avg / default_avg:.3f}x")

    logger.log_final_comparison(baseline, default_signals, corrected_signals)

    # 打印最佳系数
    print("\n最佳Zernike系数:")
    for i in range(corrector.start_mode, corrector.start_mode + corrector.num_modes):
        if i < len(ZERNIKE_MODES):
            n, m, name = ZERNIKE_MODES[i]
            print(f"  Z{i + 1} ({name}): {corrector.best_coeffs[i] / np.pi:.3f}π")

    # 交互测试
    print("\n" + "=" * 60)
    print("【交互测试】1=默认, 2=校正, 3=检查漂移, q=退出")
    print("=" * 60)

    while True:
        try:
            choice = input("\n选择: ").strip().lower()
            if choice == 'q':
                break
            elif choice == '1':
                sig = corrector.display_default()
                print(f"  默认: {sig:.6f}V ({sig / baseline:.3f}x)")
            elif choice == '2':
                sig = corrector.display_correction()
                print(f"  校正: {sig:.6f}V ({sig / baseline:.3f}x)")
            elif choice == '3':
                sig = corrector.measure_baseline()
                print(f"  当前基线: {sig:.6f}V (漂移: {sig / baseline:.3f}x)")
        except KeyboardInterrupt:
            break

    return corrector, logger


# ============================================
# 主程序
# ============================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Zernike模式扫描像差校正")
    print("=" * 60)

    # 显示可用模式
    print("\n可用Zernike模式:")
    for i, (n, m, name) in enumerate(ZERNIKE_MODES):
        print(f"  Z{i + 1}: {name} (n={n}, m={m})")

    # 起始模式
    print("\n起始模式: 1=Z2(跳过piston), 2=Z4(跳过tilt), 3=Z5(从defocus开始), 4=自定义")
    sm = input("选择 (1-4): ").strip()
    if sm == '1':
        start_mode = 1  # 从Z2开始
    elif sm == '2':
        start_mode = 3  # 从Z4开始，跳过tilt
    elif sm == '3':
        start_mode = 4  # 从Z5(defocus)开始
    elif sm == '4':
        try:
            start_mode = int(input("输入起始模式索引 (0=Z1, 1=Z2, ...): ").strip())
            start_mode = max(0, min(start_mode, len(ZERNIKE_MODES) - 1))
        except:
            start_mode = 1
    else:
        start_mode = 1

    # 模式数量
    print("\n校正模式数: 1=5个, 2=10个, 3=15个, 4=自定义")
    nm = input("选择 (1-4): ").strip()
    if nm == '1':
        num_modes = 5
    elif nm == '2':
        num_modes = 10
    elif nm == '3':
        num_modes = 15
    elif nm == '4':
        try:
            num_modes = int(input("输入模式数量: ").strip())
            num_modes = max(1, min(num_modes, len(ZERNIKE_MODES) - start_mode))
        except:
            num_modes = 10
    else:
        num_modes = 10

    # 扫描范围
    print("\n扫描范围: 1=[-π,+π], 2=[-2π,+2π], 3=[-3π,+3π], 4=自定义")
    sr = input("选择 (1-4): ").strip()
    if sr == '1':
        scan_range = 1.0
    elif sr == '2':
        scan_range = 2.0
    elif sr == '3':
        scan_range = 3.0
    elif sr == '4':
        try:
            scan_range = float(input("输入扫描范围 (单位π): ").strip())
            scan_range = max(0.5, min(scan_range, 5.0))
        except:
            scan_range = 2.0
    else:
        scan_range = 2.0

    # 扫描步数
    print("\n扫描步数: 1=11步, 2=21步, 3=41步, 4=自定义")
    ss = input("选择 (1-4): ").strip()
    if ss == '1':
        scan_steps = 11
    elif ss == '2':
        scan_steps = 21
    elif ss == '3':
        scan_steps = 41
    elif ss == '4':
        try:
            scan_steps = int(input("输入扫描步数: ").strip())
            scan_steps = max(5, min(scan_steps, 101))
        except:
            scan_steps = 21
    else:
        scan_steps = 21

    # 像素数
    print("\n像素数: 1=512, 2=1024, 3=自定义")
    pc = input("选择 (1-3): ").strip()
    if pc == '1':
        pixel_size = 512
    elif pc == '2':
        pixel_size = 1024
    elif pc == '3':
        try:
            pixel_size = int(input("输入像素数: ").strip())
            pixel_size = max(64, min(pixel_size, 1024))
        except:
            pixel_size = 1024
    else:
        pixel_size = 1024

    # 偏移
    print("\n偏移 (格式: x y，如 0 0): ")
    try:
        offset_input = input().strip().split()
        offset_x, offset_y = int(offset_input[0]), int(offset_input[1])
    except:
        offset_x, offset_y = 0, 0

    # 确认
    print(f"\n配置确认:")
    print(f"  - 模式范围: Z{start_mode + 1} 到 Z{start_mode + num_modes}")
    print(f"  - 扫描范围: [-{scan_range:.1f}π, +{scan_range:.1f}π]")
    print(f"  - 扫描步数: {scan_steps}")
    print(f"  - 像素: {pixel_size}x{pixel_size}")
    print(f"  - 偏移: ({offset_x}, {offset_y})")

    est_time = num_modes * scan_steps * 0.06  # 约60ms每步
    print(f"  - 预计时间: {est_time:.0f}秒 ≈ {est_time / 60:.1f}分钟")

    if input("\n开始? (y/n): ").strip().lower() != 'y':
        print("已取消")
        exit()

    try:
        corrector, logger = run_zernike_correction(
            num_modes, pixel_size, offset_x, offset_y,
            scan_range, scan_steps, start_mode)
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