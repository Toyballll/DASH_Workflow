"""
SLM 共轭面定位 v5

纯相位调制方案，无需额外光学元件
"""

import numpy as np
from ctypes import *
import time

SLM_SIZE = 1024
GRATING_PERIOD = 10

# SLM 初始化
print("初始化 SLM...")
cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\SDK\\Blink_C_wrapper")
slm_lib = CDLL("Blink_C_wrapper")

num_boards_found = c_uint(0)
slm_lib.Create_SDK(c_uint(12), byref(num_boards_found), byref(c_uint(-1)),
                   c_bool(1), c_bool(1), c_bool(1), c_uint(20), 0)

board_number = c_uint(1)
slm_lib.Load_LUT_file(board_number,
                      b"C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\slm6227_at1064.lut")
print("SLM OK")


def display_pattern(pattern):
    if pattern.ndim == 2:
        pattern = pattern.flatten('C')
    slm_lib.Write_image(board_number, pattern.ctypes.data_as(POINTER(c_ubyte)),
                        c_uint(SLM_SIZE * SLM_SIZE), c_uint(0), c_uint(0),
                        c_uint(0), c_uint(0), c_uint(5000))
    slm_lib.ImageWriteComplete(board_number, c_uint(5000))


def create_flat():
    return np.full((SLM_SIZE, SLM_SIZE), 128, dtype=np.uint8)


# ============================================
# 方法1: 分区域相反偏转
# ============================================

def create_split_opposite_h(period=GRATING_PERIOD):
    """
    ★ 上下相反偏转

    上半：光栅向左偏转
    下半：光栅向右偏转（相位取反）

    共轭面：上下两半光斑水平分离
    其他位置：分离距离/模式不同
    """
    x = np.arange(SLM_SIZE)
    y = np.arange(SLM_SIZE)
    X, Y = np.meshgrid(x, y)

    phase = np.zeros((SLM_SIZE, SLM_SIZE))

    # 上半：正向光栅
    top = Y < SLM_SIZE // 2
    phase[top] = (X[top] % period) / period * 2 * np.pi

    # 下半：反向光栅（相位反转）
    bottom = Y >= SLM_SIZE // 2
    phase[bottom] = ((period - X[bottom] % period) % period) / period * 2 * np.pi

    return ((phase / (2 * np.pi)) * 255).astype(np.uint8)


def create_split_opposite_v(period=GRATING_PERIOD):
    """
    ★ 左右相反偏转

    左半：光栅向上偏转
    右半：光栅向下偏转

    共轭面：左右两半光斑垂直分离
    """
    x = np.arange(SLM_SIZE)
    y = np.arange(SLM_SIZE)
    X, Y = np.meshgrid(x, y)

    phase = np.zeros((SLM_SIZE, SLM_SIZE))

    left = X < SLM_SIZE // 2
    phase[left] = (Y[left] % period) / period * 2 * np.pi

    right = X >= SLM_SIZE // 2
    phase[right] = ((period - Y[right] % period) % period) / period * 2 * np.pi

    return ((phase / (2 * np.pi)) * 255).astype(np.uint8)


def create_four_quadrant_split(period=GRATING_PERIOD):
    """
    四象限对角偏转（旧版，相位耦合）
    """
    x = np.arange(SLM_SIZE)
    y = np.arange(SLM_SIZE)
    X, Y = np.meshgrid(x, y)

    phase = np.zeros((SLM_SIZE, SLM_SIZE))
    cx, cy = SLM_SIZE // 2, SLM_SIZE // 2

    q1 = (X >= cx) & (Y < cy)
    phase[q1] = ((X[q1] + Y[q1]) % period) / period * 2 * np.pi

    q2 = (X < cx) & (Y < cy)
    phase[q2] = (((period - X[q2] % period) % period + Y[q2]) % period) / period * 2 * np.pi

    q3 = (X < cx) & (Y >= cy)
    phase[q3] = ((period - (X[q3] + Y[q3]) % period) % period) / period * 2 * np.pi

    q4 = (X >= cx) & (Y >= cy)
    phase[q4] = ((X[q4] + (period - Y[q4] % period) % period) % period) / period * 2 * np.pi

    return ((phase / (2 * np.pi)) * 255).astype(np.uint8)


def create_four_quadrant_diagonal(period=GRATING_PERIOD):
    """
    ★★ 四象限对角偏转 - 同时检测X和Y共轭

    右上：向右上偏转 (+X, -Y)
    左上：向左上偏转 (-X, -Y)
    左下：向左下偏转 (-X, +Y)
    右下：向右下偏转 (+X, +Y)

    共轭面特征：四个象限的角在中心精确对齐

    像散诊断：
    - 无像散：四象限同步向四角分离
    - 有像散：呈"十"字形错开（X/Y分离不同步）
    """
    x = np.arange(SLM_SIZE)
    y = np.arange(SLM_SIZE)
    X, Y = np.meshgrid(x, y)

    phase = np.zeros((SLM_SIZE, SLM_SIZE))
    cx, cy = SLM_SIZE // 2, SLM_SIZE // 2

    # 水平光栅相位 (控制X方向偏转)
    phase_x_pos = (X % period) / period * 2 * np.pi  # 向右偏转
    phase_x_neg = ((period - X % period) % period) / period * 2 * np.pi  # 向左偏转

    # 垂直光栅相位 (控制Y方向偏转)
    phase_y_pos = (Y % period) / period * 2 * np.pi  # 向下偏转
    phase_y_neg = ((period - Y % period) % period) / period * 2 * np.pi  # 向上偏转

    # 右上象限：向右(+X) + 向上(-Y)
    q1 = (X >= cx) & (Y < cy)
    phase[q1] = (phase_x_pos[q1] + phase_y_neg[q1]) % (2 * np.pi)

    # 左上象限：向左(-X) + 向上(-Y)
    q2 = (X < cx) & (Y < cy)
    phase[q2] = (phase_x_neg[q2] + phase_y_neg[q2]) % (2 * np.pi)

    # 左下象限：向左(-X) + 向下(+Y)
    q3 = (X < cx) & (Y >= cy)
    phase[q3] = (phase_x_neg[q3] + phase_y_pos[q3]) % (2 * np.pi)

    # 右下象限：向右(+X) + 向下(+Y)
    q4 = (X >= cx) & (Y >= cy)
    phase[q4] = (phase_x_pos[q4] + phase_y_pos[q4]) % (2 * np.pi)

    return ((phase / (2 * np.pi)) * 255).astype(np.uint8)


# ============================================
# 方法2: 闪烁对比
# ============================================

def blink_opposite(period=GRATING_PERIOD, delay=0.5):
    """
    ★ 交替闪烁：左偏 vs 右偏

    共轭面：光斑左右跳动最明显
    非共轭面：跳动幅度小或方向不同
    """
    x = np.arange(SLM_SIZE)
    y = np.arange(SLM_SIZE)
    X, Y = np.meshgrid(x, y)

    # 向左偏转
    phase_left = (X % period) / period * 2 * np.pi
    pattern_left = ((phase_left / (2 * np.pi)) * 255).astype(np.uint8)

    # 向右偏转
    phase_right = ((period - X % period) % period) / period * 2 * np.pi
    pattern_right = ((phase_right / (2 * np.pi)) * 255).astype(np.uint8)

    print("光斑左右跳动... (Ctrl+C停止)")
    print("找跳动幅度最大的位置 = 共轭面")

    try:
        while True:
            display_pattern(pattern_left)
            time.sleep(delay)
            display_pattern(pattern_right)
            time.sleep(delay)
    except KeyboardInterrupt:
        print("停止")


def blink_quadrant(period=GRATING_PERIOD, delay=0.4):
    """
    ★ 四象限轮流亮

    共轭面：明确看到四个位置轮流亮
    非共轭面：位置关系不对应
    """
    x = np.arange(SLM_SIZE)
    y = np.arange(SLM_SIZE)
    X, Y = np.meshgrid(x, y)
    cx, cy = SLM_SIZE // 2, SLM_SIZE // 2

    grating = (X % period) / period * 2 * np.pi

    quadrants = [
        (X >= cx) & (Y < cy),  # 右上
        (X < cx) & (Y < cy),  # 左上
        (X < cx) & (Y >= cy),  # 左下
        (X >= cx) & (Y >= cy),  # 右下
    ]

    print("四象限轮流... (Ctrl+C停止)")

    try:
        while True:
            for i, q in enumerate(quadrants):
                # 当前象限平场（亮），其他光栅偏转（暗）
                phase = grating.copy()
                phase[q] = np.pi
                pattern = ((phase / (2 * np.pi)) * 255).astype(np.uint8)
                display_pattern(pattern)
                print(f"  象限 {i + 1}", end='\r')
                time.sleep(delay)
    except KeyboardInterrupt:
        print("\n停止")


# ============================================
# 方法3: 渐变偏转角度
# ============================================

def sweep_grating_period(min_period=5, max_period=30, steps=20, delay=0.3):
    """
    ★ 扫描光栅周期

    不同周期 = 不同偏转角度
    观察光斑位置随周期变化的轨迹

    共轭面：轨迹是直线
    非共轭面：轨迹可能弯曲或不规则
    """
    x = np.arange(SLM_SIZE)
    y = np.arange(SLM_SIZE)
    X, Y = np.meshgrid(x, y)

    periods = np.linspace(min_period, max_period, steps)

    print(f"扫描周期 {min_period} -> {max_period}... (Ctrl+C停止)")

    try:
        while True:
            for p in periods:
                phase = (X % p) / p * 2 * np.pi
                pattern = ((phase / (2 * np.pi)) * 255).astype(np.uint8)
                display_pattern(pattern)
                time.sleep(delay)
            for p in reversed(periods):
                phase = (X % p) / p * 2 * np.pi
                pattern = ((phase / (2 * np.pi)) * 255).astype(np.uint8)
                display_pattern(pattern)
                time.sleep(delay)
    except KeyboardInterrupt:
        print("停止")


# ============================================
# 方法4: 中心vs边缘不同偏转
# ============================================

def create_radial_split(inner_r=200, period=GRATING_PERIOD):
    """
    ★ 中心和边缘相反偏转

    中心圆：向左偏转
    外圈：向右偏转

    共轭面：中心光斑和环形光斑分离
    """
    x = np.arange(SLM_SIZE) - SLM_SIZE // 2
    y = np.arange(SLM_SIZE) - SLM_SIZE // 2
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X ** 2 + Y ** 2)

    Xp = X + SLM_SIZE // 2

    phase = ((period - Xp % period) % period) / period * 2 * np.pi  # 外圈右偏

    inner = R < inner_r
    phase[inner] = (Xp[inner] % period) / period * 2 * np.pi  # 中心左偏

    return ((phase / (2 * np.pi)) * 255).astype(np.uint8)


# ============================================
# 方法5: 条纹旋转
# ============================================

def rotate_grating(period=GRATING_PERIOD, steps=36, delay=0.2):
    """
    ★ 光栅旋转

    光栅方向连续旋转，光斑绕圈移动
    共轭面：光斑轨迹是圆形
    非共轭面：轨迹可能变形
    """
    x = np.arange(SLM_SIZE) - SLM_SIZE // 2
    y = np.arange(SLM_SIZE) - SLM_SIZE // 2
    X, Y = np.meshgrid(x, y)

    print("光栅旋转，光斑绕圈... (Ctrl+C停止)")

    try:
        while True:
            for i in range(steps):
                angle = 2 * np.pi * i / steps
                # 旋转后的坐标
                Xr = X * np.cos(angle) + Y * np.sin(angle)

                phase = (Xr % period) / period * 2 * np.pi
                phase = phase % (2 * np.pi)
                pattern = ((phase / (2 * np.pi)) * 255).astype(np.uint8)
                display_pattern(pattern)
                time.sleep(delay)
    except KeyboardInterrupt:
        print("停止")


# ============================================
patterns = {
    '0': ('flat', create_flat, "平场"),
    '1': ('split_h', create_split_opposite_h, "★ 上下相反偏转 (检测X共轭)"),
    '2': ('split_v', create_split_opposite_v, "★ 左右相反偏转 (检测Y共轭)"),
    '3': ('four_q', create_four_quadrant_split, "四象限四方向(旧)"),
    '4': ('radial', create_radial_split, "中心/边缘相反偏转"),
    '5': ('diag', create_four_quadrant_diagonal, "★★ 四象限对角 (同时检测X+Y)"),
}

dynamic = {
    'a': ('blink', blink_opposite, "★ 闪烁：光斑左右跳"),
    'b': ('quad_blink', blink_quadrant, "四象限轮流亮"),
    'c': ('sweep', sweep_grating_period, "扫描光栅周期"),
    'd': ('rotate', rotate_grating, "光栅旋转"),
}

print("\n" + "=" * 50)
print("纯SLM相位调制 - 共轭面定位")
print("=" * 50)

print("\n【核心原理】")
print("  共轭面：SLM空间结构精确映射")
print("  非共轭面：映射关系有偏差")

print("\n【静态图案】")
for key, (_, _, desc) in patterns.items():
    print(f"  {key} - {desc}")

print("\n【动态图案】")
for key, (_, _, desc) in dynamic.items():
    print(f"  {key} - {desc}")

print("\n  q - 退出")

print("\n" + "-" * 50)
print("★ 推荐测试流程：")
print("  1. 按'5'显示四象限对角图案")
print("  2. 沿光路移动IR scope")
print("  3. 四个象限角在中心对齐的位置 = 共轭面")
print("")
print("★ 像散诊断：")
print("  - 无像散：四象限同步向四角分离/收拢")
print("  - 有像散：呈十字形错开(X/Y不同步)")
print("  - 用'1'和'2'分别找X和Y的共轭面，比较位置差")
print("-" * 50)

display_pattern(create_flat())

try:
    while True:
        key = input("\n选择: ").strip().lower()

        if key == 'q':
            break
        elif key in patterns:
            _, func, desc = patterns[key]
            display_pattern(func())
            print(f"显示: {desc}")
        elif key in dynamic:
            _, func, desc = dynamic[key]
            print(f"运行: {desc}")
            func()
            display_pattern(create_flat())
        else:
            print("无效")

except KeyboardInterrupt:
    pass

finally:
    display_pattern(create_flat())
    slm_lib.Delete_SDK()
    print("完成")