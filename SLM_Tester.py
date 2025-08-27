"""
SLM_test.py - 测试Meadowlark SLM连接和显示
"""

import numpy as np
from ctypes import *
import time
import matplotlib.pyplot as plt


class SLM_Tester:
    def __init__(self, sdk_path="C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\SDK\\"):
        """初始化SLM测试器"""
        self.sdk_path = sdk_path
        self.slm_lib = None
        self.slm_initialized = False

    def initialize(self):
        """初始化SLM"""
        try:
            # 加载DLL
            cdll.LoadLibrary(self.sdk_path + "Blink_C_wrapper")
            self.slm_lib = CDLL("Blink_C_wrapper")

            # 初始化参数
            bit_depth = c_uint(12)
            num_boards_found = c_uint(0)
            constructed_okay = c_uint(-1)
            is_nematic_type = c_bool(1)
            RAM_write_enable = c_bool(1)
            use_GPU = c_bool(1)
            max_transients = c_uint(20)

            # 创建SDK
            self.slm_lib.Create_SDK(
                bit_depth, byref(num_boards_found), byref(constructed_okay),
                is_nematic_type, RAM_write_enable, use_GPU, max_transients, 0
            )

            if constructed_okay.value == 0:
                print("✅ SLM SDK initialized successfully")
                print(f"   Found {num_boards_found.value} SLM board(s)")
            else:
                print("❌ Failed to initialize SLM SDK")
                return False

            # 获取SLM参数
            self.board_number = 1
            self.height = self.slm_lib.Get_image_height(self.board_number)
            self.width = self.slm_lib.Get_image_width(self.board_number)
            self.depth = self.slm_lib.Get_image_depth(self.board_number)
            self.bytes = self.depth // 8

            print(f"   SLM Resolution: {self.width}x{self.height}")
            print(f"   Bit Depth: {self.depth} bits")

            # 加载LUT
            self._load_lut()

            self.slm_initialized = True
            return True

        except Exception as e:
            print(f"❌ Error initializing SLM: {e}")
            return False

    def _load_lut(self):
        """加载查找表"""
        lut_base = "C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\"

        if self.width == 512:
            lut_file = lut_base + "512x512_linearVoltage.LUT"
        elif self.width == 1920:
            lut_file = lut_base + "1920x1152_linearVoltage.LUT"
        elif self.width == 1024:
            lut_file = lut_base + "1024x1024_linearVoltage.LUT"
        else:
            lut_file = lut_base + "linearVoltage.LUT"

        self.slm_lib.Load_LUT_file(self.board_number, lut_file.encode())
        print(f"   Loaded LUT: {lut_file.split('')[-1]}")

    def display_pattern(self, pattern_type="checkerboard", param=20):
        """
        显示测试图案

        pattern_type: 'checkerboard', 'gradient', 'blazed_grating', 'vortex'
        param: 图案参数（如棋盘格大小、光栅周期等）
        """
        if not self.slm_initialized:
            print("❌ SLM not initialized")
            return

        print(f"\n📊 Displaying {pattern_type} pattern...")

        # 生成图案
        if pattern_type == "checkerboard":
            pattern = self._generate_checkerboard(param)
        elif pattern_type == "gradient":
            pattern = self._generate_gradient()
        elif pattern_type == "blazed_grating":
            pattern = self._generate_blazed_grating(param)
        elif pattern_type == "vortex":
            pattern = self._generate_vortex(param)
        else:
            print(f"Unknown pattern type: {pattern_type}")
            return

        # 转换为灰度值
        if self.depth == 8:
            gray = (pattern * 255).astype(np.uint8)
        else:  # 12或16位
            gray = (pattern * 4095).astype(np.uint16)

        # 准备数据缓冲区
        buffer_size = self.height * self.width * self.bytes
        buffer = np.zeros(buffer_size, dtype=np.uint8)

        if self.bytes == 1:
            buffer[:] = gray.flatten()
        else:
            gray_bytes = gray.tobytes()
            buffer[:] = np.frombuffer(gray_bytes, dtype=np.uint8)

        # 写入SLM
        ret = self.slm_lib.Write_image(
            self.board_number,
            buffer.ctypes.data_as(POINTER(c_ubyte)),
            buffer_size,
            c_uint(0),  # wait_for_trigger
            c_uint(0),  # flip_immediate
            c_uint(0),  # output_pulse_flip
            c_uint(0),  # output_pulse_refresh
            c_uint(5000)  # timeout_ms
        )

        if ret == -1:
            print("❌ Failed to write to SLM")
        else:
            print("✅ Pattern displayed successfully")

            # 显示预览
            plt.figure(figsize=(8, 8))
            plt.imshow(pattern, cmap='gray')
            plt.title(f'{pattern_type.capitalize()} Pattern')
            plt.colorbar()
            plt.show()

    def _generate_checkerboard(self, square_size):
        """生成棋盘格图案"""
        pattern = np.zeros((self.height, self.width))
        for i in range(0, self.height, square_size * 2):
            for j in range(0, self.width, square_size * 2):
                pattern[i:i + square_size, j:j + square_size] = 1
                pattern[i + square_size:i + 2 * square_size, j + square_size:j + 2 * square_size] = 1
        return pattern

    def _generate_gradient(self):
        """生成渐变图案"""
        x = np.linspace(0, 1, self.width)
        y = np.linspace(0, 1, self.height)
        X, Y = np.meshgrid(x, y)
        return (X + Y) / 2

    def _generate_blazed_grating(self, period):
        """生成闪耀光栅"""
        x = np.arange(self.width)
        pattern = (x % period) / period
        return np.tile(pattern, (self.height, 1))

    def _generate_vortex(self, charge):
        """生成涡旋相位图案"""
        cx, cy = self.width // 2, self.height // 2
        x = np.arange(self.width) - cx
        y = np.arange(self.height) - cy
        X, Y = np.meshgrid(x, y)
        theta = np.arctan2(Y, X)
        pattern = (theta * charge / (2 * np.pi)) % 1
        return pattern

    def cleanup(self):
        """清理资源"""
        if self.slm_lib and self.slm_initialized:
            self.slm_lib.Delete_SDK()
            print("🧹 SLM SDK cleaned up")


# 测试脚本
def test_slm():
    print("=" * 50)
    print("SLM CONNECTION TEST")
    print("=" * 50)

    # 初始化
    slm = SLM_Tester()

    if not slm.initialize():
        return

    # 测试不同图案
    while True:
        print("\n" + "=" * 50)
        print("Select test pattern:")
        print("1. Checkerboard")
        print("2. Gradient")
        print("3. Blazed Grating")
        print("4. Vortex")
        print("5. Exit")

        choice = input("Enter choice (1-5): ")

        if choice == '1':
            size = int(input("Square size (pixels, default=20): ") or "20")
            slm.display_pattern("checkerboard", size)
        elif choice == '2':
            slm.display_pattern("gradient")
        elif choice == '3':
            period = int(input("Grating period (pixels, default=10): ") or "10")
            slm.display_pattern("blazed_grating", period)
        elif choice == '4':
            charge = int(input("Vortex charge (default=1): ") or "1")
            slm.display_pattern("vortex", charge)
        elif choice == '5':
            break

    # 清理
    slm.cleanup()


if __name__ == "__main__":
    test_slm()