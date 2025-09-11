"""
SLM-DAQ 硬件预加载触发测试
使用Load_sequence和Select_image函数
"""

import numpy as np
from ctypes import *
from PIL import Image as PILImage
import nidaqmx
from nidaqmx.constants import LineGrouping
import time


class SLM_Hardware_Preload:
    def __init__(self):
        # DAQ配置
        self.device_name = "Dev2"
        self.trigger_out_line = "port0/line0"  # P0.0 -> SLM Trigger In
        self.trigger_in_line = "port0/line1"  # P0.1 <- SLM Trigger Out

        # SLM变量
        self.slm_lib = None
        self.board_number = c_uint(1)
        self.width = 1024
        self.height = 1024
        self.ImgSize = 1024 * 1024

    def init_slm(self):
        """初始化SLM"""
        print("初始化SLM...")

        # 加载SDK
        cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\SDK\\Blink_C_wrapper")
        self.slm_lib = CDLL("Blink_C_wrapper")

        # 创建SDK
        num_boards_found = c_uint(0)
        self.slm_lib.Create_SDK(c_uint(12), byref(num_boards_found), byref(c_uint(-1)),
                                c_bool(1), c_bool(1), c_bool(1), c_uint(20), 0)

        if num_boards_found.value < 1:
            print("未找到SLM")
            return False

        print(f"找到 {num_boards_found.value} 个SLM")

        # 加载LUT
        self.slm_lib.Load_LUT_file(self.board_number,
                                   b"C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\1024x1024_linearVoltage.LUT")

        return True

    def load_images_to_hardware(self):
        """使用Load_sequence预加载两个图像到硬件"""
        print("\n使用Load_sequence预加载图像到硬件...")

        # 图像路径
        image1_path = "C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\Image Files\\1024\\Central.bmp"
        image2_path = "C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\Image Files\\1024\\mlo.bmp"

        # 加载第一个图像
        img1 = PILImage.open(image1_path)
        img1_array = np.array(img1.convert('L'), dtype=np.uint8)

        # 加载第二个图像
        img2 = PILImage.open(image2_path)
        img2_array = np.array(img2.convert('L'), dtype=np.uint8)

        # 创建图像数组（2个图像）
        list_length = 2
        total_size = self.ImgSize * list_length
        image_array = np.zeros(total_size, dtype=np.uint8)

        # 将两个图像放入数组
        image_array[0:self.ImgSize] = img1_array.flatten()
        image_array[self.ImgSize:2 * self.ImgSize] = img2_array.flatten()

        # 参数设置
        wait_for_trigger = c_uint(1)  # 等待外部触发
        flip_immediate = c_uint(0)
        output_pulse_image_flip = c_uint(1)
        output_pulse_image_refresh = c_uint(0)
        trigger_timeout_ms = c_uint(10000)

        # 加载序列到硬件
        print(f"  加载 {list_length} 个图像到硬件内存...")
        ret = self.slm_lib.Load_sequence(
            self.board_number,
            image_array.ctypes.data_as(POINTER(c_ubyte)),
            c_uint(self.ImgSize),
            c_int(list_length),
            wait_for_trigger,
            flip_immediate,
            output_pulse_image_flip,
            output_pulse_image_refresh,
            trigger_timeout_ms
        )

        if ret == -1:
            print("  图像序列加载失败")
            return False

        print(f"  ✓ 成功加载 {list_length} 个图像到硬件")
        print("    - Frame 0: Central.bmp")
        print("    - Frame 1: mlo.bmp")
        return True

    def select_and_trigger(self, frame_index):
        """选择预加载的图像并触发"""
        print(f"\n选择Frame {frame_index}并触发...")

        with nidaqmx.Task() as output_task, nidaqmx.Task() as input_task:
            # 配置DAQ
            output_task.do_channels.add_do_chan(
                f"{self.device_name}/{self.trigger_out_line}",
                line_grouping=LineGrouping.CHAN_PER_LINE
            )

            input_task.di_channels.add_di_chan(
                f"{self.device_name}/{self.trigger_in_line}",
                line_grouping=LineGrouping.CHAN_PER_LINE
            )

            # 确保触发线初始为高
            output_task.write(True)
            time.sleep(0.1)

            # 启动Select_image（在另一个线程）
            import threading
            select_done = threading.Event()

            def select_thread():
                # Select_image参数
                wait_for_trigger = c_uint(1)
                flip_immediate = c_uint(0)
                output_pulse_image_flip = c_uint(1)
                output_pulse_image_refresh = c_uint(0)
                flip_timeout_ms = c_uint(5000)

                # 选择图像（等待触发）
                self.slm_lib.Select_image(
                    self.board_number,
                    c_int(frame_index),
                    wait_for_trigger,
                    flip_immediate,
                    output_pulse_image_flip,
                    output_pulse_image_refresh,
                    flip_timeout_ms
                )
                select_done.set()

            thread = threading.Thread(target=select_thread)
            thread.start()

            # 等待硬件准备
            time.sleep(0.1)

            # 发送触发
            print("  发送触发信号...")
            trigger_time = time.perf_counter()
            output_task.write(False)  # 下降沿

            # 监控反馈
            feedback_received = False
            start_monitor = time.perf_counter()

            while time.perf_counter() - start_monitor < 1.0:
                if input_task.read():
                    feedback_time = time.perf_counter()
                    feedback_received = True
                    break
                time.sleep(0.0001)

            # 恢复高电平
            output_task.write(True)

            # 等待select完成
            select_done.wait(timeout=5)

            if feedback_received:
                delay_ms = (feedback_time - trigger_time) * 1000
                print(f"  ✓ 收到反馈，延迟: {delay_ms:.3f} ms")
                return True
            else:
                print("  ✗ 未收到反馈")
                return False

    def run_test(self):
        """运行测试"""
        print("=" * 60)
        print("SLM 硬件预加载触发测试")
        print("使用Load_sequence和Select_image函数")
        print("=" * 60)

        # 初始化
        if not self.init_slm():
            return

        # 预加载图像到硬件
        if not self.load_images_to_hardware():
            return

        current_frame = 0

        try:
            while True:
                print("\n" + "-" * 40)
                print(f"当前显示: Frame {current_frame}")
                print("按Enter切换到下一帧，输入'q'退出")

                cmd = input("> ")
                if cmd.lower() == 'q':
                    break

                # 切换到下一帧
                next_frame = 1 - current_frame  # 0->1 或 1->0

                # 选择并触发
                if self.select_and_trigger(next_frame):
                    current_frame = next_frame
                    frame_name = "Central.bmp" if current_frame == 0 else "mlo.bmp"
                    print(f"成功切换到: {frame_name} (Frame {current_frame})")

        finally:
            print("\n清理资源...")
            self.slm_lib.Delete_SDK()
            print("测试结束")


if __name__ == "__main__":
    tester = SLM_Hardware_Preload()
    tester.run_test()