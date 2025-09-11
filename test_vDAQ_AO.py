"""
PCI6110 DAQ 模拟输出测试代码
设备：PCI6110
设备名：Dev2
端口：21-AO1 (模拟输出通道1), 53-AO GND (模拟地)
"""

import nidaqmx
import numpy as np
import time
import sys
from nidaqmx.constants import AcquisitionType
import threading


class AnalogOutputTester:
    def __init__(self, device_name="Dev2", channel="ao1"):
        self.device_name = device_name
        self.channel = channel
        self.task = None
        self.is_running = False
        self.stop_flag = threading.Event()

    def generate_sine_wave(self, frequency, amplitude, offset, duration, sample_rate):
        """生成正弦波数据"""
        samples_per_period = int(sample_rate / frequency)
        num_periods = int(duration * frequency)
        total_samples = samples_per_period * num_periods

        t = np.linspace(0, duration, total_samples, endpoint=False)
        waveform = amplitude * np.sin(2 * np.pi * frequency * t) + offset

        return waveform, sample_rate

    def generate_square_wave(self, frequency, amplitude, offset, duration, sample_rate):
        """生成方波数据"""
        samples_per_period = int(sample_rate / frequency)
        num_periods = int(duration * frequency)
        total_samples = samples_per_period * num_periods

        t = np.linspace(0, duration, total_samples, endpoint=False)
        waveform = amplitude * np.sign(np.sin(2 * np.pi * frequency * t)) + offset

        return waveform, sample_rate

    def generate_triangle_wave(self, frequency, amplitude, offset, duration, sample_rate):
        """生成三角波数据"""
        samples_per_period = int(sample_rate / frequency)
        num_periods = int(duration * frequency)
        total_samples = samples_per_period * num_periods

        t = np.linspace(0, duration, total_samples, endpoint=False)
        waveform = amplitude * (2 * np.arcsin(np.sin(2 * np.pi * frequency * t)) / np.pi) + offset

        return waveform, sample_rate

    def generate_sawtooth_wave(self, frequency, amplitude, offset, duration, sample_rate):
        """生成锯齿波数据"""
        samples_per_period = int(sample_rate / frequency)
        num_periods = int(duration * frequency)
        total_samples = samples_per_period * num_periods

        t = np.linspace(0, duration, total_samples, endpoint=False)
        waveform = amplitude * (2 * (t * frequency - np.floor(t * frequency + 0.5))) + offset

        return waveform, sample_rate

    def output_continuous_waveform(self, waveform_type, frequency, amplitude, offset, sample_rate=10000):
        """持续输出波形 - 用于示波器AUTO功能"""
        print(f"\n开始持续输出{waveform_type}...")
        print(f"频率: {frequency}Hz, 幅度: {amplitude}V, 偏移: {offset}V")
        print("按 Enter 键停止输出...")

        try:
            # 生成1秒的波形数据用于循环
            duration = 1.0

            # 根据波形类型生成数据
            if waveform_type == "正弦波":
                waveform, rate = self.generate_sine_wave(frequency, amplitude, offset, duration, sample_rate)
            elif waveform_type == "方波":
                waveform, rate = self.generate_square_wave(frequency, amplitude, offset, duration, sample_rate)
            elif waveform_type == "三角波":
                waveform, rate = self.generate_triangle_wave(frequency, amplitude, offset, duration, sample_rate)
            elif waveform_type == "锯齿波":
                waveform, rate = self.generate_sawtooth_wave(frequency, amplitude, offset, duration, sample_rate)
            else:
                print("未知波形类型")
                return

            # 创建模拟输出任务
            with nidaqmx.Task() as task:
                # 添加模拟输出通道
                task.ao_channels.add_ao_voltage_chan(
                    f"{self.device_name}/{self.channel}",
                    min_val=-10.0,
                    max_val=10.0
                )

                # 配置定时
                task.timing.cfg_samp_clk_timing(
                    rate=rate,
                    sample_mode=AcquisitionType.CONTINUOUS,
                    samps_per_chan=len(waveform)
                )

                # 写入波形数据
                task.write(waveform, auto_start=False)

                # 开始输出
                task.start()
                self.is_running = True

                # 等待用户按键停止
                input()

                # 停止任务
                task.stop()
                self.is_running = False
                print("波形输出已停止")

        except nidaqmx.errors.DaqError as e:
            print(f"DAQ错误: {e}")
        except Exception as e:
            print(f"错误: {e}")

    def output_single_waveform(self, waveform_type, frequency, amplitude, offset, duration=5, sample_rate=10000):
        """输出单次波形"""
        print(f"\n输出{duration}秒的{waveform_type}...")
        print(f"频率: {frequency}Hz, 幅度: {amplitude}V, 偏移: {offset}V")

        try:
            # 根据波形类型生成数据
            if waveform_type == "正弦波":
                waveform, rate = self.generate_sine_wave(frequency, amplitude, offset, duration, sample_rate)
            elif waveform_type == "方波":
                waveform, rate = self.generate_square_wave(frequency, amplitude, offset, duration, sample_rate)
            elif waveform_type == "三角波":
                waveform, rate = self.generate_triangle_wave(frequency, amplitude, offset, duration, sample_rate)
            elif waveform_type == "锯齿波":
                waveform, rate = self.generate_sawtooth_wave(frequency, amplitude, offset, duration, sample_rate)
            else:
                print("未知波形类型")
                return

            # 创建模拟输出任务
            with nidaqmx.Task() as task:
                # 添加模拟输出通道
                task.ao_channels.add_ao_voltage_chan(
                    f"{self.device_name}/{self.channel}",
                    min_val=-10.0,
                    max_val=10.0
                )

                # 配置定时
                task.timing.cfg_samp_clk_timing(
                    rate=rate,
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=len(waveform)
                )

                # 写入并输出波形
                task.write(waveform, auto_start=True)
                task.wait_until_done(timeout=duration + 1)

                print(f"{waveform_type}输出完成")

        except nidaqmx.errors.DaqError as e:
            print(f"DAQ错误: {e}")
        except Exception as e:
            print(f"错误: {e}")

    def dc_voltage_test(self):
        """直流电压测试 - 输出不同的直流电平"""
        print("\n开始直流电压测试...")
        print("将依次输出: -5V, -2.5V, 0V, 2.5V, 5V")

        voltages = [-5.0, -2.5, 0.0, 2.5, 5.0]

        try:
            with nidaqmx.Task() as task:
                # 添加模拟输出通道
                task.ao_channels.add_ao_voltage_chan(
                    f"{self.device_name}/{self.channel}",
                    min_val=-10.0,
                    max_val=10.0
                )

                for voltage in voltages:
                    print(f"输出: {voltage}V (持续2秒)")
                    task.write(voltage, auto_start=True)
                    time.sleep(2)

                # 最后回到0V
                task.write(0.0, auto_start=True)
                print("直流电压测试完成，输出已设为0V")

        except nidaqmx.errors.DaqError as e:
            print(f"DAQ错误: {e}")
        except Exception as e:
            print(f"错误: {e}")

    def frequency_sweep_test(self):
        """频率扫描测试 - 正弦波从低频到高频"""
        print("\n开始频率扫描测试...")
        print("正弦波频率将从1Hz扫描到100Hz")

        try:
            # 创建扫频信号（10秒内从1Hz到100Hz）
            duration = 10.0
            sample_rate = 10000
            t = np.linspace(0, duration, int(sample_rate * duration))

            # 线性扫频
            f0 = 1  # 起始频率
            f1 = 100  # 结束频率
            k = (f1 - f0) / duration  # 频率变化率

            # 生成扫频信号
            phi = 2 * np.pi * (f0 * t + 0.5 * k * t ** 2)
            waveform = 2.0 * np.sin(phi)  # 2V幅度

            with nidaqmx.Task() as task:
                # 添加模拟输出通道
                task.ao_channels.add_ao_voltage_chan(
                    f"{self.device_name}/{self.channel}",
                    min_val=-10.0,
                    max_val=10.0
                )

                # 配置定时
                task.timing.cfg_samp_clk_timing(
                    rate=sample_rate,
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=len(waveform)
                )

                print("开始10秒扫频输出...")
                # 写入并输出波形
                task.write(waveform, auto_start=True)
                task.wait_until_done(timeout=duration + 1)

                print("频率扫描测试完成")

        except nidaqmx.errors.DaqError as e:
            print(f"DAQ错误: {e}")
        except Exception as e:
            print(f"错误: {e}")


def check_device_analog_output():
    """检查设备模拟输出信息"""
    print("\n" + "=" * 50)
    print("模拟输出通道检查")
    print("=" * 50)

    try:
        system = nidaqmx.system.System.local()

        # 检查设备是否存在
        if "Dev2" not in [device.name for device in system.devices]:
            print("错误: 未找到设备 'Dev2'")
            return False

        # 获取Dev2设备信息
        device = system.devices["Dev2"]
        print(f"设备: {device.name} ({device.product_type})")

        # 显示模拟输出通道信息
        print(f"\n可用的模拟输出通道:")
        for ao_channel in device.ao_physical_chans:
            print(f"  - {ao_channel.name}")

        # 显示模拟输出规格
        if len(device.ao_physical_chans) > 0:
            print(f"\n模拟输出规格:")
            print(f"  电压范围: {device.ao_voltage_rngs}")
            print(f"  最大采样率: {device.ao_max_rate} Hz")
            print(f"  最小采样率: {device.ao_min_rate} Hz")

        print("\n模拟输出通道检查完成！")
        return True

    except Exception as e:
        print(f"检查设备时出错: {e}")
        return False


def main():
    """主测试程序"""
    print("\n" + "=" * 50)
    print("PCI6110 DAQ 模拟输出测试程序")
    print("=" * 50)
    print("设备: PCI6110")
    print("设备名: Dev2")
    print("测试通道: AO1 (Pin 21) -> 示波器")
    print("地线: AO GND (Pin 53) -> 示波器地")
    print("=" * 50)

    # 检查设备
    if not check_device_analog_output():
        print("设备检查失败，请确认设备连接正确")
        sys.exit(1)

    # 创建测试器实例
    tester = AnalogOutputTester(device_name="Dev2", channel="ao1")

    while True:
        print("\n" + "=" * 50)
        print("请选择测试模式:")
        print("=" * 50)
        print("持续输出模式（用于示波器AUTO功能）:")
        print("  1. 持续输出正弦波")
        print("  2. 持续输出方波")
        print("  3. 持续输出三角波")
        print("  4. 持续输出锯齿波")
        print("\n单次输出模式:")
        print("  5. 单次波形输出（5秒）")
        print("  6. 直流电压测试")
        print("  7. 频率扫描测试（1Hz-100Hz）")
        print("\n预设测试:")
        print("  8. 标准测试 (1kHz, 2Vpp正弦波，持续输出)")
        print("  9. 低频测试 (10Hz, 5Vpp正弦波，持续输出)")
        print("\n  0. 退出")

        choice = input("\n请输入选择 (0-9): ").strip()

        if choice == "0":
            print("退出测试程序")
            break

        elif choice in ["1", "2", "3", "4"]:
            # 持续输出模式
            waveform_types = {"1": "正弦波", "2": "方波", "3": "三角波", "4": "锯齿波"}
            waveform_type = waveform_types[choice]

            print(f"\n配置{waveform_type}参数:")
            try:
                freq = float(input("  频率 (Hz) [默认100]: ") or "100")
                amp = float(input("  幅度 (V) [默认2.0]: ") or "2.0")
                offset = float(input("  偏移 (V) [默认0]: ") or "0")

                # 检查参数范围
                if amp + abs(offset) > 10:
                    print("警告: 幅度+偏移超过±10V范围，将被限制")

                tester.output_continuous_waveform(waveform_type, freq, amp, offset)

            except ValueError:
                print("输入参数无效")

        elif choice == "5":
            # 单次波形输出
            print("\n选择波形类型:")
            print("  1. 正弦波")
            print("  2. 方波")
            print("  3. 三角波")
            print("  4. 锯齿波")

            wave_choice = input("选择 (1-4): ").strip()
            if wave_choice in ["1", "2", "3", "4"]:
                waveform_types = {"1": "正弦波", "2": "方波", "3": "三角波", "4": "锯齿波"}
                waveform_type = waveform_types[wave_choice]

                try:
                    freq = float(input("频率 (Hz) [默认100]: ") or "100")
                    amp = float(input("幅度 (V) [默认2.0]: ") or "2.0")
                    offset = float(input("偏移 (V) [默认0]: ") or "0")

                    tester.output_single_waveform(waveform_type, freq, amp, offset)

                except ValueError:
                    print("输入参数无效")

        elif choice == "6":
            # 直流电压测试
            tester.dc_voltage_test()

        elif choice == "7":
            # 频率扫描测试
            tester.frequency_sweep_test()

        elif choice == "8":
            # 标准测试 - 1kHz正弦波
            print("\n标准测试: 1kHz, 2Vpp正弦波")
            tester.output_continuous_waveform("正弦波", 1000, 1.0, 0)

        elif choice == "9":
            # 低频测试 - 10Hz正弦波
            print("\n低频测试: 10Hz, 5Vpp正弦波")
            tester.output_continuous_waveform("正弦波", 10, 2.5, 0)

        else:
            print("无效选择")

    print("\n测试程序结束")


if __name__ == "__main__":
    # 检查是否安装了nidaqmx库
    try:
        import nidaqmx
    except ImportError:
        print("错误: 未安装nidaqmx库")
        print("请运行: pip install nidaqmx")
        sys.exit(1)

    # 运行主程序
    main()