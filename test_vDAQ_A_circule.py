"""
PCI6110 DAQ 模拟回环测试 - 精简版
设备：PCI-6110
设备名：Dev2

引脚连接：
- 模拟输出：Pin 21 (AO1) → Pin 33 (AI1+) 或 Pin 68 (AI0+)
- 地线连接：Pin 54 (AO GND) → Pin 66 (AI1-) 或 Pin 34 (AI0-)

重要配置：
- 使用 PSEUDO_DIFF 模式（伪差分）
- PCI-6110 不支持 RSE 和 NRSE 模式
"""

import nidaqmx
from nidaqmx.constants import TerminalConfiguration
import numpy as np
import time


class PCI6110LoopbackTest:
    """PCI6110 DAQ回环测试类"""

    def __init__(self, device_name="Dev2", ao_channel="ao1", ai_channel="ai1"):
        """
        初始化

        参数:
            device_name: 设备名称，默认"Dev2"
            ao_channel: 模拟输出通道，默认"ao1"
            ai_channel: 模拟输入通道，默认"ai1"
        """
        self.device_name = device_name
        self.ao_channel = ao_channel
        self.ai_channel = ai_channel

    def simple_test(self, test_voltage=2.5):
        """
        简单测试 - 输出一个电压并读取

        参数:
            test_voltage: 测试电压值

        返回:
            (输出电压, 输入电压, 误差)
        """
        ao_task = nidaqmx.Task()
        ai_task = nidaqmx.Task()

        try:
            # 配置模拟输出
            ao_task.ao_channels.add_ao_voltage_chan(
                f"{self.device_name}/{self.ao_channel}",
                min_val=-10.0,
                max_val=10.0
            )

            # 配置模拟输入 - 使用PSEUDO_DIFF模式
            ai_task.ai_channels.add_ai_voltage_chan(
                f"{self.device_name}/{self.ai_channel}",
                terminal_config=TerminalConfiguration.PSEUDO_DIFF,
                min_val=-10.0,
                max_val=10.0
            )

            # 输出电压
            ao_task.write(test_voltage)
            time.sleep(0.05)  # 等待信号稳定

            # 读取输入
            reading = ai_task.read()
            error = abs(test_voltage - reading)

            return test_voltage, reading, error

        finally:
            ao_task.close()
            ai_task.close()

    def dc_sweep_test(self, voltages=None):
        """
        直流扫描测试 - 测试多个电压点

        参数:
            voltages: 测试电压列表，默认[-5, -2.5, 0, 2.5, 5]

        返回:
            测试结果列表 [(输出, 输入, 误差), ...]
        """
        if voltages is None:
            voltages = [-5, -2.5, 0, 2.5, 5]

        ao_task = nidaqmx.Task()
        ai_task = nidaqmx.Task()
        results = []

        try:
            # 配置通道
            ao_task.ao_channels.add_ao_voltage_chan(
                f"{self.device_name}/{self.ao_channel}"
            )

            ai_task.ai_channels.add_ai_voltage_chan(
                f"{self.device_name}/{self.ai_channel}",
                terminal_config=TerminalConfiguration.PSEUDO_DIFF
            )

            # 测试每个电压点
            for v_out in voltages:
                ao_task.write(v_out)
                time.sleep(0.05)

                # 多次读取取平均
                readings = []
                for _ in range(5):
                    readings.append(ai_task.read())
                    time.sleep(0.01)

                v_in = np.mean(readings)
                error = abs(v_out - v_in)
                results.append((v_out, v_in, error))

            return results

        finally:
            ao_task.close()
            ai_task.close()

    def sine_wave_test(self, frequency=100, amplitude=2.0, duration=1.0, sample_rate=10000):
        """
        正弦波测试 - 输出正弦波并采集

        参数:
            frequency: 频率(Hz)
            amplitude: 幅度(V)
            duration: 持续时间(秒)
            sample_rate: 采样率(Hz)

        返回:
            (输出波形, 输入波形, RMS误差, 相关系数)
        """
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples, endpoint=False)
        output_wave = amplitude * np.sin(2 * np.pi * frequency * t)

        ao_task = nidaqmx.Task()
        ai_task = nidaqmx.Task()

        try:
            # 配置模拟输出
            ao_task.ao_channels.add_ao_voltage_chan(
                f"{self.device_name}/{self.ao_channel}"
            )
            ao_task.timing.cfg_samp_clk_timing(
                rate=sample_rate,
                sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                samps_per_chan=len(output_wave)
            )

            # 配置模拟输入
            ai_task.ai_channels.add_ai_voltage_chan(
                f"{self.device_name}/{self.ai_channel}",
                terminal_config=TerminalConfiguration.PSEUDO_DIFF
            )
            ai_task.timing.cfg_samp_clk_timing(
                rate=sample_rate,
                sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                samps_per_chan=len(output_wave)
            )

            # 开始采集
            ai_task.start()

            # 输出波形
            ao_task.write(output_wave, auto_start=True)

            # 读取输入
            input_wave = ai_task.read(number_of_samples_per_channel=len(output_wave))

            # 等待完成
            ao_task.wait_until_done()

            # 计算误差
            rms_error = np.sqrt(np.mean((output_wave - input_wave) ** 2))
            correlation = np.corrcoef(output_wave, input_wave)[0, 1]

            return output_wave, input_wave, rms_error, correlation

        finally:
            ao_task.close()
            ai_task.close()

    def continuous_output(self, frequency=10, amplitude=2.0, sample_rate=10000):
        """
        连续输出正弦波 - 用于示波器观察

        参数:
            frequency: 频率(Hz)
            amplitude: 幅度(V)
            sample_rate: 采样率(Hz)
        """
        # 生成一个周期的波形用于循环
        samples_per_period = int(sample_rate / frequency)
        t = np.linspace(0, 1 / frequency, samples_per_period, endpoint=False)
        waveform = amplitude * np.sin(2 * np.pi * frequency * t)

        ao_task = nidaqmx.Task()

        try:
            # 配置连续输出
            ao_task.ao_channels.add_ao_voltage_chan(
                f"{self.device_name}/{self.ao_channel}"
            )
            ao_task.timing.cfg_samp_clk_timing(
                rate=sample_rate,
                sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS,
                samps_per_chan=len(waveform)
            )

            # 写入波形并开始
            ao_task.write(waveform, auto_start=False)
            ao_task.start()

            print(f"正在输出 {frequency}Hz, {amplitude}Vpp 正弦波...")
            print("按Enter键停止")
            input()

            ao_task.stop()

        finally:
            ao_task.close()


def main():
    """主函数 - 演示用法"""

    # 创建测试对象
    tester = PCI6110LoopbackTest(device_name="Dev2", ao_channel="ao1", ai_channel="ai1")

    print("=" * 60)
    print("PCI6110 DAQ 回环测试")
    print("=" * 60)

    # 1. 简单测试
    print("\n1. 简单测试:")
    v_out, v_in, error = tester.simple_test(3.0)
    print(f"   输出: {v_out:.3f}V, 输入: {v_in:.3f}V, 误差: {error * 1000:.1f}mV")

    # 2. 直流扫描
    print("\n2. 直流扫描测试:")
    results = tester.dc_sweep_test()
    print("   输出(V)  输入(V)  误差(mV)")
    for v_out, v_in, error in results:
        status = "✓" if error < 0.01 else "✗"
        print(f"   {v_out:+6.2f}  {v_in:+6.3f}  {error * 1000:6.1f}  [{status}]")

    # 3. 正弦波测试
    print("\n3. 正弦波测试 (100Hz):")
    _, _, rms_error, correlation = tester.sine_wave_test(frequency=100)
    print(f"   RMS误差: {rms_error * 1000:.2f}mV")
    print(f"   相关系数: {correlation:.4f}")

    # 4. 计算精度
    avg_error = np.mean([e for _, _, e in results])
    print(f"\n总体精度: {(1 - avg_error / 5) * 100:.1f}%")

    # 5. 可选：连续输出
    choice = input("\n是否测试连续输出? (y/n): ")
    if choice.lower() == 'y':
        tester.continuous_output(frequency=100, amplitude=2.0)

    print("\n测试完成！")


if __name__ == "__main__":
    main()