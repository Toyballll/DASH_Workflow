"""
PMT信号稳定监控程序
使用单点读取模式，适合长时间运行
"""

import nidaqmx
from nidaqmx.constants import TerminalConfiguration
import numpy as np
import time
from collections import deque


def monitor_pmt_simple(channel="ai0", duration=None):
    """
    简单稳定的PMT监控 - 使用单点读取

    参数:
        channel: AI通道
        duration: 持续时间(秒)，None表示无限
    """
    print("PMT信号监控（稳定版）")
    print("=" * 50)
    print(f"通道: {channel}")
    print("按Ctrl+C停止")
    print("=" * 50)

    # 创建任务
    task = nidaqmx.Task()
    task.ai_channels.add_ai_voltage_chan(
        f"Dev2/{channel}",
        terminal_config=TerminalConfiguration.PSEUDO_DIFF,
        min_val=-10.0,
        max_val=10.0
    )

    # 数据缓存
    history = deque(maxlen=100)  # 保存最近100个点
    start_time = time.time()
    count = 0

    try:
        while True:
            # 检查时间
            elapsed = time.time() - start_time
            if duration and elapsed > duration:
                break

            # 单点读取 - 永远不会溢出
            value = task.read()
            history.append(value)
            count += 1

            # 计算统计
            if len(history) > 0:
                avg = np.mean(history)
                max_val = np.max(history)
                min_val = np.min(history)
                std = np.std(history) if len(history) > 1 else 0

                # 显示
                print(f"\r时间: {elapsed:7.1f}s | "
                      f"当前: {value:+8.5f}V | "
                      f"平均: {avg:+8.5f}V | "
                      f"峰峰: {max_val - min_val:7.5f}V | "
                      f"噪声: {std * 1000:6.3f}mV", end='')

            # 控制更新频率
            time.sleep(0.1)  # 10Hz更新率

    except KeyboardInterrupt:
        print("\n\n监控停止")
    finally:
        print(f"\n总采样: {count}")
        print(f"总时间: {time.time() - start_time:.1f}秒")
        task.close()


def monitor_pmt_fast(channel="ai0", sample_rate=1000, duration=10):
    """
    快速PMT采集 - 用于短时间高速采集

    参数:
        channel: AI通道
        sample_rate: 采样率
        duration: 持续时间(秒)，建议<60秒
    """
    print(f"快速采集 {duration}秒 @ {sample_rate}Hz")
    print("=" * 50)

    # 计算总样本数
    total_samples = int(sample_rate * duration)

    task = nidaqmx.Task()
    task.ai_channels.add_ai_voltage_chan(
        f"Dev2/{channel}",
        terminal_config=TerminalConfiguration.PSEUDO_DIFF,
        min_val=-10.0,
        max_val=10.0
    )

    # 配置定时采集（有限样本）
    task.timing.cfg_samp_clk_timing(
        rate=sample_rate,
        sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
        samps_per_chan=total_samples
    )

    try:
        print("采集中...")
        start_time = time.time()

        # 一次性读取所有数据
        data = task.read(number_of_samples_per_channel=total_samples)

        elapsed = time.time() - start_time

        # 统计
        print(f"\n采集完成！")
        print(f"实际时间: {elapsed:.2f}秒")
        print(f"样本数: {len(data)}")
        print(f"平均值: {np.mean(data):.5f}V")
        print(f"最大值: {np.max(data):.5f}V")
        print(f"最小值: {np.min(data):.5f}V")
        print(f"标准差: {np.std(data) * 1000:.3f}mV")

        return data

    finally:
        task.close()


def monitor_pmt_chunked(channel="ai0", sample_rate=1000):
    """
    分块采集模式 - 平衡速度和稳定性

    参数:
        channel: AI通道
        sample_rate: 采样率
    """
    print("PMT分块采集监控")
    print("=" * 50)
    print(f"通道: {channel}")
    print(f"采样率: {sample_rate}Hz")
    print("按Ctrl+C停止")
    print("=" * 50)

    # 每次采集0.5秒的数据
    chunk_duration = 0.5
    chunk_size = int(sample_rate * chunk_duration)

    all_data = deque(maxlen=sample_rate * 10)  # 保存10秒数据
    start_time = time.time()
    chunk_count = 0

    try:
        while True:
            # 创建新任务进行一次采集
            task = nidaqmx.Task()
            task.ai_channels.add_ai_voltage_chan(
                f"Dev2/{channel}",
                terminal_config=TerminalConfiguration.PSEUDO_DIFF,
                min_val=-10.0,
                max_val=10.0
            )

            task.timing.cfg_samp_clk_timing(
                rate=sample_rate,
                sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                samps_per_chan=chunk_size
            )

            # 采集一块数据
            data = task.read(number_of_samples_per_channel=chunk_size)
            task.close()

            # 更新缓存
            all_data.extend(data)
            chunk_count += 1

            # 统计
            if len(all_data) > 0:
                current = data[-1]
                avg = np.mean(all_data)
                max_val = np.max(all_data)
                min_val = np.min(all_data)
                std = np.std(all_data)

                elapsed = time.time() - start_time
                print(f"\r时间: {elapsed:7.1f}s | "
                      f"块: {chunk_count:4d} | "
                      f"当前: {current:+8.5f}V | "
                      f"平均: {avg:+8.5f}V | "
                      f"噪声: {std * 1000:6.3f}mV", end='')

    except KeyboardInterrupt:
        print("\n\n监控停止")
        print(f"总块数: {chunk_count}")
        print(f"总样本: {chunk_count * chunk_size}")
        print(f"总时间: {time.time() - start_time:.1f}秒")


def main():
    """主函数"""
    print("=" * 60)
    print("PMT稳定监控程序")
    print("=" * 60)

    while True:
        print("\n选择监控模式：")
        print("1. 稳定监控（单点读取，可运行数小时）")
        print("2. 快速采集（高速，限时）")
        print("3. 分块采集（中等速度，较稳定）")
        print("0. 退出")

        choice = input("\n选择: ").strip()

        if choice == "0":
            break

        elif choice == "1":
            # 稳定监控
            try:
                monitor_pmt_simple(channel="ai0", duration=None)
            except Exception as e:
                print(f"\n错误: {e}")

        elif choice == "2":
            # 快速采集
            try:
                duration = float(input("采集时长(秒) [10]: ") or "10")
                rate = int(input("采样率(Hz) [1000]: ") or "1000")

                if duration > 60:
                    print("警告：时间过长可能导致内存问题，建议<60秒")
                    if input("继续? (y/n): ").lower() != 'y':
                        continue

                data = monitor_pmt_fast(channel="ai0",
                                        sample_rate=rate,
                                        duration=duration)

                # 可选：保存数据
                save = input("\n保存数据? (y/n): ")
                if save.lower() == 'y':
                    filename = input("文件名 [pmt_data.txt]: ") or "pmt_data.txt"
                    np.savetxt(filename, data, fmt='%.6f')
                    print(f"已保存到 {filename}")

            except Exception as e:
                print(f"\n错误: {e}")

        elif choice == "3":
            # 分块采集
            try:
                rate = int(input("采样率(Hz) [1000]: ") or "1000")
                monitor_pmt_chunked(channel="ai0", sample_rate=rate)
            except Exception as e:
                print(f"\n错误: {e}")

        else:
            print("无效选择")

    print("\n程序结束")


if __name__ == "__main__":
    main()