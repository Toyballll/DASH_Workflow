
import nidaqmx
from nidaqmx.constants import (
    AcquisitionType, TerminalConfiguration, VoltageUnits
)
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import queue
from datetime import datetime


class PMTSignalReader:
    def __init__(self, pmt_channel='vDAQ0/ai0', sample_rate=10000):
        """
        Args:
            pmt_channel: PMT信号输入通道，格式如 'vDAQ0/ai0'
                        需要根据实际硬件连接确定正确的AI通道
            sample_rate: 采样率 (Hz)
        """
        self.pmt_channel = pmt_channel
        self.sample_rate = sample_rate
        self.samples_per_read = int(sample_rate / 10)  # 每次读取100ms的数据

        # 数据缓冲
        self.data_queue = queue.Queue(maxsize=100)
        self.is_running = False

        # 用于存储采集的数据
        self.all_data = []
        self.all_timestamps = []

        print(f"PMT信号读取器初始化")
        print(f"  - 通道: {pmt_channel}")
        print(f"  - 采样率: {sample_rate} Hz")

    def test_connection(self):
        """
        测试是否能够读取PMT信号
        """
        try:
            with nidaqmx.Task() as task:
                task.ai_channels.add_ai_voltage_chan(
                    self.pmt_channel,
                    terminal_config=TerminalConfiguration.RSE,
                    min_val=-10.0,
                    max_val=10.0
                )

                # 读取单个样本
                value = task.read()
                print(f"连接测试成功！当前读数: {value:.4f} V")
                return True

        except Exception as e:
            print(f"连接测试失败: {e}")
            print("请检查：")
            print("  1. PMT信号通道是否正确（可能是ai1, ai2等）")
            print("  2. ScanImage是否正在运行")
            print("  3. PMT是否已开启")
            return False

    def single_read(self, duration=1.0):
        """
        单次读取指定时长的PMT信号

        Args:
            duration: 读取时长（秒）

        Returns:
            tuple: (时间数组, 信号数组)
        """
        samples_to_read = int(self.sample_rate * duration)

        try:
            with nidaqmx.Task() as task:
                task.ai_channels.add_ai_voltage_chan(
                    self.pmt_channel,
                    terminal_config=TerminalConfiguration.RSE,
                    min_val=-10.0,
                    max_val=10.0
                )

                task.timing.cfg_samp_clk_timing(
                    self.sample_rate,
                    sample_mode=AcquisitionType.FINITE,
                    samps_per_chan=samples_to_read
                )

                print(f"读取{duration}秒数据...")
                data = task.read(
                    number_of_samples_per_channel=samples_to_read,
                    timeout=duration + 2.0
                )

                time_array = np.linspace(0, duration, samples_to_read)
                signal_array = np.array(data)

                print(f"读取完成: {len(signal_array)}个样本")
                print(f"  - 最小值: {signal_array.min():.4f} V")
                print(f"  - 最大值: {signal_array.max():.4f} V")
                print(f"  - 平均值: {signal_array.mean():.4f} V")
                print(f"  - 标准差: {signal_array.std():.4f} V")

                return time_array, signal_array

        except Exception as e:
            print(f"读取失败: {e}")
            return None, None

    def continuous_acquisition(self, save_data=False):
        """
        连续采集PMT信号（后台线程运行）

        Args:
            save_data: 是否保存所有采集的数据
        """
        self.is_running = True
        self.start_time = time.time()

        if save_data:
            self.all_data = []
            self.all_timestamps = []

        try:
            with nidaqmx.Task() as task:
                task.ai_channels.add_ai_voltage_chan(
                    self.pmt_channel,
                    terminal_config=TerminalConfiguration.RSE,
                    min_val=-10.0,
                    max_val=10.0
                )

                task.timing.cfg_samp_clk_timing(
                    self.sample_rate,
                    sample_mode=AcquisitionType.CONTINUOUS,
                    samps_per_chan=self.samples_per_read
                )

                print(f"开始连续采集 (采样率: {self.sample_rate} Hz)")

                while self.is_running:
                    try:
                        # 读取数据
                        data = task.read(
                            number_of_samples_per_channel=self.samples_per_read,
                            timeout=1.0
                        )

                        current_time = time.time() - self.start_time

                        # 保存数据
                        if save_data:
                            self.all_data.extend(data)
                            timestamps = np.linspace(
                                current_time - self.samples_per_read / self.sample_rate,
                                current_time,
                                self.samples_per_read
                            )
                            self.all_timestamps.extend(timestamps)

                        # 放入队列供实时显示使用
                        if not self.data_queue.full():
                            self.data_queue.put({
                                'data': data,
                                'timestamp': current_time
                            })

                    except nidaqmx.errors.DaqError as e:
                        if "timeout" not in str(e).lower():
                            print(f"采集错误: {e}")
                            break

        except Exception as e:
            print(f"采集失败: {e}")
        finally:
            self.is_running = False
            elapsed_time = time.time() - self.start_time
            print(f"采集已停止 (运行时间: {elapsed_time:.1f}秒)")

            if save_data and self.all_data:
                print(f"共采集了 {len(self.all_data)} 个样本")

    def start_acquisition_thread(self, save_data=False):
        """
        在后台线程中启动连续采集

        Args:
            save_data: 是否保存所有数据
        """
        self.acquisition_thread = threading.Thread(
            target=self.continuous_acquisition,
            args=(save_data,)
        )
        self.acquisition_thread.daemon = True
        self.acquisition_thread.start()
        print("后台采集线程已启动")

    def stop_acquisition(self):
        """
        停止采集
        """
        print("正在停止采集...")
        self.is_running = False
        if hasattr(self, 'acquisition_thread'):
            self.acquisition_thread.join(timeout=2)

    def get_latest_data(self, timeout=0.1):
        """
        获取队列中的最新数据
        """
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def save_to_file(self, filename=None):
        """
        将采集的数据保存到文件

        Args:
            filename: 文件名，默认使用时间戳
        """
        if not self.all_data:
            print("没有数据可保存")
            return

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pmt_data_{timestamp}.npz"

        np.savez(filename,
                 timestamps=np.array(self.all_timestamps),
                 data=np.array(self.all_data),
                 sample_rate=self.sample_rate,
                 channel=self.pmt_channel)

        print(f"数据已保存到: {filename}")
        return filename


class RealTimePlotter:
    """
    实时绘图器
    """

    def __init__(self, reader, window_size=5.0):
        """
        Args:
            reader: PMTSignalReader实例
            window_size: 显示窗口大小（秒）
        """
        self.reader = reader
        self.window_size = window_size

        # 创建图形
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 信号图
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=0.5)
        self.ax1.set_xlabel('时间 (s)')
        self.ax1.set_ylabel('PMT信号 (V)')
        self.ax1.set_title('实时PMT信号')
        self.ax1.grid(True, alpha=0.3)

        # 直方图
        self.ax2.set_xlabel('信号强度 (V)')
        self.ax2.set_ylabel('计数')
        self.ax2.set_title('信号分布')
        self.ax2.grid(True, alpha=0.3)

        # 数据缓冲
        self.time_buffer = []
        self.data_buffer = []
        self.max_points = int(window_size * reader.sample_rate)

    def update_plot(self, frame):
        """
        更新绘图
        """
        # 获取最新数据
        latest = self.reader.get_latest_data(timeout=0.01)

        if latest is not None:
            data = latest['data']
            current_time = latest['timestamp']

            # 生成时间戳
            n_samples = len(data)
            time_stamps = np.linspace(
                current_time - n_samples / self.reader.sample_rate,
                current_time,
                n_samples
            )

            # 更新缓冲区
            self.time_buffer.extend(time_stamps)
            self.data_buffer.extend(data)

            # 限制缓冲区大小
            if len(self.time_buffer) > self.max_points:
                self.time_buffer = self.time_buffer[-self.max_points:]
                self.data_buffer = self.data_buffer[-self.max_points:]

            # 更新信号图
            self.line1.set_data(self.time_buffer, self.data_buffer)
            self.ax1.set_xlim(max(0, current_time - self.window_size), current_time)

            if self.data_buffer:
                data_array = np.array(self.data_buffer)
                margin = 0.1 * (data_array.max() - data_array.min())
                if margin == 0:
                    margin = 0.1
                self.ax1.set_ylim(data_array.min() - margin, data_array.max() + margin)

                # 更新直方图
                self.ax2.clear()
                self.ax2.hist(data_array, bins=50, alpha=0.7, color='blue', edgecolor='black')
                self.ax2.set_xlabel('信号强度 (V)')
                self.ax2.set_ylabel('计数')
                self.ax2.set_title(f'信号分布 (平均: {data_array.mean():.3f}V, 标准差: {data_array.std():.3f}V)')
                self.ax2.grid(True, alpha=0.3)

        return self.line1,

    def start(self):
        """
        启动实时绘图
        """
        self.ani = FuncAnimation(
            self.fig, self.update_plot,
            interval=100,  # 100ms更新一次
            blit=False,
            cache_frame_data=False
        )
        plt.tight_layout()
        plt.show()


# 使用示例
if __name__ == "__main__":
    # 配置参数 - 请根据实际情况修改
    PMT_CHANNEL = 'vDAQ0/ai0'  # 修改为你的PMT信号通道
    SAMPLE_RATE = 10000  # Hz

    print("=" * 50)
    print("PMT信号读取器 - 与ScanImage并行工作")
    print("=" * 50)

    # 创建读取器
    reader = PMTSignalReader(PMT_CHANNEL, SAMPLE_RATE)

    # 测试连接
    print("\n测试连接...")
    if not reader.test_connection():
        print("连接失败，请检查:")
        print(f"  1. PMT通道 {PMT_CHANNEL} 是否正确")
        print("  2. ScanImage是否正在运行")
        print("  3. PMT是否已开启")
        exit(1)

    print("\n选择模式:")
    print("1. 单次读取")
    print("2. 实时监控")
    print("3. 长时间记录")

    choice = input("请选择 (1/2/3): ").strip()

    try:
        if choice == '1':
            # 单次读取模式
            duration = float(input("读取时长(秒): ") or "2.0")
            time_array, signal_array = reader.single_read(duration)

            if signal_array is not None:
                # 绘制结果
                plt.figure(figsize=(12, 6))
                plt.plot(time_array, signal_array, 'b-', linewidth=0.5)
                plt.xlabel('时间 (s)')
                plt.ylabel('PMT信号 (V)')
                plt.title('PMT信号采集结果')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()

        elif choice == '2':
            # 实时监控模式
            print("\n启动实时监控（按Ctrl+C停止）...")
            reader.start_acquisition_thread(save_data=False)

            # 创建实时绘图器
            plotter = RealTimePlotter(reader, window_size=5.0)
            plotter.start()

        elif choice == '3':
            # 长时间记录模式
            duration = float(input("记录时长(秒): ") or "60")
            print(f"\n开始记录{duration}秒数据...")

            reader.start_acquisition_thread(save_data=True)
            time.sleep(duration)
            reader.stop_acquisition()

            # 保存数据
            filename = reader.save_to_file()

            # 绘制采集的所有数据
            if reader.all_data:
                plt.figure(figsize=(14, 8))

                # 信号图
                plt.subplot(2, 1, 1)
                plt.plot(reader.all_timestamps, reader.all_data, 'b-', linewidth=0.5)
                plt.xlabel('时间 (s)')
                plt.ylabel('PMT信号 (V)')
                plt.title('完整信号记录')
                plt.grid(True, alpha=0.3)

                # 统计图
                plt.subplot(2, 1, 2)
                plt.hist(reader.all_data, bins=100, alpha=0.7, color='blue', edgecolor='black')
                plt.xlabel('信号强度 (V)')
                plt.ylabel('计数')
                plt.title('信号分布统计')
                plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.show()

    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        reader.stop_acquisition()
        print("程序结束")