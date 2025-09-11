"""
PCI6110 DAQ 数字回环测试代码
设备：PCI6110
设备名：Dev2
输出端口：17-P0.1 (数字输出), 50-DGND (数字地)
输入端口：49-P0.2 (数字输入), 15-DGND (数字地)

"""

import nidaqmx
import numpy as np
import time
import sys
from nidaqmx.constants import LineGrouping
import threading
from datetime import datetime


class DigitalLoopbackTester:
    def __init__(self, device_name="Dev2"):
        self.device_name = device_name
        self.output_line = "port0/line1"  # P0.1
        self.input_line = "port0/line2"  # P0.2
        self.test_passed = 0
        self.test_failed = 0

    def single_bit_test(self):
        """单比特回环测试 - 测试基本的高低电平"""
        print("\n" + "=" * 50)
        print("测试1: 单比特回环测试")
        print("=" * 50)

        test_values = [True, False, True, False, True]
        test_results = []

        try:
            # 创建输出和输入任务
            with nidaqmx.Task() as output_task, nidaqmx.Task() as input_task:
                # 配置数字输出通道 (P0.1)
                output_task.do_channels.add_do_chan(
                    f"{self.device_name}/{self.output_line}",
                    line_grouping=LineGrouping.CHAN_PER_LINE
                )

                # 配置数字输入通道 (P0.2)
                input_task.di_channels.add_di_chan(
                    f"{self.device_name}/{self.input_line}",
                    line_grouping=LineGrouping.CHAN_PER_LINE
                )

                print("开始单比特测试...")
                print("-" * 40)

                for i, value in enumerate(test_values):
                    # 输出值
                    output_task.write(value)

                    # 短暂延迟确保信号稳定
                    time.sleep(0.01)

                    # 读取输入值
                    read_value = input_task.read()

                    # 检查是否匹配
                    match = (value == read_value)
                    test_results.append(match)

                    # 显示结果
                    output_str = "HIGH" if value else "LOW"
                    input_str = "HIGH" if read_value else "LOW"
                    status = "✓ PASS" if match else "✗ FAIL"

                    print(f"测试 {i + 1}: 输出={output_str:4s} -> 输入={input_str:4s} [{status}]")

                    if match:
                        self.test_passed += 1
                    else:
                        self.test_failed += 1

                # 测试总结
                print("-" * 40)
                if all(test_results):
                    print("✓ 单比特测试通过！")
                    return True
                else:
                    print("✗ 单比特测试失败！")
                    print(f"  通过: {sum(test_results)}/{len(test_results)}")
                    return False

        except nidaqmx.errors.DaqError as e:
            print(f"DAQ错误: {e}")
            return False
        except Exception as e:
            print(f"错误: {e}")
            return False

    def pattern_test(self):
        """模式测试 - 测试特定的数字模式"""
        print("\n" + "=" * 50)
        print("测试2: 数字模式回环测试")
        print("=" * 50)

        # 测试模式
        patterns = [
            [True, False, True, False, True, False],  # 101010
            [False, True, False, True, False, True],  # 010101
            [True, True, False, False, True, True],  # 110011
            [False, False, True, True, False, False],  # 001100
        ]

        pattern_names = ["101010", "010101", "110011", "001100"]

        try:
            with nidaqmx.Task() as output_task, nidaqmx.Task() as input_task:
                # 配置通道
                output_task.do_channels.add_do_chan(
                    f"{self.device_name}/{self.output_line}",
                    line_grouping=LineGrouping.CHAN_PER_LINE
                )

                input_task.di_channels.add_di_chan(
                    f"{self.device_name}/{self.input_line}",
                    line_grouping=LineGrouping.CHAN_PER_LINE
                )

                print("开始模式测试...")
                print("-" * 40)

                all_patterns_passed = True

                for pattern_idx, pattern in enumerate(patterns):
                    pattern_name = pattern_names[pattern_idx]
                    print(f"\n测试模式: {pattern_name}")

                    received_pattern = []
                    pattern_match = True

                    for bit_idx, bit_value in enumerate(pattern):
                        # 输出比特
                        output_task.write(bit_value)
                        time.sleep(0.01)

                        # 读取输入
                        read_value = input_task.read()
                        received_pattern.append(read_value)

                        if bit_value != read_value:
                            pattern_match = False

                    # 将接收到的模式转换为字符串
                    received_str = ''.join(['1' if b else '0' for b in received_pattern])

                    if pattern_match:
                        print(f"  发送: {pattern_name}")
                        print(f"  接收: {received_str}")
                        print(f"  结果: ✓ PASS")
                        self.test_passed += 1
                    else:
                        print(f"  发送: {pattern_name}")
                        print(f"  接收: {received_str}")
                        print(f"  结果: ✗ FAIL")
                        self.test_failed += 1
                        all_patterns_passed = False

                print("-" * 40)
                if all_patterns_passed:
                    print("✓ 模式测试通过！")
                    return True
                else:
                    print("✗ 模式测试失败！")
                    return False

        except Exception as e:
            print(f"错误: {e}")
            return False

    def speed_test(self):
        """速度测试 - 测试不同频率的切换"""
        print("\n" + "=" * 50)
        print("测试3: 速度测试")
        print("=" * 50)

        # 测试不同的切换频率
        test_frequencies = [10, 100, 1000, 5000]  # Hz

        try:
            with nidaqmx.Task() as output_task, nidaqmx.Task() as input_task:
                # 配置通道
                output_task.do_channels.add_do_chan(
                    f"{self.device_name}/{self.output_line}",
                    line_grouping=LineGrouping.CHAN_PER_LINE
                )

                input_task.di_channels.add_di_chan(
                    f"{self.device_name}/{self.input_line}",
                    line_grouping=LineGrouping.CHAN_PER_LINE
                )

                print("开始速度测试...")
                print("-" * 40)

                all_freq_passed = True

                for freq in test_frequencies:
                    period = 1.0 / freq
                    half_period = period / 2
                    cycles = min(freq, 100)  # 测试周期数（最多100个）

                    print(f"\n频率: {freq}Hz (测试{cycles}个周期)")

                    mismatches = 0
                    for i in range(cycles):
                        # 输出高电平
                        output_task.write(True)
                        time.sleep(half_period)
                        high_read = input_task.read()

                        # 输出低电平
                        output_task.write(False)
                        time.sleep(half_period)
                        low_read = input_task.read()

                        # 检查读取值
                        if not high_read or low_read:
                            mismatches += 1

                    error_rate = (mismatches / (cycles * 2)) * 100

                    if mismatches == 0:
                        print(f"  结果: ✓ PASS (错误率: 0%)")
                        self.test_passed += 1
                    else:
                        print(f"  结果: ✗ FAIL (错误率: {error_rate:.1f}%)")
                        print(f"  错误: {mismatches}/{cycles * 2}")
                        self.test_failed += 1
                        all_freq_passed = False

                print("-" * 40)
                if all_freq_passed:
                    print("✓ 速度测试通过！")
                    return True
                else:
                    print("✗ 速度测试部分失败！")
                    return False

        except Exception as e:
            print(f"错误: {e}")
            return False

    def continuous_monitoring(self):
        """连续监控测试 - 实时显示输入状态"""
        print("\n" + "=" * 50)
        print("测试4: 连续监控测试")
        print("=" * 50)
        print("此测试将手动控制输出并实时监控输入")
        print("命令: 'h'=高电平, 'l'=低电平, 't'=切换, 'q'=退出")
        print("-" * 40)

        try:
            with nidaqmx.Task() as output_task, nidaqmx.Task() as input_task:
                # 配置通道
                output_task.do_channels.add_do_chan(
                    f"{self.device_name}/{self.output_line}",
                    line_grouping=LineGrouping.CHAN_PER_LINE
                )

                input_task.di_channels.add_di_chan(
                    f"{self.device_name}/{self.input_line}",
                    line_grouping=LineGrouping.CHAN_PER_LINE
                )

                current_output = False
                output_task.write(current_output)

                # 创建监控线程
                stop_monitor = threading.Event()

                def monitor_input():
                    while not stop_monitor.is_set():
                        try:
                            input_value = input_task.read()
                            input_str = "HIGH" if input_value else "LOW"
                            output_str = "HIGH" if current_output else "LOW"
                            match = "✓" if input_value == current_output else "✗"

                            # 使用\r实现同行更新
                            print(f"\r输出: {output_str:4s} | 输入: {input_str:4s} | 匹配: {match}  ", end='')

                        except:
                            pass
                        time.sleep(0.1)

                # 启动监控线程
                monitor_thread = threading.Thread(target=monitor_input)
                monitor_thread.start()

                print("\n开始监控（输入命令）：")

                while True:
                    # 获取用户输入
                    cmd = input("\n命令> ").strip().lower()

                    if cmd == 'q':
                        break
                    elif cmd == 'h':
                        current_output = True
                        output_task.write(current_output)
                        print("设置输出为: HIGH")
                    elif cmd == 'l':
                        current_output = False
                        output_task.write(current_output)
                        print("设置输出为: LOW")
                    elif cmd == 't':
                        current_output = not current_output
                        output_task.write(current_output)
                        state = "HIGH" if current_output else "LOW"
                        print(f"切换输出为: {state}")
                    else:
                        print("无效命令！使用: h=高, l=低, t=切换, q=退出")

                # 停止监控线程
                stop_monitor.set()
                monitor_thread.join()

                print("\n监控测试结束")
                return True

        except Exception as e:
            print(f"错误: {e}")
            return False

    def stress_test(self):
        """压力测试 - 长时间高速切换"""
        print("\n" + "=" * 50)
        print("测试5: 压力测试")
        print("=" * 50)

        duration = 10  # 测试持续时间（秒）

        try:
            with nidaqmx.Task() as output_task, nidaqmx.Task() as input_task:
                # 配置通道
                output_task.do_channels.add_do_chan(
                    f"{self.device_name}/{self.output_line}",
                    line_grouping=LineGrouping.CHAN_PER_LINE
                )

                input_task.di_channels.add_di_chan(
                    f"{self.device_name}/{self.input_line}",
                    line_grouping=LineGrouping.CHAN_PER_LINE
                )

                print(f"开始{duration}秒压力测试...")
                print("将进行快速切换并统计错误...")
                print("-" * 40)

                start_time = time.time()
                total_switches = 0
                errors = 0

                while (time.time() - start_time) < duration:
                    # 快速切换
                    for value in [True, False]:
                        output_task.write(value)
                        time.sleep(0.001)  # 1ms延迟
                        read_value = input_task.read()

                        if value != read_value:
                            errors += 1

                        total_switches += 1

                    # 每秒更新一次状态
                    if int(time.time() - start_time) % 1 == 0:
                        elapsed = int(time.time() - start_time)
                        print(f"\r进度: {elapsed}/{duration}秒 | 切换次数: {total_switches} | 错误: {errors}", end='')

                # 计算统计信息
                print(f"\n" + "-" * 40)
                error_rate = (errors / total_switches * 100) if total_switches > 0 else 0
                switches_per_sec = total_switches / duration

                print(f"测试完成统计:")
                print(f"  总切换次数: {total_switches}")
                print(f"  错误次数: {errors}")
                print(f"  错误率: {error_rate:.2f}%")
                print(f"  平均速度: {switches_per_sec:.0f} 切换/秒")

                if error_rate < 1.0:
                    print("✓ 压力测试通过！")
                    self.test_passed += 1
                    return True
                else:
                    print("✗ 压力测试失败（错误率过高）！")
                    self.test_failed += 1
                    return False

        except Exception as e:
            print(f"错误: {e}")
            return False


def check_loopback_connection():
    """检查回环连接状态"""
    print("\n" + "=" * 50)
    print("回环连接检查")
    print("=" * 50)

    try:
        with nidaqmx.Task() as output_task, nidaqmx.Task() as input_task:
            # 配置通道
            output_task.do_channels.add_do_chan(
                "Dev2/port0/line1",
                line_grouping=LineGrouping.CHAN_PER_LINE
            )

            input_task.di_channels.add_di_chan(
                "Dev2/port0/line2",
                line_grouping=LineGrouping.CHAN_PER_LINE
            )

            # 测试连接
            print("检查物理连接...")

            # 输出高电平
            output_task.write(True)
            time.sleep(0.1)
            high_read = input_task.read()

            # 输出低电平
            output_task.write(False)
            time.sleep(0.1)
            low_read = input_task.read()

            if high_read and not low_read:
                print("✓ 回环连接正常！")
                print("  P0.1 (Pin 17) -> P0.2 (Pin 49) 连接确认")
                return True
            else:
                print("✗ 回环连接异常！")
                print("\n请检查接线：")
                print("  1. Pin 17 (P0.1) 应连接到 Pin 49 (P0.2)")
                print("  2. Pin 50 (DGND) 应连接到 Pin 15 (DGND)")
                print(f"\n当前读数: 高电平测试={high_read}, 低电平测试={low_read}")

                if not high_read and not low_read:
                    print("  可能原因: 输入始终为低，检查是否正确连接")
                elif high_read and low_read:
                    print("  可能原因: 输入始终为高，可能有短路")

                return False

    except Exception as e:
        print(f"连接检查错误: {e}")
        return False


def main():
    """主测试程序"""
    print("\n" + "=" * 60)
    print("PCI6110 DAQ 数字回环测试程序")
    print("=" * 60)
    print("设备: PCI6110 (Dev2)")
    print("输出: P0.1 (Pin 17) + DGND (Pin 50)")
    print("输入: P0.2 (Pin 49) + DGND (Pin 15)")
    print("\n重要：请确保 Pin 17 已物理连接到 Pin 49！")
    print("      请确保 Pin 50 已物理连接到 Pin 15！")
    print("=" * 60)

    # 首先检查连接
    print("\n步骤1: 检查回环连接")
    if not check_loopback_connection():
        print("\n回环连接未就绪。请检查接线后重试。")
        response = input("\n是否已完成接线并继续？(y/n): ")
        if response.lower() != 'y':
            print("测试中止")
            sys.exit(1)
        else:
            # 再次检查
            if not check_loopback_connection():
                print("\n连接仍有问题，请仔细检查接线")
                sys.exit(1)

    # 创建测试器
    tester = DigitalLoopbackTester()

    while True:
        print("\n" + "=" * 50)
        print("请选择测试项目:")
        print("=" * 50)
        print("1. 单比特回环测试（基础）")
        print("2. 数字模式测试（中级）")
        print("3. 速度测试（不同频率）")
        print("4. 连续监控（手动控制）")
        print("5. 压力测试（10秒高速切换）")
        print("6. 运行全部自动测试（1-3,5）")
        print("7. 查看测试统计")
        print("0. 退出")

        choice = input("\n请选择 (0-7): ").strip()

        if choice == "0":
            print("\n测试结束")
            break

        elif choice == "1":
            tester.single_bit_test()

        elif choice == "2":
            tester.pattern_test()

        elif choice == "3":
            tester.speed_test()

        elif choice == "4":
            tester.continuous_monitoring()

        elif choice == "5":
            tester.stress_test()

        elif choice == "6":
            print("\n" + "=" * 50)
            print("运行全部自动测试")
            print("=" * 50)

            tests = [
                ("单比特测试", tester.single_bit_test),
                ("模式测试", tester.pattern_test),
                ("速度测试", tester.speed_test),
                ("压力测试", tester.stress_test),
            ]

            results = []
            for test_name, test_func in tests:
                print(f"\n正在运行: {test_name}")
                result = test_func()
                results.append((test_name, result))
                time.sleep(1)

            # 显示总结
            print("\n" + "=" * 50)
            print("测试总结")
            print("=" * 50)
            for test_name, result in results:
                status = "✓ PASS" if result else "✗ FAIL"
                print(f"{test_name:15s}: {status}")

            total_pass = sum(1 for _, r in results if r)
            print(f"\n总计: {total_pass}/{len(results)} 测试通过")

        elif choice == "7":
            print("\n" + "=" * 50)
            print("测试统计")
            print("=" * 50)
            total = tester.test_passed + tester.test_failed
            if total > 0:
                pass_rate = (tester.test_passed / total) * 100
                print(f"总测试数: {total}")
                print(f"通过: {tester.test_passed}")
                print(f"失败: {tester.test_failed}")
                print(f"通过率: {pass_rate:.1f}%")
            else:
                print("还未运行任何测试")

        else:
            print("无效选择")

    # 最终统计
    if tester.test_passed + tester.test_failed > 0:
        print("\n" + "=" * 50)
        print("最终测试报告")
        print("=" * 50)
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"设备: PCI6110 (Dev2)")
        print(f"总测试数: {tester.test_passed + tester.test_failed}")
        print(f"通过: {tester.test_passed}")
        print(f"失败: {tester.test_failed}")
        pass_rate = (tester.test_passed / (tester.test_passed + tester.test_failed)) * 100
        print(f"通过率: {pass_rate:.1f}%")

        if pass_rate == 100:
            print("\n结论: ✓ 所有测试通过，数字I/O功能正常！")
        elif pass_rate >= 80:
            print("\n结论: ⚠ 大部分测试通过，但存在一些问题")
        else:
            print("\n结论: ✗ 测试失败较多，请检查硬件连接")


if __name__ == "__main__":
    # 检查nidaqmx库
    try:
        import nidaqmx
    except ImportError:
        print("错误: 未安装nidaqmx库")
        print("请运行: pip install nidaqmx")
        sys.exit(1)

    # 运行主程序
    main()