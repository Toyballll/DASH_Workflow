"""
PCI6110 DAQ 数字输出测试代码
设备：PCI6110
设备名：Dev2
端口：52-P0.0 (数字输出), 18-DGND (数字地)
"""

import nidaqmx
import numpy as np
import time
import sys
from nidaqmx.constants import LineGrouping


def test_digital_output_single():
    """测试单点数字输出 - 简单的高低电平切换"""
    print("=" * 50)
    print("测试1: 单点数字输出测试")
    print("=" * 50)

    try:
        # 创建数字输出任务
        with nidaqmx.Task() as task:
            # 添加数字输出通道 - P0.0是端口0的第0线
            task.do_channels.add_do_chan(
                "Dev2/port0/line0",  # P0.0对应port0/line0
                line_grouping=LineGrouping.CHAN_PER_LINE
            )

            print("开始单点输出测试...")
            print("您应该在示波器上看到方波信号")

            # 循环10次，产生方波
            for i in range(10):
                # 输出高电平
                task.write(True)
                print(f"周期 {i + 1}: HIGH")
                time.sleep(0.5)  # 保持500ms

                # 输出低电平
                task.write(False)
                print(f"周期 {i + 1}: LOW")
                time.sleep(0.5)  # 保持500ms

            print("单点输出测试完成！\n")

    except nidaqmx.errors.DaqError as e:
        print(f"DAQ错误: {e}")
        return False
    except Exception as e:
        print(f"未知错误: {e}")
        return False

    return True


def test_digital_output_pattern():
    """测试数字输出模式 - 输出预定义的数字模式"""
    print("=" * 50)
    print("测试2: 数字输出模式测试")
    print("=" * 50)

    try:
        with nidaqmx.Task() as task:
            # 添加数字输出通道
            task.do_channels.add_do_chan(
                "Dev2/port0/line0",
                line_grouping=LineGrouping.CHAN_PER_LINE
            )

            # 创建测试模式：101010...
            pattern = [True, False, True, False, True, False, True, False]

            print("开始模式输出测试...")
            print("输出模式: 10101010")

            # 输出模式3次
            for cycle in range(3):
                print(f"\n周期 {cycle + 1}:")
                for i, value in enumerate(pattern):
                    task.write(value)
                    state = "HIGH" if value else "LOW"
                    print(f"  位 {i}: {state}")
                    time.sleep(0.2)  # 每个状态保持200ms

            # 最后设置为低电平
            task.write(False)
            print("\n模式输出测试完成！\n")

    except nidaqmx.errors.DaqError as e:
        print(f"DAQ错误: {e}")
        return False
    except Exception as e:
        print(f"未知错误: {e}")
        return False

    return True


def test_digital_output_frequency():
    """测试不同频率的数字输出"""
    print("=" * 50)
    print("测试3: 变频数字输出测试")
    print("=" * 50)

    try:
        with nidaqmx.Task() as task:
            # 添加数字输出通道
            task.do_channels.add_do_chan(
                "Dev2/port0/line0",
                line_grouping=LineGrouping.CHAN_PER_LINE
            )

            # 测试不同频率：1Hz, 2Hz, 5Hz, 10Hz
            frequencies = [1, 2, 5, 10]

            print("开始变频输出测试...")
            print("将测试以下频率: 1Hz, 2Hz, 5Hz, 10Hz")

            for freq in frequencies:
                period = 1.0 / freq  # 计算周期
                half_period = period / 2  # 半周期

                print(f"\n频率: {freq}Hz (周期: {period:.3f}秒)")
                print("输出10个周期...")

                for i in range(10):
                    task.write(True)
                    time.sleep(half_period)
                    task.write(False)
                    time.sleep(half_period)

                print(f"{freq}Hz 测试完成")
                time.sleep(1)  # 频率切换间隔

            # 最后设置为低电平
            task.write(False)
            print("\n变频输出测试完成！\n")

    except nidaqmx.errors.DaqError as e:
        print(f"DAQ错误: {e}")
        return False
    except Exception as e:
        print(f"未知错误: {e}")
        return False

    return True


def check_device_info():
    """检查设备信息"""
    print("=" * 50)
    print("设备信息检查")
    print("=" * 50)

    try:
        system = nidaqmx.system.System.local()

        # 检查设备是否存在
        if "Dev2" not in [device.name for device in system.devices]:
            print("错误: 未找到设备 'Dev2'")
            print("可用设备:")
            for device in system.devices:
                print(f"  - {device.name}: {device.product_type}")
            return False

        # 获取Dev2设备信息
        device = system.devices["Dev2"]
        print(f"设备名称: {device.name}")
        print(f"产品类型: {device.product_type}")
        print(f"序列号: {device.dev_serial_num}")

        # 显示数字输出端口信息
        print(f"\n数字输出端口:")
        for do_port in device.do_ports:
            print(f"  - {do_port.name}")

        print(f"\n数字线路:")
        for do_line in device.do_lines:
            print(f"  - {do_line.name}")

        print("\n设备检查完成！\n")
        return True

    except Exception as e:
        print(f"检查设备时出错: {e}")
        return False


def main():
    """主测试程序"""
    print("\n" + "=" * 50)
    print("PCI6110 DAQ 数字输出测试程序")
    print("=" * 50)
    print("设备: PCI6110")
    print("设备名: Dev2")
    print("测试端口: P0.0 (Pin 52) -> 示波器")
    print("地线: DGND (Pin 18) -> 示波器地")
    print("=" * 50 + "\n")

    # 首先检查设备
    if not check_device_info():
        print("设备检查失败，请确认设备连接正确")
        sys.exit(1)

    # 用户选择测试
    while True:
        print("\n请选择测试项目:")
        print("1. 单点数字输出测试 (简单方波)")
        print("2. 数字模式输出测试 (10101010模式)")
        print("3. 变频输出测试 (1Hz-10Hz)")
        print("4. 运行所有测试")
        print("0. 退出")

        choice = input("\n请输入选择 (0-4): ").strip()

        if choice == "0":
            print("退出测试程序")
            break
        elif choice == "1":
            test_digital_output_single()
        elif choice == "2":
            test_digital_output_pattern()
        elif choice == "3":
            test_digital_output_frequency()
        elif choice == "4":
            print("\n运行所有测试...\n")
            test_digital_output_single()
            time.sleep(2)
            test_digital_output_pattern()
            time.sleep(2)
            test_digital_output_frequency()
            print("\n所有测试完成！")
        else:
            print("无效选择，请重新输入")

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