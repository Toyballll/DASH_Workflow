"""
P95 可靠性测试 - 多功率 × 多次重复测量
- 重点测试窗口: 3ms, 5ms, 10ms, 50ms
- 验证 P95 信号值的稳定性
"""

import numpy as np
import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
import matplotlib.pyplot as plt
from datetime import datetime

# 中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================
# 配置参数
# ============================================
device_name = "Dev2"
pmt_channel = "ai0"
sample_rate = 10000  # 10 kHz
segment_time = 0.5  # 每次测量 0.5 秒
segment_samples = int(sample_rate * segment_time)

# 信号周期
SIGNAL_FREQ = 435  # Hz
SIGNAL_PERIOD_MS = 1000 / SIGNAL_FREQ  # ~2.3 ms

# 重点测试的窗口大小 (ms) - 只测试这4个
window_sizes_ms = [1.5, 3, 5, 10]

print("=" * 70)
print("P95 Stability Test - Focus on 3ms, 5ms, 10ms, 50ms windows")
print("=" * 70)
print(f"Sample rate: {sample_rate} Hz")
print(f"Single measurement duration: {segment_time} s")
print(f"Signal period: {SIGNAL_PERIOD_MS:.2f} ms")
print(f"Test windows: {window_sizes_ms} ms")
print("=" * 70)


# ============================================
# 采集函数
# ============================================
def acquire_data(device_name, channel, sample_rate, n_samples, timeout):
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(
            f"{device_name}/{channel}",
            terminal_config=TerminalConfiguration.DEFAULT,
            min_val=-10.0, max_val=10.0
        )
        task.timing.cfg_samp_clk_timing(
            rate=sample_rate,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=n_samples
        )
        task.in_stream.input_buf_size = n_samples * 2
        task.start()
        task.wait_until_done(timeout=timeout)
        data = np.array(task.read(
            number_of_samples_per_channel=n_samples,
            timeout=10.0
        ))
    return data


# ============================================
# 特征提取函数
# ============================================
def extract_features_from_segment(data, window_ms, sample_rate):
    """从一段数据中，按窗口提取特征的统计值"""
    window_samples = max(1, int(window_ms * sample_rate / 1000))
    n_windows = len(data) // window_samples

    feat_values = {'mean': [], 'p90': [], 'p95': [], 'p99': []}

    for i in range(n_windows):
        window_data = data[i * window_samples: (i + 1) * window_samples]
        feat_values['mean'].append(np.mean(window_data))
        feat_values['p90'].append(np.percentile(window_data, 90))
        feat_values['p95'].append(np.percentile(window_data, 95))
        feat_values['p99'].append(np.percentile(window_data, 99))

    return {feat: np.mean(vals) for feat, vals in feat_values.items()}


# ============================================
# 采集多个功率 × 多次重复
# ============================================
print("\nHow many power levels to test?")
n_powers = int(input("Number of power levels: "))

print("How many repeats per power level?")
n_repeats = int(input("Number of repeats (e.g., 5): "))

all_data = {}
power_list = []

for i in range(n_powers):
    power = input(f"\nEnter power level {i + 1}/{n_powers} (e.g., '2.0%'): ").strip()
    power_list.append(power)
    all_data[power] = []

    print(f"\nSet laser to {power}, press Enter to start {n_repeats} measurements...")
    input()

    for rep in range(n_repeats):
        print(f"  Acquiring {power} - repeat {rep + 1}/{n_repeats}...", end=" ")
        data = acquire_data(device_name, pmt_channel, sample_rate, segment_samples, segment_time + 10)
        all_data[power].append(data)
        p95 = np.percentile(data, 95)
        print(f"P95 = {p95:.4f} V")

print("\n" + "=" * 70)
print("Data collection complete!")
print("=" * 70)

# ============================================
# 保存原始数据到 txt 文件
# ============================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
raw_data_file = f"raw_data_{timestamp}.txt"

with open(raw_data_file, 'w', encoding='utf-8') as f:
    f.write("# P95 Stability Test - Raw Data\n")
    f.write(f"# Timestamp: {timestamp}\n")
    f.write(f"# Sample Rate: {sample_rate} Hz\n")
    f.write(f"# Segment Time: {segment_time} s\n")
    f.write(f"# Segment Samples: {segment_samples}\n")
    f.write(f"# Power Levels: {power_list}\n")
    f.write(f"# Repeats per Power: {n_repeats}\n")
    f.write(f"# Test Windows (ms): {window_sizes_ms}\n")
    f.write("#" + "=" * 69 + "\n\n")

    for power in power_list:
        f.write(f"# Power Level: {power}\n")
        f.write("#" + "-" * 50 + "\n")
        for rep_idx, data in enumerate(all_data[power]):
            f.write(f"# Repeat {rep_idx + 1}\n")
            for val in data:
                f.write(f"{val:.6f}\n")
            f.write("\n")
        f.write("\n")

print(f"\n>>> Raw data saved to: {raw_data_file}")

# ============================================
# 计算所有窗口的 P95 统计
# ============================================
repeatability_results = {}

for window_ms in window_sizes_ms:
    repeatability_results[window_ms] = {}
    for power in power_list:
        p95_values = []
        for data in all_data[power]:
            feat = extract_features_from_segment(data, window_ms, sample_rate)
            p95_values.append(feat['p95'])

        p95_values = np.array(p95_values)
        mean_p95 = np.mean(p95_values)
        std_p95 = np.std(p95_values)
        cv = (std_p95 / abs(mean_p95) * 100) if abs(mean_p95) > 1e-9 else 0

        repeatability_results[window_ms][power] = {
            'values': p95_values,
            'mean': mean_p95,
            'std': std_p95,
            'cv': cv
        }

# ============================================
# 打印所有窗口的 P95 稳定性结果
# ============================================
print("\n" + "=" * 70)
print("P95 Stability Analysis - ALL WINDOWS")
print("=" * 70)

for window_ms in window_sizes_ms:
    print(f"\n{'=' * 20} Window = {window_ms} ms {'=' * 20}")
    print(f"{'Power':<10} | {'Mean P95':>10} | {'Std P95':>10} | {'CV(%)':>8} | {'Min':>10} | {'Max':>10}")
    print("-" * 70)

    for power in power_list:
        r = repeatability_results[window_ms][power]
        print(f"{power:<10} | {r['mean']:>10.4f} | {r['std']:>10.4f} | {r['cv']:>8.2f} | "
              f"{np.min(r['values']):>10.4f} | {np.max(r['values']):>10.4f}")

# ============================================
# 打印 CV 对比总结表
# ============================================
print("\n" + "=" * 70)
print("CV (Stability) Comparison Table - ALL WINDOWS")
print("=" * 70)

# 表头
header = f"{'Power':<10} |"
for w in window_sizes_ms:
    header += f" {w}ms CV% |"
print(header)
print("-" * (12 + 10 * len(window_sizes_ms)))

# 每行一个功率
for power in power_list:
    row = f"{power:<10} |"
    for window_ms in window_sizes_ms:
        cv = repeatability_results[window_ms][power]['cv']
        row += f" {cv:>7.2f} |"
    print(row)

# 平均CV
print("-" * (12 + 10 * len(window_sizes_ms)))
avg_row = f"{'Avg':<10} |"
for window_ms in window_sizes_ms:
    avg_cv = np.mean([repeatability_results[window_ms][p]['cv'] for p in power_list])
    avg_row += f" {avg_cv:>7.2f} |"
print(avg_row)

# ============================================
# 区分度分析
# ============================================
print("\n" + "=" * 70)
print("Discrimination Analysis (SNR between adjacent powers)")
print("=" * 70)


def compute_discrimination(power_list, rep_data):
    results = []
    for i in range(len(power_list) - 1):
        p1, p2 = power_list[i], power_list[i + 1]
        mean1, mean2 = rep_data[p1]['mean'], rep_data[p2]['mean']
        std1, std2 = rep_data[p1]['std'], rep_data[p2]['std']

        diff = mean2 - mean1
        combined_std = np.sqrt(std1 ** 2 + std2 ** 2)
        snr = abs(diff) / combined_std if combined_std > 1e-9 else float('inf')

        results.append({
            'pair': f"{p1} vs {p2}",
            'diff': diff,
            'snr': snr,
            'distinguishable': snr > 3
        })
    return results


for window_ms in window_sizes_ms:
    print(f"\n--- Window = {window_ms} ms ---")
    disc_results = compute_discrimination(power_list, repeatability_results[window_ms])

    for r in disc_results:
        status = "✓ YES" if r['distinguishable'] else "✗ NO"
        print(f"  {r['pair']:<15}: Diff={r['diff']:>8.4f}V, SNR={r['snr']:>6.2f}, Distinguish: {status}")

# ============================================
# 绘图
# ============================================

# 图1: 各功率的P95重复测量值（箱线图）- 4个窗口
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
axes1 = axes1.flatten()

for idx, window_ms in enumerate(window_sizes_ms):
    ax = axes1[idx]

    box_data = []
    for power in power_list:
        box_data.append(repeatability_results[window_ms][power]['values'].tolist())

    # 使用 tick_labels 替代已弃用的 labels
    bp = ax.boxplot(box_data, tick_labels=power_list, patch_artist=True)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(power_list)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel('Power Level')
    ax.set_ylabel('P95 (V)')
    ax.set_title(f'P95 Repeatability @ {window_ms}ms window')
    ax.grid(True, alpha=0.3, axis='y')

    # 添加散点
    for i, data_list in enumerate(box_data):
        x = np.random.normal(i + 1, 0.04, size=len(data_list))
        ax.scatter(x, data_list, alpha=0.6, s=30, color='red', zorder=3)

plt.tight_layout()
plot1_file = f"p95_boxplot_4windows_{timestamp}.png"
plt.savefig(plot1_file, dpi=150)
print(f"\n>>> Saved: {plot1_file}")

# 图2: CV vs 窗口大小
fig2, ax2 = plt.subplots(figsize=(10, 6))

for power in power_list:
    cvs = [repeatability_results[w][power]['cv'] for w in window_sizes_ms]
    ax2.plot(window_sizes_ms, cvs, 'o-', markersize=10, linewidth=2, label=power)

ax2.axhline(5, color='green', linestyle='--', alpha=0.7, label='CV=5% (Excellent)')
ax2.axhline(10, color='orange', linestyle='--', alpha=0.7, label='CV=10% (Good)')
ax2.set_xlabel('Window Size (ms)', fontsize=12)
ax2.set_ylabel('Coefficient of Variation (%)', fontsize=12)
ax2.set_title('P95 Repeatability (CV) vs Window Size', fontsize=14)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(window_sizes_ms)

plt.tight_layout()
plot2_file = f"p95_cv_vs_window_{timestamp}.png"
plt.savefig(plot2_file, dpi=150)
print(f">>> Saved: {plot2_file}")

# 图3: P95均值柱状图
fig3, ax3 = plt.subplots(figsize=(12, 6))

x = np.arange(len(power_list))
width = 0.2
colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for idx, window_ms in enumerate(window_sizes_ms):
    means = [repeatability_results[window_ms][p]['mean'] for p in power_list]
    stds = [repeatability_results[window_ms][p]['std'] for p in power_list]
    ax3.bar(x + idx * width, means, width, yerr=stds, capsize=4,
            label=f'{window_ms}ms', alpha=0.8, color=colors_bar[idx])

ax3.set_xlabel('Power Level', fontsize=12)
ax3.set_ylabel('P95 (V)', fontsize=12)
ax3.set_title(f'P95 by Power Level and Window Size (n={n_repeats} repeats)', fontsize=14)
ax3.set_xticks(x + width * 1.5)
ax3.set_xticklabels(power_list)
ax3.legend(title='Window')
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot3_file = f"p95_by_power_window_{timestamp}.png"
plt.savefig(plot3_file, dpi=150)
print(f">>> Saved: {plot3_file}")

# 图4: SNR vs 窗口大小
fig4, ax4 = plt.subplots(figsize=(10, 6))

snr_by_window = []
for window_ms in window_sizes_ms:
    disc_results = compute_discrimination(power_list, repeatability_results[window_ms])
    valid_snrs = [r['snr'] for r in disc_results if r['snr'] != float('inf')]
    avg_snr = np.mean(valid_snrs) if valid_snrs else 0
    snr_by_window.append(avg_snr)

ax4.bar(range(len(window_sizes_ms)), snr_by_window, color=colors_bar, alpha=0.8, edgecolor='black')
ax4.axhline(3, color='orange', linestyle='--', linewidth=2, label='SNR=3 (Threshold)')
ax4.axhline(5, color='green', linestyle='--', linewidth=2, label='SNR=5 (Good)')

ax4.set_xlabel('Window Size (ms)', fontsize=12)
ax4.set_ylabel('Average SNR', fontsize=12)
ax4.set_title('Average SNR vs Window Size', fontsize=14)
ax4.set_xticks(range(len(window_sizes_ms)))
ax4.set_xticklabels([f'{w}ms' for w in window_sizes_ms])
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

for i, snr in enumerate(snr_by_window):
    ax4.text(i, snr + 0.2, f'{snr:.2f}', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plot4_file = f"snr_vs_window_{timestamp}.png"
plt.savefig(plot4_file, dpi=150)
print(f">>> Saved: {plot4_file}")

# 图5: CV热力图
fig5, ax5 = plt.subplots(figsize=(10, 6))

cv_matrix = np.array([[repeatability_results[w][p]['cv'] for w in window_sizes_ms] for p in power_list])

im = ax5.imshow(cv_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=20)
ax5.set_xticks(range(len(window_sizes_ms)))
ax5.set_xticklabels([f'{w}ms' for w in window_sizes_ms])
ax5.set_yticks(range(len(power_list)))
ax5.set_yticklabels(power_list)
ax5.set_xlabel('Window Size', fontsize=12)
ax5.set_ylabel('Power Level', fontsize=12)
ax5.set_title('P95 CV Heatmap (lower = more stable)', fontsize=14)

for i in range(len(power_list)):
    for j in range(len(window_sizes_ms)):
        text_color = 'white' if cv_matrix[i, j] > 10 else 'black'
        ax5.text(j, i, f'{cv_matrix[i, j]:.1f}%', ha='center', va='center',
                 color=text_color, fontsize=11, fontweight='bold')

cbar = plt.colorbar(im, ax=ax5)
cbar.set_label('CV (%)')

plt.tight_layout()
plot5_file = f"cv_heatmap_{timestamp}.png"
plt.savefig(plot5_file, dpi=150)
print(f">>> Saved: {plot5_file}")

# ============================================
# 保存分析结果到 txt
# ============================================
results_file = f"analysis_results_{timestamp}.txt"

with open(results_file, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("P95 Stability Analysis Results\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Test Configuration:\n")
    f.write(f"  - Power levels: {power_list}\n")
    f.write(f"  - Repeats per power: {n_repeats}\n")
    f.write(f"  - Measurement duration: {segment_time}s\n")
    f.write(f"  - Sample rate: {sample_rate} Hz\n")
    f.write(f"  - Test windows: {window_sizes_ms} ms\n\n")

    f.write("=" * 70 + "\n")
    f.write("P95 Statistics by Window Size\n")
    f.write("=" * 70 + "\n\n")

    for window_ms in window_sizes_ms:
        f.write(f"--- Window = {window_ms} ms ---\n")
        f.write(f"{'Power':<10} | {'Mean P95':>10} | {'Std P95':>10} | {'CV(%)':>8}\n")
        f.write("-" * 50 + "\n")

        for power in power_list:
            r = repeatability_results[window_ms][power]
            f.write(f"{power:<10} | {r['mean']:>10.4f} | {r['std']:>10.4f} | {r['cv']:>8.2f}\n")
        f.write("\n")

    f.write("=" * 70 + "\n")
    f.write("CV Comparison Summary\n")
    f.write("=" * 70 + "\n\n")

    header = f"{'Power':<10} |"
    for w in window_sizes_ms:
        header += f" {w}ms CV% |"
    f.write(header + "\n")
    f.write("-" * (12 + 10 * len(window_sizes_ms)) + "\n")

    for power in power_list:
        row = f"{power:<10} |"
        for window_ms in window_sizes_ms:
            cv = repeatability_results[window_ms][power]['cv']
            row += f" {cv:>7.2f} |"
        f.write(row + "\n")

print(f">>> Saved: {results_file}")

# ============================================
# 最终总结
# ============================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"\nGenerated files:")
print(f"  1. {raw_data_file} (raw voltage data)")
print(f"  2. {results_file} (analysis results)")
print(f"  3. {plot1_file} (boxplot)")
print(f"  4. {plot2_file} (CV vs window)")
print(f"  5. {plot3_file} (P95 bar chart)")
print(f"  6. {plot4_file} (SNR vs window)")
print(f"  7. {plot5_file} (CV heatmap)")

# 保存npz
npz_file = f"p95_stability_{timestamp}.npz"
np.savez(npz_file,
         power_list=power_list,
         n_repeats=n_repeats,
         window_sizes_ms=window_sizes_ms,
         sample_rate=sample_rate,
         segment_time=segment_time)
print(f"  8. {npz_file} (numpy data)")

plt.show()