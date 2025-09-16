"""
DASH Implementation with Fixed Settling Time
Real-time enhancement monitoring and data logging
"""

import numpy as np
from ctypes import *
import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType, LineGrouping
from scipy import signal
import matplotlib.pyplot as plt
import time
from datetime import datetime


class DASH_System:
    def __init__(self, n_modes=225, settle_time_ms=40):
        # DAQ configuration
        self.device_name = "Dev2"
        self.trigger_line = "port0/line0"
        self.pmt_channel = "ai0"

        # Timing parameters
        self.sample_rate = 10000  # 10kHz
        self.settle_time_ms = settle_time_ms  # Liquid crystal settling time
        self.integration_time_ms = 5  # Integration time after settling
        self.total_samples = int((settle_time_ms + self.integration_time_ms) * self.sample_rate / 1000)
        self.settle_samples = int(settle_time_ms * self.sample_rate / 1000)

        # SLM parameters
        self.slm_lib = None
        self.board_number = c_uint(1)
        self.image_size = 1024 * 1024
        self.n_modes = n_modes
        self.modes_per_side = int(np.sqrt(n_modes))

        # Phase stepping
        self.n_phase_steps = 5
        self.phase_values = np.linspace(0, 2 * np.pi, self.n_phase_steps, endpoint=False)

        # DASH parameters
        self.f = 0.3  # Power fraction
        self.correction = np.zeros(self.n_modes, dtype=complex)

        # 50Hz notch filter
        b, a = signal.iirnotch(50.0, 30.0, self.sample_rate)
        self.notch_b = b
        self.notch_a = a

        # Data storage
        self.enhancement_history = []
        self.signal_history = []
        self.baseline_signal = None

    def init_slm(self):
        """Initialize SLM"""
        cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\SDK\\Blink_C_wrapper")
        self.slm_lib = CDLL("Blink_C_wrapper")

        num_boards_found = c_uint(0)
        self.slm_lib.Create_SDK(c_uint(12), byref(num_boards_found), byref(c_uint(-1)),
                                c_bool(1), c_bool(1), c_bool(1), c_uint(20), 0)

        if num_boards_found.value < 1:
            raise Exception("SLM not found")

        self.slm_lib.Load_LUT_file(self.board_number,
                                   b"C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\1024x1024_linearVoltage.LUT")

        print(f"SLM initialized successfully")
        return True

    def generate_mode_pattern(self, mode_idx, phase_step):
        """Generate DASH phase pattern"""
        # Create mode grating
        nx = mode_idx % self.modes_per_side
        ny = mode_idx // self.modes_per_side

        x = np.linspace(-np.pi, np.pi, 1024)
        y = np.linspace(-np.pi, np.pi, 1024)
        X, Y = np.meshgrid(x, y)

        kx = (nx - self.modes_per_side // 2) * 2 * np.pi / self.modes_per_side
        ky = (ny - self.modes_per_side // 2) * 2 * np.pi / self.modes_per_side
        mode_phase = kx * X / np.pi + ky * Y / np.pi

        # DASH combination
        mode_field = np.sqrt(self.f) * np.exp(1j * (mode_phase + self.phase_values[phase_step]))

        if np.any(self.correction != 0):
            # Add correction field
            correction_phase = self.reconstruct_correction_phase()
            correction_field = np.sqrt(1 - self.f) * np.exp(1j * correction_phase)
            combined = mode_field + correction_field
        else:
            combined = mode_field

        # Convert to phase-only (0-255)
        phase_pattern = np.angle(combined) + np.pi
        gray_levels = (phase_pattern * 255 / (2 * np.pi)).astype(np.uint8)

        return gray_levels.flatten()

    def reconstruct_correction_phase(self):
        """Reconstruct 2D correction pattern from modes"""
        pattern = np.zeros((1024, 1024), dtype=complex)
        for idx, coeff in enumerate(self.correction):
            if coeff != 0:
                nx = idx % self.modes_per_side
                ny = idx // self.modes_per_side
                x = np.linspace(-np.pi, np.pi, 1024)
                y = np.linspace(-np.pi, np.pi, 1024)
                X, Y = np.meshgrid(x, y)
                kx = (nx - self.modes_per_side // 2) * 2 * np.pi / self.modes_per_side
                ky = (ny - self.modes_per_side // 2) * 2 * np.pi / self.modes_per_side
                pattern += coeff * np.exp(1j * (kx * X / np.pi + ky * Y / np.pi))
        return np.angle(pattern)

    def load_and_trigger(self, pattern, trigger_task):
        """Load pattern and send trigger"""
        # Load to SLM
        ret = self.slm_lib.Load_sequence(
            self.board_number,
            pattern.ctypes.data_as(POINTER(c_ubyte)),
            c_uint(self.image_size),
            c_int(1),
            c_uint(1), c_uint(0), c_uint(1), c_uint(0),
            c_uint(1000)
        )

        if ret == -1:
            return False

        # Send trigger
        trigger_task.write(True)
        time.sleep(0.001)
        trigger_task.write(False)  # Falling edge
        time.sleep(0.001)
        trigger_task.write(True)

        return True

    def measure_mode(self, mode_idx, pmt_task, trigger_task):
        """Measure single mode with phase stepping"""
        intensities = np.zeros(self.n_phase_steps)

        for p in range(self.n_phase_steps):
            # Generate pattern
            pattern = self.generate_mode_pattern(mode_idx, p)

            # Load and trigger
            if not self.load_and_trigger(pattern, trigger_task):
                print(f"Failed to load mode {mode_idx}, phase {p}")
                continue

            # Acquire signal (includes settling time)
            raw_data = pmt_task.read(number_of_samples_per_channel=self.total_samples)

            # Use only stable portion
            stable_data = raw_data[self.settle_samples:]

            # Filter and integrate
            filtered = signal.filtfilt(self.notch_b, self.notch_a, stable_data)
            intensities[p] = np.mean(filtered)

        # Extract amplitude and phase
        real_part = np.sum(intensities * np.cos(self.phase_values))
        imag_part = np.sum(intensities * np.sin(self.phase_values))

        amplitude = 2 * np.sqrt(real_part ** 2 + imag_part ** 2) / self.n_phase_steps
        phase = np.arctan2(imag_part, real_part)
        mean_intensity = np.mean(intensities)

        return amplitude, phase, mean_intensity

    def update_correction(self, mode_idx, amplitude, phase):
        """DASH continuous update"""
        # Update correction coefficient for this mode
        self.correction[mode_idx] = amplitude * np.exp(1j * phase)

    def calculate_enhancement(self, current_signal):
        """Calculate enhancement relative to baseline"""
        if self.baseline_signal is None or self.baseline_signal == 0:
            return 1.0
        return current_signal / self.baseline_signal

    def run_dash(self):
        """Main DASH loop with user input"""
        # Get user parameters
        print("\n" + "=" * 60)
        print("DASH OPTIMIZATION SETUP")
        print("=" * 60)

        iterations = int(input(f"Enter number of iterations (default 3): ") or "3")
        n_modes = int(input(f"Enter number of modes (default {self.n_modes}): ") or str(self.n_modes))
        self.n_modes = n_modes
        self.modes_per_side = int(np.sqrt(n_modes))

        print(f"\nConfiguration:")
        print(f"  Modes: {self.n_modes} ({self.modes_per_side}x{self.modes_per_side})")
        print(f"  Iterations: {iterations}")
        print(f"  Settling time: {self.settle_time_ms}ms")
        print(f"  Integration time: {self.integration_time_ms}ms")
        print(f"  Total time per mode: ~{(self.settle_time_ms + self.integration_time_ms) * self.n_phase_steps}ms")

        input("\nPress Enter to start optimization...")

        # Initialize hardware
        if not self.init_slm():
            return None

        # Setup DAQ
        with nidaqmx.Task() as pmt_task, nidaqmx.Task() as trigger_task:
            # Configure PMT input
            pmt_task.ai_channels.add_ai_voltage_chan(
                f"{self.device_name}/{self.pmt_channel}",
                terminal_config=TerminalConfiguration.PSEUDO_DIFF,
                min_val=-10.0, max_val=10.0
            )
            pmt_task.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=self.total_samples
            )

            # Configure trigger
            trigger_task.do_channels.add_do_chan(
                f"{self.device_name}/{self.trigger_line}",
                line_grouping=LineGrouping.CHAN_PER_LINE
            )

            # Main optimization
            start_time = datetime.now()

            for iteration in range(iterations):
                print(f"\n{'=' * 40}")
                print(f"ITERATION {iteration + 1}/{iterations}")
                print(f"{'=' * 40}")

                for mode in range(self.n_modes):
                    # Measure mode
                    amplitude, phase, intensity = self.measure_mode(mode, pmt_task, trigger_task)

                    # Store baseline
                    if iteration == 0 and mode == 0:
                        self.baseline_signal = intensity

                    # Calculate enhancement
                    enhancement = self.calculate_enhancement(intensity)

                    # Update correction
                    self.update_correction(mode, amplitude, phase)

                    # Store data
                    self.signal_history.append(intensity)
                    self.enhancement_history.append(enhancement)

                    # Display progress
                    if mode % 10 == 0 or mode == self.n_modes - 1:
                        elapsed = (datetime.now() - start_time).total_seconds()
                        print(f"  Mode {mode + 1:3d}/{self.n_modes}: "
                              f"Signal={intensity:6.3f}V, "
                              f"Enhancement={enhancement:5.2f}x, "
                              f"Time={elapsed:5.1f}s")

                # Iteration summary
                iter_signals = self.signal_history[-self.n_modes:]
                print(f"\nIteration {iteration + 1} Summary:")
                print(f"  Mean signal: {np.mean(iter_signals):.3f}V")
                print(f"  Max signal: {np.max(iter_signals):.3f}V")
                print(f"  Mean enhancement: {np.mean(self.enhancement_history[-self.n_modes:]):.2f}x")

        # Cleanup
        self.slm_lib.Delete_SDK()

        # Save and plot results
        self.save_results()
        self.plot_results()

        return self.signal_history, self.enhancement_history

    def save_results(self):
        """Save results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dash_results_{timestamp}.txt"

        with open(filename, 'w') as f:
            f.write("# DASH Optimization Results\n")
            f.write(f"# Timestamp: {timestamp}\n")
            f.write(f"# Modes: {self.n_modes}\n")
            f.write(f"# Settling time: {self.settle_time_ms}ms\n")
            f.write("# Measurement\tSignal(V)\tEnhancement\n")

            for i, (sig, enh) in enumerate(zip(self.signal_history, self.enhancement_history)):
                f.write(f"{i}\t{sig:.6f}\t{enh:.4f}\n")

        print(f"\nResults saved to {filename}")

    def plot_results(self):
        """Plot optimization progress"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Signal plot
        ax1.plot(self.signal_history, 'b-', linewidth=0.8)
        ax1.set_xlabel('Measurement Number')
        ax1.set_ylabel('PMT Signal (V)')
        ax1.set_title('DASH Optimization: Signal')
        ax1.grid(True, alpha=0.3)

        # Enhancement plot
        ax2.plot(self.enhancement_history, 'r-', linewidth=0.8)
        ax2.set_xlabel('Measurement Number')
        ax2.set_ylabel('Enhancement Factor')
        ax2.set_title('DASH Optimization: Enhancement')
        ax2.grid(True, alpha=0.3)

        # Mark iteration boundaries
        for i in range(len(self.signal_history) // self.n_modes):
            ax1.axvline(i * self.n_modes, color='k', linestyle='--', alpha=0.3)
            ax2.axvline(i * self.n_modes, color='k', linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.show()


def main():
    """Run DASH optimization"""
    dash = DASH_System()
    results = dash.run_dash()

    if results:
        signals, enhancements = results
        print(f"\nFinal Results:")
        print(f"  Peak enhancement: {np.max(enhancements):.2f}x")
        print(f"  Final enhancement: {enhancements[-1]:.2f}x")


if __name__ == "__main__":
    main()