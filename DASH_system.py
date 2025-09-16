"""
DASH Implementation - SLM Control and PMT Signal Processing
Hardware: NI PCI-6110 DAQ, Meadowlark SLM, PMT
"""

import numpy as np
from ctypes import *
import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType, LineGrouping
from scipy import signal
import matplotlib.pyplot as plt
import time


class DASH_System:
    def __init__(self):
        # DAQ configuration
        self.device_name = "Dev2"
        self.trigger_line = "port0/line0"  # P0.0 -> SLM Trigger
        self.pmt_channel = "ai0"  # AI0 <- PMT signal

        # Sampling parameters
        self.sample_rate = 10000  # 10kHz sampling
        self.integration_samples = 50  # 5ms = 50 samples at 10kHz

        # SLM parameters
        self.slm_lib = None
        self.board_number = c_uint(1)
        self.image_size = 1024 * 1024
        self.n_modes = 225  # 15x15 modes

        # Phase stepping parameters
        self.n_phase_steps = 5
        self.phase_values = np.linspace(0, 2 * np.pi, self.n_phase_steps, endpoint=False)

        # DASH parameters
        self.f = 0.3  # Power fraction for test mode
        self.correction_phase = np.zeros(self.n_modes)

        # 50Hz notch filter coefficients
        b, a = signal.iirnotch(50.0, 30.0, self.sample_rate)
        self.notch_b = b
        self.notch_a = a

    def init_slm(self):
        """Initialize SLM hardware"""
        cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\SDK\\Blink_C_wrapper")
        self.slm_lib = CDLL("Blink_C_wrapper")

        num_boards_found = c_uint(0)
        self.slm_lib.Create_SDK(c_uint(12), byref(num_boards_found), byref(c_uint(-1)),
                                c_bool(1), c_bool(1), c_bool(1), c_uint(20), 0)

        if num_boards_found.value < 1:
            raise Exception("SLM not found")

        self.slm_lib.Load_LUT_file(self.board_number,
                                   b"C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\1024x1024_linearVoltage.LUT")

        print(f"SLM initialized")
        return True

    def generate_mode_pattern(self, mode_idx, phase_step):
        """Generate phase pattern for given mode and phase step"""
        # Simple grating pattern for mode m
        nx = mode_idx % 15
        ny = mode_idx // 15

        x = np.linspace(-np.pi, np.pi, 1024)
        y = np.linspace(-np.pi, np.pi, 1024)
        X, Y = np.meshgrid(x, y)

        # Mode phase pattern
        mode_phase = (nx - 7) * X / 7 + (ny - 7) * Y / 7

        # Combine with correction using DASH formula
        if mode_idx == 0:
            # First mode, no correction yet
            combined = np.sqrt(self.f) * np.exp(1j * (mode_phase + self.phase_values[phase_step]))
        else:
            # Add correction phase
            correction = self.get_correction_pattern()
            combined = (np.sqrt(self.f) * np.exp(1j * (mode_phase + self.phase_values[phase_step])) +
                        np.sqrt(1 - self.f) * np.exp(1j * correction))

        # Convert to phase-only pattern (0-255)
        phase_pattern = np.angle(combined) + np.pi
        gray_levels = (phase_pattern * 255 / (2 * np.pi)).astype(np.uint8)

        return gray_levels.flatten()

    def load_pattern_to_slm(self, pattern):
        """Load single pattern to SLM"""
        ret = self.slm_lib.Load_sequence(
            self.board_number,
            pattern.ctypes.data_as(POINTER(c_ubyte)),
            c_uint(self.image_size),
            c_int(1),  # Single pattern
            c_uint(1),  # Wait for trigger
            c_uint(0),  # No immediate flip
            c_uint(1),  # Output pulse on flip
            c_uint(0),  # No refresh pulse
            c_uint(1000)  # 1s timeout
        )
        return ret != -1

    def trigger_slm(self, trigger_task):
        """Send trigger pulse to SLM"""
        trigger_task.write(True)
        time.sleep(0.001)
        trigger_task.write(False)  # Falling edge triggers
        time.sleep(0.001)
        trigger_task.write(True)

    def acquire_pmt_signal(self, pmt_task):
        """Acquire PMT signal for one integration period"""
        # Read samples
        data = pmt_task.read(number_of_samples_per_channel=self.integration_samples)

        # Apply 50Hz notch filter
        filtered = signal.filtfilt(self.notch_b, self.notch_a, data)

        # Return mean value (integration)
        return np.mean(filtered)

    def measure_mode(self, mode_idx, pmt_task, trigger_task):
        """Measure single mode with phase stepping"""
        intensities = np.zeros(self.n_phase_steps)

        for p in range(self.n_phase_steps):
            # Generate and load pattern
            pattern = self.generate_mode_pattern(mode_idx, p)
            if not self.load_pattern_to_slm(pattern):
                print(f"Failed to load pattern for mode {mode_idx}, phase {p}")
                continue

            # Trigger SLM
            self.trigger_slm(trigger_task)

            # Wait for pattern to stabilize
            time.sleep(0.01)

            # Acquire signal
            intensities[p] = self.acquire_pmt_signal(pmt_task)

        # Calculate amplitude and phase from phase stepping
        # Using formula from DASH paper
        real_part = np.sum(intensities * np.cos(self.phase_values))
        imag_part = np.sum(intensities * np.sin(self.phase_values))

        amplitude = 2 * np.sqrt(real_part ** 2 + imag_part ** 2) / self.n_phase_steps
        phase = np.arctan2(imag_part, real_part)

        return amplitude, phase, np.mean(intensities)

    def update_correction(self, mode_idx, amplitude, phase):
        """Update correction phase (DASH algorithm)"""
        # Add weighted mode contribution to correction
        self.correction_phase[mode_idx] = phase

    def get_correction_pattern(self):
        """Generate full correction pattern from mode phases"""
        # Simplified - in reality would reconstruct full 2D pattern
        # Here returning average phase for demonstration
        return np.mean(self.correction_phase[:max(1, np.count_nonzero(self.correction_phase))])

    def run_dash(self, iterations=3):
        """Main DASH optimization loop"""
        print("Starting DASH optimization...")

        # Initialize hardware
        if not self.init_slm():
            return

        # Storage for results
        signals = np.zeros((iterations, self.n_modes))

        # Setup DAQ tasks
        with nidaqmx.Task() as pmt_task, nidaqmx.Task() as trigger_task:

            # Configure PMT input
            pmt_task.ai_channels.add_ai_voltage_chan(
                f"{self.device_name}/{self.pmt_channel}",
                terminal_config=TerminalConfiguration.PSEUDO_DIFF,
                min_val=-10.0,
                max_val=10.0
            )
            pmt_task.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=AcquisitionType.FINITE,
                samps_per_chan=self.integration_samples
            )

            # Configure trigger output
            trigger_task.do_channels.add_do_chan(
                f"{self.device_name}/{self.trigger_line}",
                line_grouping=LineGrouping.CHAN_PER_LINE
            )

            # Main optimization loop
            for iteration in range(iterations):
                print(f"\nIteration {iteration + 1}/{iterations}")

                for mode in range(self.n_modes):
                    # Measure mode response
                    amplitude, phase, mean_signal = self.measure_mode(mode, pmt_task, trigger_task)

                    # Store signal
                    signals[iteration, mode] = mean_signal

                    # Update correction (DASH continuous update)
                    self.update_correction(mode, amplitude, phase)

                    # Progress indicator
                    if mode % 15 == 0:
                        print(f"  Mode {mode}/{self.n_modes}, Signal: {mean_signal:.3f}")

        # Cleanup
        self.slm_lib.Delete_SDK()

        return signals

    def plot_results(self, signals):
        """Plot optimization progress"""
        plt.figure(figsize=(10, 6))

        # Flatten signals for plotting
        flat_signals = signals.flatten()
        measurements = np.arange(len(flat_signals))

        plt.plot(measurements, flat_signals, 'b-', linewidth=0.8)
        plt.xlabel('Measurement Number')
        plt.ylabel('PMT Signal (V)')
        plt.title('DASH Optimization Progress')
        plt.grid(True, alpha=0.3)

        # Mark iteration boundaries
        for i in range(signals.shape[0]):
            plt.axvline(i * self.n_modes, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()


def main():
    """Run DASH optimization"""
    dash = DASH_System()

    # Run optimization
    signals = dash.run_dash(iterations=3)

    # Plot results
    dash.plot_results(signals)

    # Print statistics
    print(f"\nOptimization complete")
    print(f"Initial signal: {np.mean(signals[0, :5]):.3f} V")
    print(f"Final signal: {np.mean(signals[-1, -5:]):.3f} V")
    print(f"Enhancement: {np.mean(signals[-1, -5:]) / np.mean(signals[0, :5]):.1f}x")


if __name__ == "__main__":
    main()