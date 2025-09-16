"""
SLM Response Time Test - Using BMP Files
Record PMT signal and trigger events with actual Meadowlark images
"""

import numpy as np
from ctypes import *
from PIL import Image as PILImage
import nidaqmx
from nidaqmx.constants import LineGrouping, TerminalConfiguration, AcquisitionType
import time
import threading
from datetime import datetime
import matplotlib.pyplot as plt
import os


class SLM_PMT_Recorder:
    def __init__(self, num_patterns=2):
        # DAQ configuration
        self.device_name = "Dev2"
        self.daq_trigger_output = "port0/line0"  # DAQ -> SLM Trigger In
        self.daq_feedback_input = "port0/line1"  # DAQ <- SLM Trigger Out
        self.pmt_channel = "ai0"

        # SLM parameters
        self.slm_lib = None
        self.board_number = c_uint(1)
        self.width = 1024
        self.height = 1024
        self.ImgSize = 1024 * 1024
        self.num_patterns = num_patterns

        # Recording parameters
        self.sample_rate = 10000  # Hz
        self.sample_period = 1.0 / self.sample_rate  # seconds

        # Data storage
        self.pmt_data = []
        self.trigger_events = []  # (sample_index, event_type)
        self.current_sample = 0
        self.recording = False
        self.recording_lock = threading.Lock()

        # Image paths
        self.image_paths = [
            "C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\Image Files\\1024\\Central.bmp",
            "C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\Image Files\\1024\\mlo.bmp"
        ]
        if num_patterns >= 3:
            self.image_paths.append(
                "C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\Image Files\\1024\\Astig3.bmp"
            )

    def init_slm(self):
        """Initialize SLM"""
        print("Initializing SLM...")

        cdll.LoadLibrary("C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\SDK\\Blink_C_wrapper")
        self.slm_lib = CDLL("Blink_C_wrapper")

        num_boards_found = c_uint(0)
        self.slm_lib.Create_SDK(c_uint(12), byref(num_boards_found), byref(c_uint(-1)),
                                c_bool(1), c_bool(1), c_bool(1), c_uint(20), 0)

        if num_boards_found.value < 1:
            print("SLM not found")
            return False

        self.slm_lib.Load_LUT_file(self.board_number,
                                   b"C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\LUT Files\\1024x1024_linearVoltage.LUT")

        print(f"Found {num_boards_found.value} SLM(s)")
        return True

    def load_patterns(self):
        """Load BMP images to hardware"""
        print(f"\nLoading {self.num_patterns} BMP images to hardware...")

        # Create combined image array
        total_size = self.ImgSize * self.num_patterns
        image_array = np.zeros(total_size, dtype=np.uint8)

        # Load each BMP file
        for i, image_path in enumerate(self.image_paths[:self.num_patterns]):
            # Extract filename for display
            filename = os.path.basename(image_path)
            print(f"  Loading: {filename}")

            # Load image (following reference code method)
            img = PILImage.open(image_path)
            img_array = np.array(img.convert('L'), dtype=np.uint8)

            # Place in combined array
            start_idx = i * self.ImgSize
            end_idx = (i + 1) * self.ImgSize
            image_array[start_idx:end_idx] = img_array.flatten()

        # Load sequence to hardware
        wait_for_trigger = c_uint(1)
        flip_immediate = c_uint(0)
        output_pulse_image_flip = c_uint(1)
        output_pulse_image_refresh = c_uint(0)
        trigger_timeout_ms = c_uint(10000)

        ret = self.slm_lib.Load_sequence(
            self.board_number,
            image_array.ctypes.data_as(POINTER(c_ubyte)),
            c_uint(self.ImgSize),
            c_int(self.num_patterns),
            wait_for_trigger,
            flip_immediate,
            output_pulse_image_flip,
            output_pulse_image_refresh,
            trigger_timeout_ms
        )

        if ret == -1:
            print("Failed to load image sequence")
            return False

        print(f"Successfully loaded {self.num_patterns} images to hardware:")
        for i, path in enumerate(self.image_paths[:self.num_patterns]):
            filename = os.path.basename(path)
            print(f"    - Frame {i}: {filename}")
        return True

    def start_recording(self):
        """Start PMT recording thread"""
        self.recording = True
        self.current_sample = 0
        thread = threading.Thread(target=self._recording_thread)
        thread.daemon = True
        thread.start()
        time.sleep(0.5)  # Let recording stabilize
        print("PMT recording started")

    def _recording_thread(self):
        """Continuous PMT recording"""
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(
                f"{self.device_name}/{self.pmt_channel}",
                terminal_config=TerminalConfiguration.PSEUDO_DIFF,
                min_val=-10.0,
                max_val=10.0
            )

            task.timing.cfg_samp_clk_timing(
                rate=self.sample_rate,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=1000
            )

            task.start()

            while self.recording:
                try:
                    data = task.read(number_of_samples_per_channel=100, timeout=0.1)
                    with self.recording_lock:
                        self.pmt_data.extend(data)
                        self.current_sample += len(data)
                except nidaqmx.errors.DaqError:
                    continue

            task.stop()

    def switch_pattern(self, frame_index):
        """Switch to specified frame and record trigger events"""

        with nidaqmx.Task() as trigger_task, nidaqmx.Task() as feedback_task:
            # Configure DAQ lines
            trigger_task.do_channels.add_do_chan(
                f"{self.device_name}/{self.daq_trigger_output}",
                line_grouping=LineGrouping.CHAN_PER_LINE
            )

            feedback_task.di_channels.add_di_chan(
                f"{self.device_name}/{self.daq_feedback_input}",
                line_grouping=LineGrouping.CHAN_PER_LINE
            )

            # Set initial high
            trigger_task.write(True)
            time.sleep(0.01)

            # Start SLM select in thread
            select_done = threading.Event()

            def select_thread():
                wait_for_trigger = c_uint(1)
                flip_immediate = c_uint(0)
                output_pulse_image_flip = c_uint(1)
                output_pulse_image_refresh = c_uint(0)
                flip_timeout_ms = c_uint(5000)

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
            time.sleep(0.01)

            # Record trigger sent sample
            with self.recording_lock:
                trigger_sent_sample = self.current_sample
                self.trigger_events.append((trigger_sent_sample, 'TRIGGER_SENT'))

            # Send trigger (falling edge)
            trigger_task.write(False)

            # Monitor feedback
            feedback_received = False
            start_time = time.perf_counter()

            while time.perf_counter() - start_time < 1.0:
                if feedback_task.read():
                    with self.recording_lock:
                        feedback_sample = self.current_sample
                        self.trigger_events.append((feedback_sample, 'FEEDBACK_RECEIVED'))
                    feedback_received = True
                    break
                time.sleep(0.00001)

            # Restore high
            trigger_task.write(True)
            select_done.wait(timeout=5)

            return feedback_received

    def run_test(self, num_switches=10, wait_between=2.0):
        """Run the test sequence"""
        print("\n" + "=" * 60)
        print(f"SLM-PMT Recording Test ({self.num_patterns} patterns)")
        print("=" * 60)

        # Initialize
        if not self.init_slm():
            return

        if not self.load_patterns():
            return

        print(f"\nTest parameters:")
        print(f"  Patterns: {self.num_patterns}")
        print(f"  Switches: {num_switches}")
        print(f"  Wait time: {wait_between}s")
        print(f"  Sample rate: {self.sample_rate} Hz")

        input("\nAdjust sample and laser, press Enter to start...")

        # Start recording
        test_start = datetime.now()
        self.start_recording()

        # Perform switches
        current_frame = 0

        try:
            for i in range(num_switches):
                # Calculate next frame
                next_frame = (current_frame + 1) % self.num_patterns
                image_name = os.path.basename(self.image_paths[next_frame])
                print(f"\nSwitch {i + 1}/{num_switches}: Frame {current_frame} -> {next_frame} ({image_name})")

                # Switch pattern
                if self.switch_pattern(next_frame):
                    current_frame = next_frame
                    print("  Success")
                else:
                    print("  Failed")

                # Wait
                time.sleep(wait_between)

        except KeyboardInterrupt:
            print("\nTest interrupted")

        # Stop recording
        print("\nStopping recording...")
        self.recording = False
        time.sleep(0.5)

        # Save data and plot
        self.save_data(test_start)
        self.plot_pmt_signal()

        # Cleanup
        self.slm_lib.Delete_SDK()
        print("\nTest completed")

    def save_data(self, test_start):
        """Save PMT data and trigger events"""
        timestamp = test_start.strftime("%Y%m%d_%H%M%S")

        # Save PMT data
        pmt_filename = f"pmt_data_{self.num_patterns}patterns_{timestamp}.txt"
        with open(pmt_filename, 'w') as f:
            # Header with metadata
            f.write(f"# PMT Recording Data\n")
            f.write(f"# Test Start: {test_start}\n")
            f.write(f"# Sample Rate: {self.sample_rate} Hz\n")
            f.write(f"# Sample Period: {self.sample_period * 1000:.3f} ms\n")
            f.write(f"# Total Samples: {len(self.pmt_data)}\n")
            f.write(f"# Total Duration: {len(self.pmt_data) / self.sample_rate:.3f} seconds\n")
            f.write(f"# Number of Patterns: {self.num_patterns}\n")

            # Image names
            image_names = [os.path.basename(p) for p in self.image_paths[:self.num_patterns]]
            f.write(f"# Images: {', '.join(image_names)}\n")
            f.write("#\n")
            f.write("# Sample_Index\tPMT_Voltage\n")

            for i, value in enumerate(self.pmt_data):
                f.write(f"{i}\t{value:.6f}\n")

        print(f"PMT data saved: {pmt_filename}")

        # Save trigger events
        trigger_filename = f"trigger_events_{self.num_patterns}patterns_{timestamp}.txt"
        with open(trigger_filename, 'w') as f:
            f.write(f"# Trigger Events\n")
            f.write(f"# Test Start: {test_start}\n")
            f.write(f"# Sample Rate: {self.sample_rate} Hz\n")
            f.write(f"# Sample Period: {self.sample_period * 1000:.3f} ms\n")
            f.write(f"#\n")
            f.write("# Sample_Index\tEvent_Type\tTime_ms\n")

            for sample_idx, event_type in self.trigger_events:
                time_ms = sample_idx * self.sample_period * 1000
                f.write(f"{sample_idx}\t{event_type}\t{time_ms:.3f}\n")

        print(f"Trigger events saved: {trigger_filename}")

        # Print summary
        print(f"\nRecording Summary:")
        print(f"  Total samples: {len(self.pmt_data)}")
        print(f"  Duration: {len(self.pmt_data) / self.sample_rate:.2f} seconds")
        print(f"  Trigger events: {len(self.trigger_events)}")

    def plot_pmt_signal(self):
        """Plot PMT signal with trigger marks"""
        if len(self.pmt_data) == 0:
            print("No data to plot")
            return

        print("\nPlotting PMT signal...")

        # Create sample indices and time array
        samples = np.arange(len(self.pmt_data))
        times = samples * self.sample_period
        pmt = np.array(self.pmt_data)

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot PMT signal
        ax.plot(times, pmt, 'b-', linewidth=0.5, alpha=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('PMT Signal (V)')
        ax.set_title(f'PMT Signal Recording - {self.num_patterns} Patterns')
        ax.grid(True, alpha=0.3)

        # Mark trigger events
        for sample_idx, event_type in self.trigger_events:
            t = sample_idx * self.sample_period
            if event_type == 'TRIGGER_SENT':
                ax.axvline(t, color='red', linestyle='--', alpha=0.6, linewidth=1)
            elif event_type == 'FEEDBACK_RECEIVED':
                ax.axvline(t, color='green', linestyle='--', alpha=0.6, linewidth=1)

        # Add legend
        ax.axvline(-1000, color='red', linestyle='--', label='Trigger Sent')
        ax.axvline(-1000, color='green', linestyle='--', label='Feedback Received')
        ax.legend(loc='upper right')
        ax.set_xlim(0, times[-1])

        plt.tight_layout()

        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_filename = f"pmt_signal_{self.num_patterns}patterns_{timestamp}.png"
        plt.savefig(fig_filename, dpi=150)
        print(f"Figure saved: {fig_filename}")

        plt.show()


def main():
    """Main function with mode selection"""
    print("SLM-PMT Recording Test")
    print("Select mode:")
    print("  1. Two patterns (Central.bmp, mlo.bmp)")
    print("  2. Three patterns (Central.bmp, mlo.bmp, Astig3.bmp)")

    mode = input("Enter mode (1 or 2): ").strip()

    if mode == '1':
        num_patterns = 2
    elif mode == '2':
        num_patterns = 3
    else:
        print("Invalid mode, using 2 patterns")
        num_patterns = 2

    # Create recorder and run test
    recorder = SLM_PMT_Recorder(num_patterns=num_patterns)
    recorder.run_test(
        num_switches=10,
        wait_between=2.0
    )


if __name__ == "__main__":
    main()