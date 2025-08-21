import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
import pickle
import pyroomacoustics as pra
from fir_room_correction import FIRCorrectionFilter
import os
from scipy.signal import stft
import pandas as pd
import analysis_utils

class VirtualMicTest:
    def __init__(self, room_dims, fs, fir_corrected_stereo, mic_array_positions, ref_positions, src_positions):
        self.room_dims = room_dims
        self.fs = fs
        self.fir_corrected_stereo = fir_corrected_stereo
        self.mic_array_positions = mic_array_positions
        self.ref_positions = ref_positions
        self.src_positions = src_positions
        self.recorded_responses = []
        self.geometry = None
        self.array_radius = None
        self.num_mics = None
        self.mode = None
        self.start_time = 0
        self.end_time = 0
        self.start_freq = 0
        self.end_freq = 0
        self.total_delay = 0

    @staticmethod
    def apply_fir(fir_filter, stereo_audio):
        """Apply FIR filter """
        left = lfilter(fir_filter.h_left, [1.0], stereo_audio[:, 0])
        right = lfilter(fir_filter.h_right, [1.0], stereo_audio[:, 1])
        output = np.stack([left, right], axis=-1)

        return output

    @classmethod
    def from_checkpoint_and_fir(cls, checkpoint_path, fir_filter, top_rank_index=0):
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

        dataset = checkpoint["dataset"]
        ranking_df = checkpoint["ranking_df"]
        fs = checkpoint["fs"]
        room_dims = checkpoint["room_dims"]
        signal_stereo = checkpoint["signal_stereo"][0]  # shape: (2, N)

        if signal_stereo.ndim != 2:
            raise ValueError("Expected stereo input signal with shape (2, N)")

        test_stereo = signal_stereo.T  # shape (N, 2)

        # Apply FIR with normalization
        corrected_signal = cls.apply_fir(fir_filter, test_stereo)

        best_label = ranking_df.iloc[top_rank_index]
        base_label = f"geometry={best_label.geometry},method={best_label.method},radius={best_label.radius}," \
                     f"num_mics={best_label.num_mics},mic_array_order={best_label.mic_array_order}," \
                     f"array=room_fixed,audio_channel_format=stereo"

        mic_array_positions = dataset[f"{base_label},ref=room_pos0"]["mic_array_position"]
        ref_positions = [dataset[f"{base_label},ref=room_pos{i}"]["ref_position"] for i in range(5)]
        src_positions = dataset[f"{base_label},ref=room_pos0"]["src_position"]

        instance = cls(room_dims, fs, corrected_signal, mic_array_positions, ref_positions, src_positions)
        instance.reference_input = test_stereo  # Save original input for plotting
        return instance
    
    @classmethod
    def from_checkpoint_with_fir_mode(
        cls,
        dataset_checkpoint_path,
        mode="multi",
        numtaps=256,
        fir_file_prefix="",
        ref_index=0,
        start_time=0.0,
        end_time=5,
        start_freq=20,
        end_freq=20000
    ):
        """
        Loads or trains FIR filters based on the mode ("single", "multi", or "avg"),
        simulates playback, and returns a populated VirtualMicTest instance.

        Args:
            dataset_checkpoint_path (str): Path to the simulation checkpoint (.pkl).
            mode (str): One of "single", "multi", or "avg".
            numtaps (int): Number of FIR filter taps.
            fir_file_prefix (str): Optional prefix for FIR filter file names.
            ref_index (int): Reference mic index for single-mode simulation.

        Returns:
            VirtualMicTest: Instance with FIR-corrected signal and simulation results.
        """
        if mode not in ["single", "multi", "avg"]:
            raise ValueError("mode must be 'single', 'multi', or 'avg'")
        
        # FIR file suffix and naming
        prefix = fir_file_prefix or (
            "center" if mode == "single" else
            "avg" if mode == "avg" else
            "spatial"
        )
        left_path = f"fir_left_{mode}.npy"
        right_path = f"fir_right_{mode}.npy"

        left_path = "../test_files/bin/" + left_path
        right_path = "../test_files/bin/" + right_path

        # Load or train FIR filters
        if os.path.exists(left_path) and os.path.exists(right_path):
            print(f"📂 FIR filters for mode '{mode}' already exist. Loading from disk...")
            fir = FIRCorrectionFilter()
            fir.load(mode=mode)
        else:
            print(f"🚀 Training FIR filters from checkpoint using mode: '{mode}'")
            if mode == "single":
                fir = FIRCorrectionFilter.train_from_checkpoint_single(
                    dataset_checkpoint_path, top_rank_index=0, numtaps=numtaps
                )
            elif mode == "avg":
                fir = FIRCorrectionFilter.train_from_checkpoint_avg(
                    dataset_checkpoint_path, top_rank_index=0, numtaps=numtaps
                )
            else:  # "multi"
                mode == "multi"
                fir = FIRCorrectionFilter.train_fir_from_checkpoint_multi(
                    dataset_checkpoint_path, top_rank_index=0, numtaps=numtaps
                )

        # Load simulation and FIR-corrected signal into test object
        results = cls.from_checkpoint_and_fir(
            checkpoint_path=dataset_checkpoint_path,
            fir_filter=fir,
            top_rank_index=0
        )

        # Extract virtual mic config from checkpoint
        import pickle
        with open(dataset_checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        best_config = checkpoint["ranking_df"].iloc[0]

        # Populate metadata for logging, plots, and CSV
        results.geometry = best_config.geometry
        results.array_radius = best_config.radius
        results.num_mics = best_config.num_mics
        results.mode = mode

        results.start_time = start_time
        results.end_time = end_time
        results.start_freq = start_freq
        results.end_freq = end_freq

        # Run room simulation
        if mode == "single":
            results.simulate_corrected_response_at_index(ref_index=ref_index)
        else:
            results.simulate_corrected_responses_spatial(plot_diff=True)

        return results



    def simulate_corrected_responses_spatial(self, plot_diff=True):
        self.recorded_responses = []

        for ref_pos in self.ref_positions:
            room = pra.ShoeBox(self.room_dims, fs=self.fs, absorption=0.3, max_order=5)

            room.add_source(self.src_positions[0], signal=self.fir_corrected_stereo[:, 0])
            room.add_source(self.src_positions[1], signal=self.fir_corrected_stereo[:, 1])

            room.add_microphone_array(pra.MicrophoneArray(np.array(ref_pos).reshape(3, 1), self.fs))
            room.compute_rir()
            room.simulate()

            recorded_signal = room.mic_array.signals[0]

            #normalized_signal = self.normalize_by_percentile(recorded_signal, 0.8, 100)

            normalized_signal = recorded_signal
            self.recorded_responses.append(normalized_signal)

        print("✅ Simulated FIR-corrected responses at all reference positions.")

        if plot_diff:
            self.plot_error_vs_input_with_freq_axis(
                start_time=self.start_time,
                end_time=self.end_time,
                start_freq=self.start_freq,
                end_freq=self.end_freq )

        return self.recorded_responses
    

    def simulate_corrected_response_at_index(self, ref_index=0, plot_diff=True):
        """
        Simulates playback of FIR-corrected stereo signal through the room,
        recording the response at a single reference mic.

        Args:
            ref_index (int): Index into self.ref_positions (0 = center).
            plot_diff (bool): Whether to plot the response against input signal.

        Returns:
            list: A single-item list containing the recorded response at the selected mic.
        """
        self.recorded_responses = []

        if ref_index < 0 or ref_index >= len(self.ref_positions):
            raise IndexError(f"ref_index {ref_index} is out of bounds. Valid range: 0-{len(self.ref_positions)-1}")

        ref_pos = np.array(self.ref_positions[ref_index]).reshape(3, 1)

        room = pra.ShoeBox(self.room_dims, fs=self.fs, absorption=0.3, max_order=5)

        room.add_source(self.src_positions[0], signal=self.fir_corrected_stereo[:, 0])
        room.add_source(self.src_positions[1], signal=self.fir_corrected_stereo[:, 1])

        room.add_microphone_array(pra.MicrophoneArray(ref_pos, self.fs))

        room.compute_rir()
        room.simulate()

        recorded_signal = room.mic_array.signals[0]

        # ✅ Normalize to [-normalize_range, +normalize_range]
        #normalized_signal = self.normalize_by_percentile(recorded_signal, 0.8, 100)

        normalized_signal = recorded_signal
        self.recorded_responses.append(normalized_signal)

        print(f"✅ Simulated FIR-corrected response at ref mic index {ref_index}.")

        if plot_diff:
            self.plot_error_vs_input_with_freq_axis(
                start_time=self.start_time,
                end_time=self.end_time,
                start_freq=self.start_freq,
                end_freq=self.end_freq )

        return self.recorded_responses
    
    def delay_test_signal(self, input_signal, src_pos, mic_pos, numtaps):
        """
        Returns the input signal delayed by FIR group delay and acoustic propagation delay.

        Args:
            input_signal (np.ndarray): 1D or 2D array of the test signal.
            src_pos (list): Source 3D position.
            mic_pos (list): Reference mic 3D position.
            numtaps (int): Number of FIR taps used in correction.

        Returns:
            np.ndarray: Delayed version of the input signal.
        """
        c = 343.0  # speed of sound (m/s)
        distance = np.linalg.norm(np.array(src_pos) - np.array(mic_pos))
        delay_sec = distance / c
        acoustic_delay = int(round(delay_sec * self.fs))

        fir_delay = (numtaps - 1) // 2
        total_delay = acoustic_delay + fir_delay

        if input_signal.ndim == 1:
            delayed = np.pad(input_signal, (total_delay, 0), mode='constant')[:len(input_signal)]
        else:
            delayed = np.pad(input_signal, ((total_delay, 0), (0, 0)), mode='constant')[:len(input_signal)]

        self.total_delay = total_delay

        return delayed



    def plot_error_vs_input(self, duration_sec=5.0, numtaps=512):
        """
        Plots the FIR-corrected response at each mic against the appropriately delayed input signal.
        Applies both FIR group delay and acoustic delay compensation.
        Assumes mono test input (uses only left channel of stereo reference input).
        """
        num_samples = int(self.fs * duration_sec)
        t = np.linspace(0, duration_sec, num_samples)

        for i, rec in enumerate(self.recorded_responses):
            # Trim both signals to ensure matching length
            rec_trimmed = rec[:num_samples]

            # Reference mic and source position (assume using Left source)
            ref_pos = self.ref_positions[i]
            src_pos = self.src_positions[0]

            # Delay-compensated version of the mono input (left channel)
            delayed_ref = self.delay_test_signal(
                self.reference_input[:, 0], src_pos, ref_pos, numtaps
            )
            delayed_ref = delayed_ref[:num_samples]

            # Plot comparison
            plt.figure(figsize=(12, 4))
            plt.plot(t, delayed_ref, label="Delayed Input Signal (Mono)", alpha=0.8, linestyle='--')
            plt.plot(t, rec_trimmed, label=f"Recorded Response @ Ref {i}", alpha=0.7)
            plt.title(f"Ref Mic {i} - Mode: {getattr(self, 'mode', 'N/A')} - FIR-Corrected vs Delayed Input")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show(block=False)



    def normalize_by_percentile(self, signal, target_peak=1.0, percentile=90):
        """
        Normalize based on the nth percentile of absolute signal values.
        Avoids being skewed by single-sample spikes.
        """
        p = np.percentile(np.abs(signal), percentile)
        if p == 0:
            return signal
        return signal * (target_peak / p)
    

    def evaluate_metrics_for_all_positions(
        self,
        mode='avg',
        numtaps=512,
        fs=48000,
        n_fft=1024,
        csv_path=None
    ):
        """
        Evaluates quantitative metrics between the FIR-corrected responses
        and the appropriately delayed input signal.
        Assumes mono input: compares recorded signal to delayed Left channel only.
        Exports metrics to CSV.

        Returns:
            pd.DataFrame: One row per mic position with metrics and metadata.
        """
        results = []

        # Metadata for export
        array_geometry = self.geometry_type if hasattr(self, 'geometry_type') else "unknown"
        array_radius = self.array_radius if hasattr(self, 'array_radius') else np.nan
        num_mics = self.num_mics if hasattr(self, 'num_mics') else (
            self.virtual_mic_positions.shape[1] if hasattr(self, 'virtual_mic_positions') else np.nan
        )
        src_center = np.mean(np.array(self.src_positions), axis=0)

        for i, rec in enumerate(self.recorded_responses):
            rec = rec[:len(self.reference_input)]
            ref_pos = self.ref_positions[i]
            srcL = self.src_positions[0]  # Assume mono source

            # Use only Left channel for mono evaluation
            delayed_ref = self.delay_test_signal(self.reference_input[:, 0], srcL, ref_pos, numtaps)
            delayed_ref = delayed_ref[:len(rec)]

            # Match lengths
            min_len = min(len(rec), len(delayed_ref))
            rec = rec[:min_len]
            delayed_ref = delayed_ref[:min_len]

            # Amplitude scale match
            ref_p = np.percentile(np.abs(delayed_ref), 95)
            rec_p = np.percentile(np.abs(rec), 95)
            if ref_p > 0:
                delayed_ref *= (rec_p / ref_p)

            # Time-domain metrics
            mse = np.mean((rec - delayed_ref) ** 2)
            snr = 10 * np.log10(np.mean(delayed_ref ** 2) / (mse + 1e-10))
            correlation = np.correlate(rec, delayed_ref, mode='valid')[0]
            correlation /= (np.linalg.norm(rec) * np.linalg.norm(delayed_ref) + 1e-10)

            # Frequency-domain metrics
            f, _, Zxx_ref = stft(delayed_ref, fs=fs, nperseg=n_fft)
            _, _, Zxx_test = stft(rec, fs=fs, nperseg=n_fft)

            min_frames = min(Zxx_ref.shape[1], Zxx_test.shape[1])
            Zxx_ref = Zxx_ref[:, :min_frames]
            Zxx_test = Zxx_test[:, :min_frames]

            mag_ref = np.abs(Zxx_ref)
            mag_test = np.abs(Zxx_test)
            epsilon = 1e-10
            mag_err_db = 20 * np.log10((np.abs(mag_ref - mag_test) + epsilon) / (mag_ref + epsilon))
            spectral_mag_error = np.mean(np.abs(mag_err_db))

            phase_ref = np.angle(Zxx_ref)
            phase_test = np.angle(Zxx_test)
            phase_err = np.unwrap(phase_ref - phase_test, axis=0)
            spectral_phase_error = np.mean(np.abs(phase_err))

            # Distance for reporting
            mic_dist = np.linalg.norm(np.array(ref_pos) - src_center)

            results.append({
                'Mode': mode,
                'Array Geometry': self.geometry,
                'Array Radius (m)': self.array_radius,
                'Num Mics': self.num_mics,
                'Mic Index': i,
                'Source - Mic Dist (m)': round(mic_dist, 3),
                'SNR (dB)': round(snr, 2),
                'MSE': round(mse, 6),
                'Correlation': round(correlation, 4),
                'Spectral Mag Error (dB)': round(spectral_mag_error, 2),
                'Spectral Phase Error (rad)': round(spectral_phase_error, 2),
            })

        df = pd.DataFrame(results)

        # Save to CSV
        if csv_path is None:
            csv_path = f"../test_files/csv/fir_evaluation_metrics_{mode}.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None
        df.to_csv(csv_path, index=False)
        print(f"✅ Evaluation metrics saved to: {csv_path}")

        return df
    
    def evaluate_metrics_for_all_positions_window(
        self,
        mode='avg',
        numtaps=512,
        fs=48000,
        n_fft=1024,
        csv_path=None
    ):
        """
        Same as evaluate_metrics_for_all_positions, but clips the evaluation window
        to avoid FIR boundary effects.

        Args:
            start_time (float): Start time in seconds for evaluation (default 0.5s).
            end_time (float): End time in seconds for evaluation (default 4.5s).
        """
        results = []

        array_geometry = getattr(self, 'geometry_type', 'unknown')
        array_radius = getattr(self, 'array_radius', np.nan)
        num_mics = getattr(self, 'num_mics', np.nan)
        src_center = np.mean(np.array(self.src_positions), axis=0)

        start_sample = int(fs * self.start_time)
        end_sample = int(fs * self.end_time)

        for i, rec in enumerate(self.recorded_responses):
            ref_pos = self.ref_positions[i]
            srcL = self.src_positions[0]

            delayed_ref = self.delay_test_signal(self.reference_input[:, 0], srcL, ref_pos, numtaps)

            # Clip both to specified region
            rec_clipped = rec[start_sample:end_sample]
            ref_clipped = delayed_ref[start_sample:end_sample]

            # Match lengths (safety)
            min_len = min(len(rec_clipped), len(ref_clipped))
            rec_clipped = rec_clipped[:min_len]
            ref_clipped = ref_clipped[:min_len]

            # Normalize reference
            ''' 
            ref_p = np.percentile(np.abs(ref_clipped), 95)
            rec_p = np.percentile(np.abs(rec_clipped), 95)
            if ref_p > 0:
                ref_clipped *= (rec_p / ref_p)
            '''

            # Time-domain metrics
            mse = np.mean((rec_clipped - ref_clipped) ** 2)
            snr = 10 * np.log10(np.mean(ref_clipped ** 2) / (mse + 1e-10))
            correlation = np.correlate(rec_clipped, ref_clipped, mode='valid')[0]
            correlation /= (np.linalg.norm(rec_clipped) * np.linalg.norm(ref_clipped) + 1e-10)

            # Frequency-domain metrics
            f, _, Zxx_ref = stft(ref_clipped, fs=fs, nperseg=n_fft)
            _, _, Zxx_test = stft(rec_clipped, fs=fs, nperseg=n_fft)

            min_frames = min(Zxx_ref.shape[1], Zxx_test.shape[1])
            Zxx_ref = Zxx_ref[:, :min_frames]
            Zxx_test = Zxx_test[:, :min_frames]

            mag_ref = np.abs(Zxx_ref)
            mag_test = np.abs(Zxx_test)
            epsilon = 1e-10
            mag_err_db = 20 * np.log10((np.abs(mag_ref - mag_test) + epsilon) / (mag_ref + epsilon))
            spectral_mag_error = np.mean(np.abs(mag_err_db))

            phase_ref = np.angle(Zxx_ref)
            phase_test = np.angle(Zxx_test)
            phase_diff = phase_ref - phase_test
            wrapped_phase_err = np.angle(np.exp(1j * phase_diff))  # maps to [-π, π]
            spectral_phase_error = np.mean(np.abs(wrapped_phase_err))

            combined_score = analysis_utils.compute_combined_error_score(snr, spectral_mag_error, spectral_phase_error)

            mic_dist = np.linalg.norm(np.array(ref_pos) - src_center)

            results.append({
                'Mode': mode,
                'Array Geometry': array_geometry,
                'Array Radius (m)': array_radius,
                'Num Mics': num_mics,
                'Mic Index': i,
                'Source - Mic Dist (m)': round(mic_dist, 3),
                'SNR (dB)': round(snr, 2),
                'MSE': round(mse, 6),
                'Correlation': round(correlation, 4),
                'Spectral Mag Error (dB)': round(spectral_mag_error, 2),
                'Spectral Phase Error (rad)': round(spectral_phase_error, 2),
                'Combined Score': round(combined_score, 3),
            })

        df = pd.DataFrame(results)

        # Save clipped version
        if csv_path is None:
            csv_path = f"../test_files/csv/fir_eval_clipped_{mode}.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None
        df.to_csv(csv_path, index=False)
        print(f"✅ Clipped evaluation metrics saved to: {csv_path}")

        return df

    
    def plot_error_vs_input_with_freq_axis(
        self,
        duration_sec=5.0,
        start_time=0.0,
        end_time=5,
        start_freq=20,
        end_freq=20000,
        numtaps=512
    ):
        """
        Plots FIR-corrected response vs delayed input using frequency as x-axis.
        Accounts for FIR + propagation delay. Assumes mono sweep input (L channel).
        """
        num_samples = int(self.fs * duration_sec) + self.total_delay
        t = np.arange(num_samples) / self.fs  # Absolute time

        # Actual sweep starts after delay
        sweep_start_time = self.total_delay / self.fs
        sweep_duration = duration_sec

        # Time vector relative to sweep onset
        t_relative = t - sweep_start_time

        # Compute frequency for each sample (linear sweep)
        freqs = start_freq + (end_freq - start_freq) * (t_relative / sweep_duration)

        # Clip to user-specified window within sweep
        mask = (t_relative >= start_time) & (t_relative <= end_time)
        freqs = freqs[mask]

        for i, rec in enumerate(self.recorded_responses):
            ref_pos = self.ref_positions[i]
            src_pos = self.src_positions[0]

            # Delay input (adds silence before sweep)
            delayed_ref = self.delay_test_signal(self.reference_input[:, 0], src_pos, ref_pos, numtaps)
            delayed_ref = delayed_ref[:num_samples]

            # Trim recorded response to match
            rec_trimmed = rec[:num_samples]

            # Apply mask
            delayed_ref = delayed_ref[mask]
            rec_trimmed = rec_trimmed[mask]

            # Plot
            plt.figure(figsize=(12, 4))
            plt.plot(freqs, delayed_ref, label=f"Delayed Input (Mono) – {duration_sec:.1f}s Sweep", linestyle='--')
            plt.plot(freqs, rec_trimmed, label=f"Recorded Response @ Ref {i}")
            plt.title(f"Ref Mic {i} – Mode: {getattr(self, 'mode', 'N/A')} – FIR-Corrected vs Delayed Input")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.xlim([start_freq, end_freq])
            plt.ylim([-2, 2])
            plt.show(block=False)




    
