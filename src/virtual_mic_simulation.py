# -------------------------------------------------------------
# Virtual Microphone Simulation
# -------------------------------------------------------------

# Import required libraries
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from scipy.signal import resample, correlate
from scipy.spatial.distance import euclidean
from sklearn.metrics import mean_squared_error
import scipy.io.wavfile as wav
from pydub import AudioSegment
import numpy as np
from scipy.signal import resample
import os
import soundfile as sf
from scipy.signal import chirp
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import stft, istft
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import pandas as pd
import os
import soundfile as sf
from numpy.linalg import inv
from itertools import product
from tqdm import tqdm
import pickle
import seaborn as sns
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import art3d
from fir_room_correction import FIRCorrectionFilter
from virtual_mic_test import VirtualMicTest

class VirtualMicSimulation:
    def __init__(self, dataset_checkpoint_path, room_dims=[12, 10, 4.0]):
        self.dataset_checkpoint_path = dataset_checkpoint_path
        self.room_dims = room_dims
        self.virtual_mic_positions = None
        self.src_positions = None
        self.fs = 48000
        self.simulated_signals = []
    def generate_array_geometry(self, center, geometry_type='spherical_uniform', radius=0.1, num_mics=8, seed=None):
        """
        Generate microphone array geometry.

        Parameters:
            center (list): [x, y, z] coordinates of array center.
            geometry_type (str): Type of geometry to generate. Options:
                'spherical_uniform' (deterministic Fibonacci sphere),
                'spherical_random' (random on sphere surface, reproducible via seed),
                'tetrahedral', 'octahedral', 'icosahedral'
            radius (float): Distance from center to microphones.
            num_mics (int): Used for spherical_uniform / spherical_random.
            seed (int): RNG seed for spherical_random (and optional jitter if you extend later)

        Returns:
            np.ndarray: Shape (3, N) array of microphone positions.
        """
        center = np.array(center, dtype=float).reshape(3,)

        if geometry_type == 'spherical_uniform':
            # Deterministic uniform-ish spherical points using Fibonacci spiral
            n = int(num_mics)
            indices = np.arange(0, n, dtype=float) + 0.5
            phi = np.arccos(1.0 - 2.0 * indices / n)
            theta = np.pi * (1.0 + 5.0**0.5) * indices

            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
            positions = np.vstack((x, y, z))

        elif geometry_type == 'spherical_random':
            # Random points uniformly distributed on sphere surface (reproducible via seed)
            rng = np.random.default_rng(seed)
            pts = rng.standard_normal((3, int(num_mics)))
            pts /= (np.linalg.norm(pts, axis=0, keepdims=True) + 1e-12)
            positions = radius * pts

        elif geometry_type == 'tetrahedral':
            # 4 points on a tetrahedron
            scale = radius / np.sqrt(3.0)
            positions = scale * np.array([
                [1,  1,  1],
                [-1, -1,  1],
                [-1,  1, -1],
                [1, -1, -1]
            ], dtype=float).T

        elif geometry_type == 'octahedral':
            # 6 points on an octahedron
            positions = radius * np.array([
                [1, -1,  0,  0,  0,  0],
                [0,  0,  1, -1,  0,  0],
                [0,  0,  0,  0,  1, -1]
            ], dtype=float)

        elif geometry_type == 'icosahedral':
            # 12 points of an icosahedron
            phi = (1.0 + np.sqrt(5.0)) / 2.0
            verts = []
            for i in [-1, 1]:
                for j in [-1, 1]:
                    verts += [
                        [0.0, float(i), float(j) * phi],
                        [float(i), float(j) * phi, 0.0],
                        [float(i) * phi, 0.0, float(j)]
                    ]
            verts = np.array(verts, dtype=float).T
            # Normalize to unit radius then scale
            verts = verts / (np.linalg.norm(verts[:, 0]) + 1e-12)
            positions = radius * verts

        else:
            raise ValueError(f"Unsupported geometry type: {geometry_type}")

        return positions + center.reshape(3, 1)

    def estimate_array_order(self, geometry_type, num_mics):
        """
        Estimate the spatial/beamforming order of a microphone array.
        For spherical arrays, based on Ambisonics resolution.
        """
        if geometry_type == "spherical_uniform":
            return max(int(np.floor(np.sqrt(num_mics)) - 1), 0)
        elif geometry_type == "tetrahedral":
            return 1  # Typically 1st-order
        elif geometry_type == "octahedral":
            return 2  # Up to 2nd-order
        elif geometry_type == "icosahedral":
            return 3  # 12 mics (common full sphere configuration)
        else:
            return None


    def create_simulation_room(self, room_dims, fs, absorption=0.3, max_order=5, use_reflections=True):
        """
        Create a Pyroomacoustics simulation room with given parameters.

        Args:
            room_dims (list): Room dimensions [x, y, z].
            fs (int): Sampling frequency.
            absorption (float): Absorption coefficient for the walls.
            max_order (int): Maximum order of reflections.

        Returns:
            pra.ShoeBox: Room object ready for sources and mic arrays.
        """

        if not use_reflections:
            max_order = 0

        room = pra.ShoeBox(
            p=room_dims,
            fs=fs,
            materials=pra.Material(absorption),
            max_order=max_order
        )

        return room


    def compute_virtual_microphone_delay_sum_time_beamformer(self, room, virtual_position, array_positions, window_type="hamming"):
        """
        Estimate the signal at a virtual microphone position using time-domain fractional delay alignment.

        Parameters:
            room (pra.ShoeBox): Simulated room.
            virtual_position (list or np.ndarray): Target position for virtual mic [x, y, z].
            array_positions (np.ndarray): (3, N) mic array positions.
            window_type (str): 'hamming' or 'kaiser'

        Returns:
            np.ndarray: Estimated virtual mic signal (time domain).
        """
        def fractional_delay(signal, delay, filter_length=101, beta=8.6):
            n = np.arange(-filter_length // 2, filter_length // 2 + 1)
            h = np.sinc(n - delay)
            if window_type == "hamming":
                h *= np.hamming(len(h))
            elif window_type == "kaiser":
                h *= np.kaiser(len(h), beta)
            elif window_type == "blackman":
                h *= np.blackman(len(h))
            else:
                raise ValueError(f"Unsupported window type: {window_type}")
            h /= np.sum(h)
            return np.convolve(signal, h, mode='same')


        speed_of_sound = 343.0
        fs = room.fs
        n_mics = array_positions.shape[1]
        mic_signals = room.mic_array.signals[:n_mics]

        # Compute delays to the virtual mic position
        delays = []
        for mic_pos in array_positions.T:
            d = np.linalg.norm(mic_pos - virtual_position)
            delay_samples = (d / speed_of_sound) * fs
            delays.append(delay_samples)

        max_delay = max(delays)

        # Apply delay compensation
        aligned_signals = []
        for i in range(n_mics):
            fractional_shift = max_delay - delays[i]
            delayed = fractional_delay(mic_signals[i], fractional_shift)
            aligned_signals.append(delayed)

        # Average across all aligned mic signals
        virtual_signal = np.mean(aligned_signals, axis=0)

        # Normalize RMS to match mic 0
        ref_rms = np.sqrt(np.mean(mic_signals[0] ** 2))
        vm_rms = np.sqrt(np.mean(virtual_signal ** 2))
        virtual_signal *= ref_rms / (vm_rms + 1e-10)

        return virtual_signal


    def compute_virtual_microphone_delay_sum_freq_beamformer(self, room, virtual_position, array_positions):
        """
        Estimate signal at a virtual microphone position using frequency-domain delay-and-sum beamforming.

        Parameters:
            room (pra.ShoeBox): Simulated room with sources and microphones.
            virtual_position (list or np.ndarray): Target position for virtual microphone [x, y, z].
            array_positions (np.ndarray): (3, N) array of microphone positions used in the array.

        Returns:
            np.ndarray: Time-domain estimated signal at the virtual mic position.
        """
        speed_of_sound = 343.0
        fs = room.fs
        n_mics = array_positions.shape[1]
        signals = room.mic_array.signals[:n_mics]
        signal_len = signals.shape[1]

        # FFT size (power of 2)
        n_fft = 2**int(np.ceil(np.log2(signal_len)))
        freqs = np.fft.rfftfreq(n_fft, d=1/fs)
        omega = 2 * np.pi * freqs

        # FFT each mic signal
        X = np.array([fft(sig, n=n_fft)[:len(freqs)] for sig in signals])

        # Calculate steering vector
        steering = np.zeros((n_mics, len(freqs)), dtype=complex)
        for i, mic_pos in enumerate(array_positions.T):
            d = np.linalg.norm(mic_pos - virtual_position)
            tau = d / speed_of_sound
            steering[i, :] = np.exp(-1j * omega * tau)

        # Frequency-domain beamforming (delay-and-sum)
        Y = np.sum(X * steering.conj(), axis=0) / n_mics

        # Convert back to time domain
        virtual_full = ifft(Y, n=n_fft).real
        virtual_signal = virtual_full[:signal_len]

        # Normalize to match reference mic RMS (optional)
        ref_rms = np.sqrt(np.mean(room.mic_array.signals[0]**2))
        vm_rms = np.sqrt(np.mean(virtual_signal**2))
        virtual_signal *= ref_rms / (vm_rms + 1e-10)

        return virtual_signal

    def compute_virtual_microphone_mvdr_beamformer(self, room, virtual_position, array_positions, n_fft=1024, diagonal_loading=1e-6):
        """
        MVDR beamforming using STFT across all frames.

        Args:
            room (pra.ShoeBox): Simulated room with sources and mic array.
            virtual_position (np.ndarray): [x, y, z] position of the virtual mic.
            array_positions (np.ndarray): (3, N) mic positions.
            n_fft (int): FFT size (also STFT frame size).
            diagonal_loading (float): Stability parameter for covariance matrix.

        Returns:
            np.ndarray: Reconstructed full-length virtual mic signal.
        """
        c = 343.0
        fs = room.fs
        n_mics = array_positions.shape[1]
        mic_signals = room.mic_array.signals[:n_mics]

        # STFT parameters
        hop_size = n_fft // 2
        window = pra.hamming(n_fft)

        # STFT of mic signals (shape: n_mics x n_bins x n_frames)
        X = []
        for sig in mic_signals:
            _, _, Zxx = stft(sig, fs=fs, nperseg=n_fft, noverlap=n_fft // 2, window=window)
            X.append(Zxx)
        X = np.stack(X)  # Shape: (n_mics, n_bins, n_frames)

        n_bins, n_frames = X.shape[1], X.shape[2]

        # Frequencies
        freqs = np.fft.rfftfreq(n_fft, d=1/fs)
        omega = 2 * np.pi * freqs

        # Steering vector (shape: n_mics x n_bins)
        steering = np.zeros((n_mics, n_bins), dtype=complex)
        for i, pos in enumerate(array_positions.T):
            delay = np.linalg.norm(pos - virtual_position) / c
            steering[i, :] = np.exp(-1j * omega * delay)

        # Beamform across all frames
        Y = np.zeros((n_bins, n_frames), dtype=complex)

        for f in range(n_bins):
            d = steering[:, f].reshape(-1, 1)
            for t in range(n_frames):
                x_f = X[:, f, t].reshape(-1, 1)
                R = x_f @ x_f.conj().T + diagonal_loading * np.eye(n_mics)

                # Efficient solve instead of inverse
                w = np.linalg.solve(R, d)
                norm = np.conj(d.T) @ w
                w /= norm + 1e-10

                Y[f, t] = np.conj(w.T) @ x_f

        # Inverse STFT to time domain
        _, y_time = istft(Y, fs=fs, nperseg=n_fft, noverlap=n_fft // 2, window=window)

        # RMS normalization (optional but helps with scale consistency)
        ref_rms = np.sqrt(np.mean(room.mic_array.signals[0] ** 2))
        y_rms = np.sqrt(np.mean(y_time ** 2))
        if y_rms > 0:
            y_time *= ref_rms / y_rms

        return y_time

    def compute_virtual_microphone(self, room, method, virtual_position, array_positions, **kwargs):
        if method == "delay_sum_time":
            return self.compute_virtual_microphone_delay_sum_time_beamformer(
                room, virtual_position, array_positions, window_type=kwargs.get('window_type', 'hamming')
            )
        elif method == "delay_sum_freq":
            return self.compute_virtual_microphone_delay_sum_freq_beamformer(
                room, virtual_position, array_positions
            )
        elif method == "mvdr":
            return self.compute_virtual_microphone_mvdr_beamformer(
                room, virtual_position, array_positions
            )
        else:
            raise ValueError(f"Unsupported beamforming method: {method}")



    def align_signals_correlation(self, ref_signal: np.ndarray, target_signal: np.ndarray) -> tuple[np.ndarray, int]:
        """
        Aligns the target signal to the reference signal using cross-correlation.

        Parameters:
            ref_signal (np.ndarray): The reference signal.
            target_signal (np.ndarray): The signal to be aligned.

        Returns:
            aligned_signal (np.ndarray): The delay-corrected target signal.
            lag (int): The number of samples the target was shifted.
        """
        # Remove DC offset
        ref_signal = ref_signal - np.mean(ref_signal)
        target_signal = target_signal - np.mean(target_signal)

        # Compute cross-correlation and find lag
        correlation = correlate(target_signal, ref_signal, mode='full')
        lag = np.argmax(correlation) - len(ref_signal) + 1

        # Apply lag to align the target signal
        if lag > 0:
            aligned_signal = target_signal[lag:]
        elif lag < 0:
            aligned_signal = np.pad(target_signal, (abs(lag), 0), mode='constant')[:len(ref_signal)]
        else:
            aligned_signal = target_signal

        # Final trim to match length
        min_len = min(len(aligned_signal), len(ref_signal))
        aligned_signal = aligned_signal[:min_len]
        ref_signal = ref_signal[:min_len]

        return aligned_signal, lag


    def plot_short_time_snr_comparison(self, results_dict, fs=48000, window_size=2048, hop_size=1024):
        """
        Plots short-time SNR curves over time for multiple configurations.

        Parameters:
            results_dict (dict): 
                Key = label (e.g., 'radius = 0.1'), 
                Value = tuple(ref_signal, test_signal)
            fs (int): Sampling rate (Hz)
            window_size (int): Number of samples per window
            hop_size (int): Step size between windows
        """
        time_axis = None
        plt.figure(figsize=(10, 5))

        for label, (ref, test) in results_dict.items():
            min_len = min(len(ref), len(test))
            ref = ref[:min_len]
            test = test[:min_len]

            snr_values = []
            times = []
            for i in range(0, min_len - window_size, hop_size):
                ref_win = ref[i:i+window_size]
                test_win = test[i:i+window_size]
                signal_power = np.mean(ref_win ** 2)
                noise_power = np.mean((ref_win - test_win) ** 2) + 1e-10
                snr = 10 * np.log10(signal_power / noise_power)
                snr_values.append(snr)
                times.append(i / fs)

            plt.plot(times, snr_values, label=label)

        plt.title("Short-Time SNR Comparison")
        plt.xlabel("Time (s)")
        plt.ylabel("SNR (dB)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)


    def plot_frequency_magnitude_error_overlay(
        self,
        results_dict,
        fs=48000,
        window='hann',
        n_fft=2048,
        title="Overlayed Frequency Magnitude Errors",
        smoothing_window=15  # ⬅️ size of the moving average window (in frequency bins)
    ):
        """
        Plots overlayed STFT magnitude error (in dB) over frequency for multiple configurations,
        with optional smoothing applied.

        Parameters:
            results_dict (dict): 
                Key = label (e.g., 'radius = 0.1'), 
                Value = tuple(ref_signal, test_signal)
            fs (int): Sampling rate
            window (str): Window type for STFT
            n_fft (int): FFT length
            title (str): Plot title
            smoothing_window (int): Width of moving average filter
        """
        plt.figure(figsize=(10, 5))

        for label, (ref, test) in results_dict.items():
            min_len = min(len(ref), len(test))
            ref = ref[:min_len]
            test = test[:min_len]

            # STFT
            f, _, Zxx_ref = stft(ref, fs=fs, window=window, nperseg=n_fft)
            _, _, Zxx_test = stft(test, fs=fs, window=window, nperseg=n_fft)

            epsilon = 1e-10
            mag_ref = np.abs(Zxx_ref)
            mag_test = np.abs(Zxx_test)
            error_db = 20 * np.log10((np.abs(mag_ref - mag_test) + epsilon) / (mag_ref + epsilon))
            mean_error_db = np.mean(np.abs(error_db), axis=1)

            # Apply simple moving average for smoothing
            smoothed_error = np.convolve(mean_error_db, np.ones(smoothing_window)/smoothing_window, mode='same')

            plt.plot(f, smoothed_error, label=label)

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Mean Magnitude Error (dB)")
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)

    def plot_frequency_phase_error_overlay(self, results_dict, fs=48000, window='hann', n_fft=2048, title="Overlayed Frequency Phase Errors"):
        """
        Plots overlayed STFT phase error (in radians) over frequency for multiple configurations.

        Parameters:
            results_dict (dict): 
                Key = label (e.g., 'radius = 0.1'), 
                Value = tuple(ref_signal, test_signal)
            fs (int): Sampling rate
            window (str): Window type for STFT
            n_fft (int): FFT length
            title (str): Plot title
        """
        plt.figure(figsize=(10, 5))

        for label, (ref, test) in results_dict.items():
            # Ensure signals are same length
            min_len = min(len(ref), len(test))
            ref = ref[:min_len]
            test = test[:min_len]

            # STFT
            f, _, Zxx_ref = stft(ref, fs=fs, window=window, nperseg=n_fft)
            _, _, Zxx_test = stft(test, fs=fs, window=window, nperseg=n_fft)

            # Phase error in radians
            phase_ref = np.angle(Zxx_ref)
            phase_test = np.angle(Zxx_test)
            phase_error = np.unwrap(phase_ref - phase_test, axis=0)
            mean_phase_error = np.mean(np.abs(phase_error), axis=1)

            plt.plot(f, mean_phase_error, label=label)

        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Mean Phase Error (radians)")
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)


    def fractional_delay(self, signal, delay, fs, filter_length=81):
        """
        Apply a fractional delay to a signal using sinc interpolation.
        
        Parameters:
            signal (np.ndarray): Input signal
            delay (float): Delay in samples (can be fractional)
            fs (int): Sampling rate
            filter_length (int): Sinc filter length (default 81)
        
        Returns:
            np.ndarray: Fractionally delayed signal
        """
        n = np.arange(-filter_length//2, filter_length//2 + 1)
        h = np.sinc(n - delay)
        h *= np.hamming(len(h))
        h /= np.sum(h)
        return np.convolve(signal, h, mode='same')


    def plot_room_configuration(
        self,
        room_positions,
        eval_positions=None,
        mic_positions=None,
        room_dims=None,
        title="Room Configuration"
    ):
        """
        Plot the room layout: mic array, reference mic positions, sources, and triangle geometry.

        Args:
            room_positions (dict): Output from get_room_style_positions(...)
            mic_positions (np.ndarray or None): Optional mic array positions (3, N)
            room_dims (list or None): Room dimensions [X, Y, Z]
            title (str): Plot title
        """
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib.patches as mpatches
        from mpl_toolkits.mplot3d import art3d

        def sort_points_radially(points, center):
            vectors = points - center
            angles = np.arctan2(vectors[:, 1], vectors[:, 0])
            return points[np.argsort(angles)]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Plot mic array center
        mac = room_positions['mic_array_centre']
        ax.scatter(*mac, color='orange', marker='o', s=40, label='Mic Array Centre')

        # Plot mic array elements
        if mic_positions is not None:
            ax.scatter(mic_positions[0], mic_positions[1], mic_positions[2],
                    color='blue', marker='.', s=30, label='Array Mics')

        # Plot evaluation positions and connecting lines
        if mic_positions is not None and eval_positions is not None:
            mic_centre = np.mean(mic_positions, axis=1)
            for ref_pos in eval_positions:
                ax.plot(
                    [mic_centre[0], ref_pos[0]],
                    [mic_centre[1], ref_pos[1]],
                    [mic_centre[2], ref_pos[2]],
                    linestyle='dotted', linewidth=1, color='blue', alpha=0.6
                )

        # Plot evaluation mic positions
        if eval_positions is not None:
            for i, pos in enumerate(eval_positions):
                ax.scatter(*pos, color='red', marker='^', s=60)
                ax.text(pos[0], pos[1], pos[2] + 0.1, f"Ref{i}", color='red', fontsize=9)

            if len(eval_positions) >= 2:
                center = np.array(eval_positions[0])
                radius = np.linalg.norm(np.array(eval_positions[1]) - center)
                circle = mpatches.Circle(
                    (center[0], center[1]), radius,
                    linestyle='dotted', linewidth=1, edgecolor='red', fill=False, label="Evaluation Radius"
                )
                ax.add_patch(circle)
                art3d.pathpatch_2d_to_3d(circle, z=center[2], zdir="z")

        # ✅ Sort sources by Y (ensure left is lower Y)
        src1, src2 = room_positions['source_positions']
        if src1[1] >= src2[1]:
            srcL, srcR = src1, src2
        else:
            srcL, srcR = src2, src1

        ref = room_positions['ref_mic_pos']
        ax.scatter(*srcL, color='green', marker='s', s=80, label='Source L')
        ax.scatter(*srcR, color='green', marker='s', s=80, label='Source R')
        ax.text(srcL[0], srcL[1], srcL[2] + 0.3, "Source L", color='green', fontsize=9)
        ax.text(srcR[0], srcR[1], srcR[2] + 0.3, "Source R", color='green', fontsize=9)

        # 🔺 Draw triangle lines
        for src in [srcL, srcR]:
            ax.plot(
                [src[0], ref[0]],
                [src[1], ref[1]],
                [src[2], ref[2]],
                linestyle='dashed', color='gray', linewidth=1, alpha=0.7
            )
        ax.plot(
            [srcL[0], srcR[0]],
            [srcL[1], srcR[1]],
            [srcL[2], srcR[2]],
            linestyle='dashed', color='gray', linewidth=1, alpha=0.7
        )

        # Aesthetic adjustments
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        ax.view_init(elev=20, azim=130)

        if room_dims:
            ax.set_xlim(0, room_dims[0])
            ax.set_ylim(0, room_dims[1])
            ax.set_zlim(0, room_dims[2])
            ax.set_box_aspect([room_dims[0], room_dims[1], room_dims[2]])  # [X, Y, Z]

        plt.tight_layout()




    def get_room_style_positions(self, room_dims):
        """
        Places two speakers and a reference mic in an equilateral triangle.
        Speaker spacing is W/2 (1:2:1 width layout). The listener (ref mic)
        is placed behind the speaker line to complete the triangle.

        Returns:
            dict with:
                - source_positions: [left, right]
                - ref_mic_pos: listener at triangle apex
                - mic_array_centre: above listener
        """
        import numpy as np

        L, W, H = room_dims
        height_ref = 1.5
        height_array = H - 0.75

        # Spatial layout
        y_left = W / 4
        y_right = 3 * W / 4
        y_center = W / 2

        spacing = y_right - y_left

        triangle_height = (np.sqrt(3) / 2) * spacing
        x_src = L * 0.9
        x_ref = x_src - triangle_height 

        ref_position = [x_ref, y_center, height_ref]
        mic_array_centre = [x_ref, y_center, height_array]

        # Make sure left has lower Y, right has higher Y
        left_src = [x_src, min(y_left, y_right), height_ref]
        right_src = [x_src, max(y_left, y_right), height_ref]

        return {
            'source_positions': [left_src, right_src],
            'ref_mic_pos': ref_position,
            'mic_array_centre': mic_array_centre
        }











    def get_room_evaluation_positions(self, center_pos, radius=1.0):
        """
        Returns 5 positions: center + 4 on a horizontal circle (front, back, left, right).
        """
        cx, cy, cz = center_pos
        offsets = [
            (0.0, 0.0),  # center
            (radius, 0.0),
            (-radius, 0.0),
            (0.0, radius),
            (0.0, -radius)
        ]
        return [[cx + dx, cy + dy, cz] for dx, dy in offsets]
    
    def triangle_with_scaled_spacing(self, room_dims, z_height=1.5, spacing_ratio=1/3):
        """
        Place an equilateral triangle near the rear wall with spacing based on room width.

        Args:
            room_dims (list): Room dimensions [X, Y, Z]
            z_height (float): Height in meters for sources and virtual mic
            spacing_ratio (float): Ratio of room width to use as speaker spacing (default = 1/3)

        Returns:
            tuple: (virtual_mic_position, source1_position, source2_position)
        """
        room_x, room_y, _ = room_dims

        # Compute speaker spacing across width
        spacing = room_y * spacing_ratio

        # Rear offset (2.5 m from rear wall)
        rear_offset = 2.5
        center_x = room_x - rear_offset
        center_y = room_y / 2

        # Base of triangle (sources)
        s1_y = center_y - spacing / 2
        s2_y = center_y + spacing / 2
        s1 = np.array([center_x, s1_y, z_height])
        s2 = np.array([center_x, s2_y, z_height])

        # Apex of triangle (virtual mic), 60° triangle height
        height = np.sqrt(3) / 2 * spacing
        vm_x = center_x - height  # Pointing forward into the room
        vm = np.array([vm_x, center_y, z_height])

        return vm, s1, s2

    