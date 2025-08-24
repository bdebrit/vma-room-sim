import numpy as np
from scipy.linalg import toeplitz, lstsq
from scipy.signal import lfilter
import pickle
from tqdm import tqdm
import os

class FIRCorrectionFilter:
    def __init__(self, numtaps=256):
        self.numtaps = numtaps
        self.h_left = None
        self.h_right = None

    def _design_fir_multichannel(self, virtual_signals, target_signal, lambda_reg=1e-2):
        """
        Regularized FIR design using multichannel virtual signals and target signal.

        Args:
            virtual_signals (List[np.ndarray]): List of input virtual mic signals.
            target_signal (np.ndarray): Desired output signal.
            lambda_reg (float): Regularization strength (Tikhonov).

        Returns:
            np.ndarray: Concatenated FIR filter coefficients across all channels.
        """
        from scipy.linalg import toeplitz

        N = len(target_signal)
        num_channels = len(virtual_signals)
        X_all = []

        print("📐 Designing REGULARIZED FIR filter from multichannel virtual mic signals...")
        for ch in tqdm(range(num_channels), desc="Building Toeplitz matrices"):
            x = np.concatenate([virtual_signals[ch], np.zeros(self.numtaps - 1)])
            col = x[self.numtaps - 1:N + self.numtaps - 1]
            row = x[self.numtaps - 1::-1]
            X = toeplitz(col, row)
            X_all.append(X[:N])

        X_stacked = np.hstack(X_all)
        y = target_signal[:X_stacked.shape[0]]

        # Regularized least squares: (XᵀX + λI)h = Xᵀy
        XtX = X_stacked.T @ X_stacked
        reg_matrix = lambda_reg * np.eye(XtX.shape[0])
        Xty = X_stacked.T @ y
        h = np.linalg.solve(XtX + reg_matrix, Xty)

        return h
    
    def _design_fir_from_averaged_virtuals(self, virtual_signals, target_signal):
        """
        Averages multichannel virtual mic signals (after trimming) and designs an FIR filter.

        Args:
            virtual_signals (List[np.ndarray]): List of virtual mic signals.
            target_signal (np.ndarray): Desired target output signal.

        Returns:
            np.ndarray: FIR filter taps.
        """
        print("📐 Designing FIR filter from averaged virtual mic signals...")

        # Step 1: Trim all virtuals to the length of the shortest one
        min_len = min(len(v) for v in virtual_signals)
        virtuals_trimmed = [v[:min_len] for v in virtual_signals]

        # Step 2: Average across the stacked virtuals
        avg_virtual = np.mean(np.vstack(virtuals_trimmed), axis=0)

        # Step 3: Build Toeplitz and fit FIR
        N = min(min_len, len(target_signal))
        x = np.concatenate([avg_virtual, np.zeros(self.numtaps - 1)])
        col = x[self.numtaps - 1:N + self.numtaps - 1]
        row = x[self.numtaps - 1::-1]
        X = toeplitz(col, row)

        y = target_signal[:X.shape[0]]
        h, *_ = lstsq(X, y)
        return h


    
    def train_from_single_virtual_mic(self, virtual_signal, test_signal_left, test_signal_right, numtaps=None):
        """
        Train FIR filters using a single virtual mic signal as input (same for both channels).
        """
        if numtaps is not None:
            self.numtaps = numtaps

        print("🎯 Training FIR filters from single virtual mic signal...")
        self.h_left = self._design_fir_multichannel([virtual_signal], test_signal_left)
        self.h_right = self._design_fir_multichannel([virtual_signal], test_signal_right)
        print("✅ FIR filters trained from single-channel input.")

    def train_from_averaged_virtuals(self, virtual_signals, test_signal_left, test_signal_right, numtaps=None):
        """
        Train FIR filters using the average of multiple virtual mic signals.

        Args:
            virtual_signals (List[np.ndarray]): List of virtual mic signals.
            test_signal_left (np.ndarray): Left channel of target stereo signal.
            test_signal_right (np.ndarray): Right channel of target stereo signal.
            numtaps (int): Optional override for number of FIR taps.
        """
        if numtaps is not None:
            self.numtaps = numtaps

        print("🎯 Training FIR filters from averaged 5-channel virtual mic signals...")
        self.h_left = self._design_fir_from_averaged_virtuals(virtual_signals, test_signal_left)
        self.h_right = self._design_fir_from_averaged_virtuals(virtual_signals, test_signal_right)
        print("✅ FIR filters trained (L & R) from averaged virtual input.")


    def train_from_simulation(self, virtual_left_set, ref_left, virtual_right_set, ref_right):
        self.h_left = self._design_fir_multichannel(virtual_left_set, ref_left)
        self.h_right = self._design_fir_multichannel(virtual_right_set, ref_right)
        print("✅ FIR filters trained from 5-channel virtual inputs (L and R)")

    def train_to_input_signal_stereo(self, virtual_signals, test_signal_left, test_signal_right, numtaps=None):
        """
        Train FIR filters to reconstruct original stereo input signal from virtual mic responses.
        Assumes same 5-channel input for both channels, but different targets (e.g., stereo music).
        """
        if numtaps is not None:
            self.numtaps = numtaps

        print("🎯 Training FIR filters to recover input stereo signal from virtual mic signals...")
        self.h_left = self._design_fir_multichannel(virtual_signals, test_signal_left)
        self.h_right = self._design_fir_multichannel(virtual_signals, test_signal_right)
        print("✅ FIR filters trained for L and R channels (from same 5 virtual inputs)")

    def apply(self, stereo_audio):
        if self.h_left is None or self.h_right is None:
            raise ValueError("FIR filters not trained yet.")

        left = lfilter(self.h_left, [1.0], stereo_audio[:, 0])
        right = lfilter(self.h_right, [1.0], stereo_audio[:, 1])
        return np.stack([left, right], axis=-1)

    def export(self, mode="Unknown", directory="test_files/bin"):
        if mode == "Unknown":
            print("Unkown FIR method, load aborted.")
            exit(1)
        left_path = directory + f"/fir_left_{mode}.npy"
        right_path = directory + f"/fir_right_{mode}.npy"
        np.save(left_path, self.h_left)
        np.save(right_path, self.h_right)
        print(f"✅ FIR filters exported: {left_path}, {right_path}")


    def load(self, mode="Unknown", directory="test_files/bin"):

        if mode == "Unknown":
            print("Unkown FIR method, load aborted.")
            exit(1)
        left_path = directory + f"/fir_left_{mode}.npy"
        right_path = directory + f"/fir_right_{mode}.npy"
        self.h_left = np.load(left_path)
        self.h_right = np.load(right_path)
        print(f"✅ FIR filters loaded: {left_path}, {right_path}")


    @staticmethod
    def get_virtual_and_ref_sets_from_checkpoint(checkpoint, base_label, ref_key="ref"):
        index = checkpoint["index_by_config"][base_label]
        virtuals = [checkpoint["dataset"][lbl]["virtual"] for lbl in index]
        refs = [checkpoint["dataset"][lbl][ref_key] for lbl in index]
        return virtuals, refs

    @classmethod
    def train_fir_from_checkpoint_multi(cls, dataset_checkpoint_path, top_rank_index=0, numtaps=256):
        """
        Trains stereo FIR filters to recover the original stereo input from virtual mic outputs
        in a room-style simulation.
        """
        with open(dataset_checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

        signal_stereo = checkpoint["signal_stereo"][0]  # shape: (2, N)
        if signal_stereo.ndim != 2:
            raise ValueError("Expected stereo input signal with shape (2, N) in checkpoint.")

        left_input = signal_stereo[0]
        right_input = signal_stereo[1]

        ranking_df = checkpoint["ranking_df"]
        best_label = ranking_df.iloc[top_rank_index]
        base_label = f"geometry={best_label.geometry},method={best_label.method},radius={best_label.radius}," \
                     f"num_mics={best_label.num_mics},mic_array_order={best_label.mic_array_order}," \
                     f"array=room_fixed,audio_channel_format=stereo"

        label_list = checkpoint["index_by_config"][base_label]
        virtual_signals = [checkpoint["dataset"][lbl]["virtual"] for lbl in label_list]

        fir = cls(numtaps=numtaps)
        fir.train_to_input_signal_stereo(virtual_signals, left_input, right_input)
        fir.export(mode='multi')
        return fir
    
    @classmethod
    def train_from_checkpoint_single(cls, dataset_checkpoint_path, top_rank_index=0, numtaps=256):
        """
        Trains FIR filters using only the single virtual mic output at the central position (room_pos0).
        Useful for evaluating correction based on one mic only.

        Args:
            dataset_checkpoint_path (str): Path to the .pkl simulation checkpoint.
            top_rank_index (int): Ranking index to select config (default = 0 = top ranked).
            numtaps (int): Number of FIR taps.

        Returns:
            FIRCorrectionFilter: Trained instance with h_left and h_right populated.
        """
        with open(dataset_checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

        signal_stereo = checkpoint["signal_stereo"][0]
        if signal_stereo.ndim != 2 or signal_stereo.shape[0] != 2:
            raise ValueError("Expected stereo input signal with shape (2, N)")

        left_input = signal_stereo[0]
        right_input = signal_stereo[1]

        ranking_df = checkpoint["ranking_df"]
        best_label = ranking_df.iloc[top_rank_index]
        base_label = f"geometry={best_label.geometry},method={best_label.method},radius={best_label.radius}," \
                    f"num_mics={best_label.num_mics},mic_array_order={best_label.mic_array_order}," \
                    f"array=room_fixed,audio_channel_format=stereo"

        # Use only room_pos0
        center_label = f"{base_label},ref=room_pos0"
        virtual_center = checkpoint["dataset"][center_label]["virtual"]

        fir = cls(numtaps=numtaps)
        fir.train_from_single_virtual_mic(virtual_center, left_input, right_input)
        fir.export(mode='single')
        return fir
    
    @classmethod
    def train_from_checkpoint_avg(cls, dataset_checkpoint_path, top_rank_index=0, numtaps=256):
        """
        Trains FIR filters from the averaged 5 virtual mic inputs based on simulation checkpoint.

        Args:
            dataset_checkpoint_path (str): Path to simulation checkpoint file (.pkl).
            top_rank_index (int): Ranking index from evaluation.
            numtaps (int): Number of FIR taps.

        Returns:
            FIRCorrectionFilter: Trained filter instance.
        """
        with open(dataset_checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

        signal_stereo = checkpoint["signal_stereo"][0]  # shape: (2, N)
        if signal_stereo.ndim != 2:
            raise ValueError("Expected stereo input signal with shape (2, N)")

        left_input = signal_stereo[0]
        right_input = signal_stereo[1]

        ranking_df = checkpoint["ranking_df"]
        best_label = ranking_df.iloc[top_rank_index]
        base_label = f"geometry={best_label.geometry},method={best_label.method},radius={best_label.radius}," \
                    f"num_mics={best_label.num_mics},mic_array_order={best_label.mic_array_order}," \
                    f"array=room_fixed,audio_channel_format=stereo"

        label_list = checkpoint["index_by_config"][base_label]
        virtual_signals = [checkpoint["dataset"][lbl]["virtual"] for lbl in label_list]

        fir = cls(numtaps=numtaps)
        fir.train_from_averaged_virtuals(virtual_signals, left_input, right_input)
        fir.export(mode='avg')
        return fir

