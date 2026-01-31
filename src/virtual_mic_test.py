import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import analysis_utils


class VirtualMicTest:
    """Lightweight VMA-only test/inspection helper.

    This class intentionally contains **no FIR correction logic**. It is meant to:
      - load a simulation checkpoint (.pkl) produced by your VMA runs
      - select a top-ranked configuration (or any row index)
      - expose ref/virtual signals per reference position (room_pos0..room_pos4)
      - compute summary metrics and optionally plot quick comparisons

    Expected checkpoint keys (typical):
      - dataset: dict[label -> dict with keys: 'ref', 'virtual', 'ref_position', ...]
      - ranking_df: pd.DataFrame (optional but recommended)
      - fs: int
      - room_dims: list/tuple
    """

    def __init__(self, room_dims, fs, dataset, ranking_df=None):
        self.room_dims = room_dims
        self.fs = int(fs)
        self.dataset = dataset
        self.ranking_df = ranking_df

        self.best_config_row = None
        self.base_label = None

        # Populated after selecting a configuration
        self.ref_positions = []
        self.src_positions = None
        self.mic_array_positions = None

        self.ref_signals = []      # list of 1D np.ndarray (per ref position)
        self.virtual_signals = []  # list of 1D np.ndarray (per ref position)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, top_rank_index: int = 0):
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

        dataset = checkpoint.get("dataset", checkpoint)
        ranking_df = checkpoint.get("ranking_df", None)
        fs = checkpoint.get("fs", 48000)
        room_dims = checkpoint.get("room_dims", None)

        if room_dims is None:
            raise ValueError("Checkpoint is missing 'room_dims'.")

        inst = cls(room_dims=room_dims, fs=fs, dataset=dataset, ranking_df=ranking_df)

        if ranking_df is not None and len(ranking_df) > 0:
            inst.select_config_from_ranking(top_rank_index)
        else:
            # If no ranking_df exists, user can call select_config_by_base_label manually.
            pass

        return inst

    def select_config_from_ranking(self, top_rank_index: int = 0):
        if self.ranking_df is None or len(self.ranking_df) == 0:
            raise ValueError("ranking_df is not available in this checkpoint.")

        row = self.ranking_df.iloc[int(top_rank_index)]
        self.best_config_row = row

        # Keep label construction consistent with the rest of your pipeline
        self.base_label = (
            f"geometry={getattr(row, 'geometry', row.get('geometry', 'unknown'))},"
            f"method={getattr(row, 'method', row.get('method', 'unknown'))},"
            f"radius={getattr(row, 'radius', row.get('radius', np.nan))},"
            f"num_mics={getattr(row, 'num_mics', row.get('num_mics', np.nan))},"
            f"mic_array_order={getattr(row, 'mic_array_order', row.get('mic_array_order', np.nan))},"
            f"array=room_fixed,audio_channel_format=stereo"
        )

        self._load_signals_for_base_label(self.base_label)

    def select_config_by_base_label(self, base_label: str):
        """Use this if your checkpoint doesn't include a ranking_df."""
        self.base_label = base_label
        self._load_signals_for_base_label(base_label)

    def _load_signals_for_base_label(self, base_label: str):
        # Pull positions from ref=room_pos0 for convenience (if present)
        key0 = f"{base_label},ref=room_pos0"
        if key0 not in self.dataset:
            # fallback: try to find any key starting with base_label
            candidates = [k for k in self.dataset.keys() if k.startswith(base_label)]
            raise KeyError(
                f"Could not find dataset entry '{key0}'. "
                f"Found {len(candidates)} candidates starting with base_label."
            )

        d0 = self.dataset[key0]
        self.mic_array_positions = d0.get("mic_array_position", None)
        self.src_positions = d0.get("src_position", None)

        self.ref_positions = []
        self.ref_signals = []
        self.virtual_signals = []

        # Expect 5 ref positions (room_pos0..room_pos4) but tolerate missing
        for i in range(5):
            k = f"{base_label},ref=room_pos{i}"
            if k not in self.dataset:
                continue
            d = self.dataset[k]
            self.ref_positions.append(d.get("ref_position", None))
            self.ref_signals.append(np.asarray(d["ref"]).reshape(-1))
            self.virtual_signals.append(np.asarray(d["virtual"]).reshape(-1))

        if len(self.ref_signals) == 0:
            raise ValueError("No ref/virtual signals found for this base_label.")

    # ------------------------------------------------------------------
    # Metrics / export
    # ------------------------------------------------------------------
    def metrics_dataframe(self) -> pd.DataFrame:
        rows = []
        for i, (ref, vm) in enumerate(zip(self.ref_signals, self.virtual_signals)):
            snr_db = analysis_utils.compute_snr(ref, vm)
            freq_err_db = analysis_utils.compute_freq_error(ref, vm, self.fs)
            _, phase_err = analysis_utils.compute_phase_error(ref, vm, self.fs, n_fft=1024)
            phase_err_scalar = float(np.mean(np.abs(phase_err)))
            combined = analysis_utils.compute_combined_error_score(snr_db, freq_err_db, phase_err_scalar)

            rows.append({
                "ref_index": i,
                "snr_db": float(np.round(snr_db, 3)),
                "freq_error_db": float(np.round(freq_err_db, 3)),
                "phase_error_rad": float(np.round(phase_err_scalar, 3)),
                "combined_score": float(np.round(combined, 6)),
            })

        return pd.DataFrame(rows)

    def save_metrics_csv(self, csv_path: str = "test_files/csv/vma_test_metrics.csv") -> pd.DataFrame:
        df = self.metrics_dataframe()
        os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None
        df.to_csv(csv_path, index=False)
        print(f"✅ VMA-only metrics saved to: {csv_path}")
        return df

    # ------------------------------------------------------------------
    # Quick plots (optional)
    # ------------------------------------------------------------------
    def plot_overlay(self, ref_index: int = 0, seconds: float = 1.0):
        """Overlay virtual vs reference in time domain for a short window."""
        if ref_index < 0 or ref_index >= len(self.ref_signals):
            raise IndexError(f"ref_index out of range (0..{len(self.ref_signals)-1})")

        ref = self.ref_signals[ref_index]
        vm = self.virtual_signals[ref_index]

        n = int(min(len(ref), len(vm), seconds * self.fs))
        t = np.arange(n) / self.fs

        plt.figure(figsize=(12, 4))
        plt.plot(t, ref[:n], label="Reference mic")
        plt.plot(t, vm[:n], label="Virtual mic estimate", alpha=0.8)
        plt.title(f"VMA overlay (ref_index={ref_index})")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)
