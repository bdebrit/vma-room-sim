# Import required libraries
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import os
import soundfile as sf
from scipy.signal import stft
import pandas as pd
import os
import soundfile as sf
from itertools import product
from tqdm import tqdm
import pickle
from sklearn.metrics import mean_squared_error
from scipy.signal import chirp
from numpy import angle, unwrap


def summarize_rankings(dataset, fs, room_dims, room_positions, csv_path="test_files/csv/ranking_all_permutations.csv", include_columns=None):
    """
    Summarizes and ranks dataset results (room-only mode).
    Assumes 5 reference mic positions per configuration (room_pos0-4).
    """
    all_records = []

    for label, data in dataset.items():
        record = {}
        record["label"] = label

        # Parse config from label string
        parts = [kv.split("=") for kv in label.split(",")]
        for k, v in parts:
            v = v.strip()
            if k in ["radius", "num_mics", "mic_array_order"]:
                try:
                    v = float(v) if "." in v else int(v)
                except ValueError:
                    pass
            record[k] = v

        # Add metrics
        ref = data["ref"]
        virtual = data["virtual"]

        snr = compute_snr(ref, virtual)
        freq_err = compute_freq_error(ref, virtual, fs)
        _,phase_error = compute_phase_error(ref, virtual, fs)
        if isinstance(phase_error, (tuple, list)):
            phase_error = phase_error[0]
        phase_err_scalar = float(np.mean(np.abs(phase_error)))
        combined = compute_combined_error_score(snr, freq_err, phase_err_scalar)

        record["snr_db"] = round(snr, 2)
        record["freq_error_db"] = round(freq_err, 2)
        record["phase_error_rad"] = round(phase_err_scalar, 2)
        record["combined_score"] = round(combined, 3)

        # Identify position tag
        record["ref_mic_position"] = data.get("ref_mic_position", "room_pos?")
        record["mic_array_position"] = data.get("mic_array_position", "room_fixed")

        # Compute Euclidean distance
        try:
            ref_pos = np.array(data["ref_position"])
            array_pos = np.array(room_positions["mic_array_centre"])  # fixed
            dist = np.linalg.norm(np.array(ref_pos) - np.array(array_pos))
            record["euclidean_distance"] = round(dist, 3)
        except Exception as e:
            print(f"⚠️ Could not compute distance for {label}: {e}")
            record["euclidean_distance"] = None

        all_records.append(record)

    summary_df = pd.DataFrame(all_records)

    # Group by configuration (excluding position)
    group_cols = include_columns or [
        'method', 'num_mics', 'geometry', 'mic_array_order', 'audio_channel_format', 'radius', 'euclidean_distance'
    ]

    grouped = summary_df.groupby(group_cols).agg({
        "snr_db": "mean",
        "freq_error_db": "mean",
        "phase_error_rad": "mean",
        "combined_score": "mean",
        "euclidean_distance": "mean"
    }).reset_index()

    grouped["snr_db"] = grouped["snr_db"].round(2)
    grouped["freq_error_db"] = grouped["freq_error_db"].round(3)
    grouped["phase_error_rad"] = grouped["phase_error_rad"].round(3)
    grouped["combined_score"] = grouped["combined_score"].round(3)
    grouped["euclidean_distance"] = grouped["euclidean_distance"].round(3)

    # Rank by best combined score (lower is better)
    grouped = grouped.sort_values(by="combined_score", ascending=True)
    grouped["rank"] = range(1, len(grouped) + 1)

    # Reorder columns to place euclidean_distance after radius
    cols = list(grouped.columns)
    if 'euclidean_distance' in cols:
        radius_idx = cols.index('radius')
        # Remove and re-insert after 'radius'
        cols.remove('euclidean_distance')
        cols.insert(radius_idx + 1, 'euclidean_distance')
        grouped = grouped[cols]

    if csv_path:
        summary_df.to_csv("test_files/csv/summary_per_position.csv", index=False)
        grouped.to_csv(csv_path, index=False)

    return grouped





def _coerce_stereo_to_2xN(stereo_signal: np.ndarray) -> np.ndarray:
    """Return stereo as shape (2, N). Accepts (2, N) or (N, 2)."""
    stereo_signal = np.asarray(stereo_signal)
    if stereo_signal.ndim != 2:
        raise ValueError(f"Expected stereo signal with 2 dims, got shape {stereo_signal.shape}")
    if stereo_signal.shape[0] == 2:
        return stereo_signal
    if stereo_signal.shape[1] == 2:
        return stereo_signal.T
    raise ValueError(f"Expected stereo signal with shape (2, N) or (N, 2), got {stereo_signal.shape}")


def add_stereo_signal(room, stereo_signal, fs, left_pos, right_pos):
    """
    Adds two mono sources (left and right) from a stereo signal at specified positions.

    Accepts stereo_signal as either:
      - shape (2, N)
      - shape (N, 2)
    """
    stereo_2n = _coerce_stereo_to_2xN(stereo_signal)
    room.add_source(position=left_pos, signal=stereo_2n[0], delay=0.0)
    room.add_source(position=right_pos, signal=stereo_2n[1], delay=0.0)

def run_all_room_mic_tests(vms, room_dims, test_signal, dataset_checkpoint_path, use_reflections=True, absorption=0.3, max_order_reflections=5): 
    """
    Run all room-style simulations over method, geometry, radius, num_mics.
    Evaluates each configuration at 5 reference mic positions: 1 center + 4 perimeter.
    """

    fs = 48000
    radius_values = [0.1, 0.25, 0.5]
    method_list = ['mvdr', 'delay_sum_time', 'delay_sum_freq']
    audio_format = "stereo"

    # Logical number of mics per geometry type
    geometry_mic_map = {
        'spherical_uniform': [4, 8, 12, 16, 20],
        'tetrahedral': [4],
        'octahedral': [6],
        'icosahedral': [12]
    }

    # Build combinations: (geometry, num_mics) × method × radius
    geometry_num_mic_combinations = [
        (geom, n) for geom, counts in geometry_mic_map.items() for n in counts
    ]
    all_combinations = list(product(geometry_num_mic_combinations, method_list, radius_values))

    room_positions = vms.get_room_style_positions(room_dims)
    mic_array_centre = room_positions['mic_array_centre']
    source_positions = room_positions['source_positions']
    eval_positions = vms.get_room_evaluation_positions(room_positions['ref_mic_pos'], radius=1.0)

    # Load or generate signal
    if isinstance(test_signal, str) and test_signal.lower().endswith('.wav'):
        signal_data, file_fs = sf.read(test_signal)
        if file_fs != fs:
            raise ValueError(f"Expected {fs} Hz, got {file_fs}")
        signal_stereo = signal_data.T if signal_data.ndim > 1 else np.stack([signal_data, signal_data], axis=0)
    elif isinstance(test_signal, np.ndarray):
        signal_stereo = np.stack([test_signal, test_signal], axis=0)
    else:
        raise TypeError("test_signal must be a WAV file path or a NumPy array")

    # ✅ If checkpoint exists, load it
    if os.path.exists(dataset_checkpoint_path):
        print(f"🔁 Loading checkpoint from {dataset_checkpoint_path}")
        with open(dataset_checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        dataset = checkpoint['dataset']
        fs = checkpoint.get('fs', fs)
        room_dims = checkpoint.get('room_dims', room_dims)

        summary_df = summarize_metrics_table(
            dataset,
            metric_fn=compute_time_domain_error,
            sort_by='combined_score',
            ascending=True,
            round_digits=3,
            csv_path=None
        )

        ranking_df = summarize_rankings(
            dataset, fs, room_dims, room_positions,
            csv_path="test_files/csv/ranking_all_permutations.csv",
            include_columns=['method', 'num_mics', 'geometry', 'mic_array_order', 'audio_channel_format', 'radius']
        )

        return summary_df, ranking_df

    # ❌ No checkpoint — generate full dataset
    dataset = {}
    index_by_config = {}

    for (geom, num_mics), method, mic_radius in tqdm(all_combinations, desc="Running room tests"):
        mic_positions = vms.generate_array_geometry(mic_array_centre, geom, mic_radius, num_mics)
        array_order = vms.estimate_array_order(geom, num_mics)

        base_label = (
            f"geometry={geom},method={method},radius={mic_radius},num_mics={num_mics},"
            f"mic_array_order={array_order},array=room_fixed,audio_channel_format={audio_format}"
        )
        index_by_config[base_label] = []

        for i, ref_position in enumerate(eval_positions):
            max_order = 0 if not use_reflections else max_order_reflections
            room = vms.create_simulation_room(room_dims, fs, absorption=absorption, max_order=max_order)

            add_stereo_signal(room, signal_stereo, fs, source_positions[0], source_positions[1])
            room.add_microphone_array(pra.MicrophoneArray(mic_positions, fs))
            room.add_microphone_array(pra.MicrophoneArray(np.array(ref_position).reshape(3, 1), fs))

            room.compute_rir()
            room.simulate()

            ref_signal = room.mic_array.signals[-1]
            virtual_signal = vms.compute_virtual_microphone(
                room, method, np.array(ref_position), mic_positions
            )[:len(ref_signal)]

            if len(ref_signal) != len(virtual_signal):
                print("⚠️ Signal length mismatch!")
                exit(0)

            label = f"{base_label},ref=room_pos{i}"
            index_by_config[base_label].append(label)

            dataset[label] = {
                'ref': ref_signal,
                'virtual': virtual_signal,
                'audio_channel_format': audio_format,
                'geometry': geom,
                'method': method,
                'radius': mic_radius,
                'num_mics': num_mics,
                'mic_array_order': array_order,
                'ref_position': ref_position,
                'src_position': source_positions,
                'ref_mic_position': f'room_pos{i}',
                'mic_array_position': 'room_fixed'
            }

    # ✅ Analyze and save checkpoint
    summary_df = summarize_metrics_table(
        dataset,
        metric_fn=compute_time_domain_error,
        sort_by='combined_score',
        ascending=True,
        round_digits=3,
        csv_path=None
    )

    ranking_df = summarize_rankings(
        dataset, fs, room_dims, room_positions,
        csv_path="test_files/csv/ranking_all_permutations.csv",
        include_columns=['method', 'num_mics', 'geometry', 'mic_array_order', 'audio_channel_format', 'radius']
    )

    with open(dataset_checkpoint_path, "wb") as f:
        pickle.dump({
            'dataset': dataset,
            'ranking_df': ranking_df,
            'room_dims': room_dims,
            'signal_stereo': signal_stereo,
            'fs': fs,
            'index_by_config': index_by_config
        }, f)

    print(f"✅ Checkpoint saved to {dataset_checkpoint_path}")
    return summary_df, ranking_df

def generate_test_signal(
    signal_type='sine_sweep',
    duration=4.0,
    fs=48000,
    f_start=20,
    f_end=20000,
    stereo=True,
    normalize=True,
    apply_fade=True,
    fade_percent=0.01,
    preview=True
):
    """
    Generate a mono or stereo test signal.

    Args:
        signal_type (str): 'sine', 'sine_sweep', or 'noise'
        duration (float): Duration in seconds
        fs (int): Sampling rate
        f_start (float): Start freq for sweep
        f_end (float): End freq for sweep
        stereo (bool): Duplicate mono to stereo
        normalize (bool): Normalize signal to [-1, 1]
        apply_fade (bool): Apply fade-in/out to avoid clicks
        fade_percent (float): Fraction of signal for fading (e.g., 0.01 = 1%)
        preview (bool): Show waveform

    Returns:
        np.ndarray: signal
        int: sample rate
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    if signal_type == 'sine':
        f = 440
        mono = np.sin(2 * np.pi * f * t)

    elif signal_type == 'sine_sweep_log':
        mono = np.sin(2 * np.pi * f_start * duration / np.log(f_end / f_start) * 
                    (np.exp(t * np.log(f_end / f_start) / duration) - 1))
        
    elif signal_type == 'sine_sweep_linear':
        mono = chirp(t, f0=f_start, f1=f_end, t1=duration, method='linear')

    elif signal_type == 'noise':
        mono = np.random.normal(0, 1, len(t))

    else:
        raise ValueError("Unknown signal_type. Use 'sine', 'sine_sweep', or 'noise'.")

    # Apply fade-in/out
    if apply_fade:
        fade_len = int(len(mono) * fade_percent)
        window = np.ones_like(mono)
        window[:fade_len] = np.linspace(0, 1, fade_len)
        window[-fade_len:] = np.linspace(1, 0, fade_len)
        mono *= window

    # Normalize
    if normalize:
        mono = mono / np.max(np.abs(mono))

    # Duplicate to stereo if needed
    signal = np.stack([mono, mono], axis=0) if stereo else mono  

    print(f"⚙️ Generated {signal_type} signal ({'stereo' if stereo else 'mono'}), duration: {duration}s")

    if preview:
        preview_signal(signal, fs)

    return signal, fs

def load_wav_signal(filepath, fs_expected=48000, normalize=True, preview=False):
    signal, fs = sf.read(filepath)
    if fs != fs_expected:
        raise ValueError(f"Expected sample rate {fs_expected}, got {fs}")

    if normalize:
        signal = signal / np.max(np.abs(signal))

    if signal.ndim == 1:
        signal_out = signal
    else:
        signal_out = signal.T

    print(f"🎧 Loaded WAV: {filepath}, shape: {signal_out.shape}, fs: {fs}")

    if preview:
        preview_signal(signal_out, fs)

    return signal_out, fs
    
def preview_signal(signal, fs):
    plt.figure(figsize=(10, 2))
    if signal.ndim == 1:
        plt.plot(np.arange(len(signal)) / fs, signal, label="Mono")
    else:
        plt.plot(np.arange(signal.shape[1]) / fs, signal[0], label="Left")
        plt.plot(np.arange(signal.shape[1]) / fs, signal[1], label="Right")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Signal Preview")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show(block=False)



def load_signals_checkpoint(path="checkpoint_full_results.pkl"):
    """
    Load simulation dataset, metrics, and metadata from a unified checkpoint.

    Returns:
        dataset (dict): All simulation signals and metadata.
        ranking_df (pd.DataFrame): Performance metrics.
        room_dims (list): Room dimensions used in simulation.
        signal_mono (np.ndarray): Mono test signal.
        signal_stereo (np.ndarray): Stereo test signal (L & R).
    """
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)

    dataset = checkpoint['dataset']
    ranking_df = checkpoint['ranking_df']
    room_dims = checkpoint['room_dims']
    signal_mono = checkpoint['signal_mono']
    signal_stereo = checkpoint['signal_stereo']

    print(f"✅ Loaded checkpoint from {path}")
    return dataset, ranking_df, room_dims, signal_mono, signal_stereo


def compute_phase_error(ref_signal, test_signal, fs, n_fft=1024, hop_length=512, epsilon=1e-10):
    """
    Computes phase error between two signals using STFT.
    
    Args:
        ref_signal: Ground truth reference signal (1D array)
        test_signal: Estimated signal (1D array)
        fs: Sampling rate (Hz)
        n_fft: FFT size for STFT
        hop_length: Hop length for STFT (default: n_fft // 2)
        epsilon: Small value to avoid divide-by-zero or log(0)

    Returns:
        freqs: Frequency bins (Hz)
        avg_phase_error: Mean phase error per bin (1D array)
    """
    if hop_length is None:
        hop_length = n_fft // 2

    f, _, ref_stft = stft(ref_signal, fs=fs, nperseg=n_fft, noverlap=n_fft - hop_length)
    _, _, test_stft = stft(test_signal, fs=fs, nperseg=n_fft, noverlap=n_fft - hop_length)

    # Compute wrapped phase difference
    ref_phase = np.angle(ref_stft)
    test_phase = np.angle(test_stft)
    phase_diff = np.angle(np.exp(1j * (test_phase - ref_phase)))  # wrap into [-π, π]

    avg_phase_error = np.mean(np.abs(phase_diff), axis=1)  # mean across time

    return f, avg_phase_error


def compute_time_domain_error(ref_signal, test_signal):
    """
    Computes time-domain error metrics between a reference and a test signal.
    
    Parameters:
        ref_signal (np.array): Reference signal (e.g. from physical mic)
        test_signal (np.array): Test signal (e.g. virtual mic output)
        
    Returns:
        dict: {
            'mse': Mean Squared Error,
            'snr': Signal-to-Noise Ratio (dB),
            'correlation': Normalized cross-correlation peak
        }
    """
    # Ensure equal length
    min_len = min(len(ref_signal), len(test_signal))
    ref = ref_signal[:min_len]
    test = test_signal[:min_len]

    # MSE
    mse = mean_squared_error(ref, test)

    # SNR
    signal_power = np.mean(ref ** 2)
    noise_power = np.mean((ref - test) ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf

    # Cross-correlation peak (normalized)
    correlation = np.correlate(ref, test, mode='valid')[0]
    norm_corr = correlation / (np.linalg.norm(ref) * np.linalg.norm(test))

    return {'mse': mse, 'snr': snr, 'correlation': norm_corr}

def get_fixed_config_variants(dataset, ranking_df, fixed_fields, min_variants=3):
    """
    Find sets of configurations in ranking_df where fixed_fields are identical, but other factors vary.
    
    Args:
        dataset (dict): Keyed by label, values include metadata (src_position, ref_position, etc.).
        ranking_df (pd.DataFrame): DataFrame containing 'label' and metadata columns.
        fixed_fields (list of str): Fields that must match to group configs together.
        min_variants (int): Minimum number of configs required per group to be considered.
    
    Returns:
        grouped_dfs (list of pd.DataFrame): Each contains rows with the same fixed_fields.
    """
    # Build a new DataFrame from dataset metadata
    records = []
    for label, entry in dataset.items():
        record = {
            'label': label,
            'src_position': entry.get('src_position'),
            'ref_position': entry.get('ref_position'),
            'geometry': entry.get('geometry'),
            'method': entry.get('method'),
            'radius': entry.get('radius'),
            'num_mics': entry.get('num_mics'),
            'mic_array_order': entry.get('mic_array_order'),
            'audio_channel_format': entry.get('audio_channel_format'),
        }
        records.append(record)
    
    meta_df = pd.DataFrame(records)
    merged_df = pd.merge(meta_df, ranking_df, on='label', suffixes=('_meta', '_rank'))

    # Group by fixed fields
    grouped_dfs = []
    grouped = merged_df.groupby(fixed_fields)

    for _, group in grouped:
        if len(group) >= min_variants:
            grouped_dfs.append(group)

    print(f"✅ Found {len(grouped_dfs)} config groups with ≥ {min_variants} variants.")
    return grouped_dfs

def plot_combined_metrics_summary(
    dataset: dict,
    metric_fn=compute_time_domain_error,
    metrics_to_plot=['snr', 'mse', 'correlation'],
    geometry=None,
    method=None,
    group_size=3,
    figsize=(10, 5),
    bar=True
):
    """
    Plots bar charts comparing multiple time-domain metrics across test cases,
    grouped by subsets (e.g., radius variations) to keep plots readable.

    Parameters:
        dataset (dict): Key = test label, Value = (ref_signal, virtual_signal)
        metric_fn (function): Function that returns a dict of metric values
        metrics_to_plot (list): Metric names to include in plot
        geometry (str): Geometry name for the subtitle
        method (str): Beamforming method for the subtitle
        group_size (int): How many configurations to include per figure
        figsize (tuple): Plot size
        bar (bool): If True, use bar plot; else line plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import re

    labels = list(dataset.keys())
    results = [metric_fn(ref, test) for ref, test in dataset.values()]
    data = {metric: [r.get(metric, np.nan) for r in results] for metric in metrics_to_plot}

    total_groups = (len(labels) + group_size - 1) // group_size

    for group_index in range(total_groups):
        start = group_index * group_size
        end = min(start + group_size, len(labels))
        sub_labels = labels[start:end]
        sub_data = {metric: values[start:end] for metric, values in data.items()}

        # ✅ Extract just the radius from each label
        radius_labels = []
        for label in sub_labels:
            match = re.search(r"radius=([\d.]+)", label)
            radius_labels.append(f"r={match.group(1)}" if match else label)

        plt.figure(figsize=figsize)

        if bar:
            width = 0.2
            x = np.arange(len(radius_labels))
            for i, metric in enumerate(metrics_to_plot):
                offset = (i - len(metrics_to_plot)/2) * width
                unit = "(dB)" if metric == "snr" else "(unitless)"
                plt.bar(x + offset, sub_data[metric], width=width, label=f"{metric} {unit}")
            plt.xticks(ticks=x, labels=radius_labels)
        else:
            for metric in metrics_to_plot:
                plt.plot(radius_labels, sub_data[metric], marker='o', label=metric)

        plt.xlabel("Test Configuration (radius)")
        plt.ylabel("Metric Value")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(loc='lower right')

        # 📌 Bold main title, bold subtitle below with spacing
        title_main = "SNR | MSE | Correlation"
        title_sub = f"geometry = {geometry}; method = {method}"
        plt.suptitle(f"{title_main}\n\n{title_sub}", fontsize=14, fontweight='bold', y=0.95)

        plt.tight_layout(rect=[0, 0, 1, 0.88])  # Extra space for 2-line title
        plt.show(block=False)


''' 
def compute_combined_error_score(ref_signal, test_signal, fs=48000, n_fft=1024):
    """
    Computes a combined performance score across multiple error metrics.

    Returns:
        dict: {
            'combined_score': float,
            'snr': float,
            'mse': float,
            'correlation': float,
            'spectral_mag_error': float,
            'spectral_phase_error': float
        }
    """
    from scipy.signal import stft

    # Ensure equal length
    min_len = min(len(ref_signal), len(test_signal))
    ref = ref_signal[:min_len]
    test = test_signal[:min_len]

    # Time-domain metrics
    mse = mean_squared_error(ref, test)
    snr = 10 * np.log10(np.mean(ref ** 2) / (np.mean((ref - test) ** 2) + 1e-10))
    corr = np.correlate(ref, test, mode='valid')[0] / (np.linalg.norm(ref) * np.linalg.norm(test))

    # Spectral metrics
    f, _, Zxx_ref = stft(ref, fs=fs, nperseg=n_fft)
    _, _, Zxx_test = stft(test, fs=fs, nperseg=n_fft)

    # Align STFT output shapes
    min_time_frames = min(Zxx_ref.shape[1], Zxx_test.shape[1])
    Zxx_ref = Zxx_ref[:, :min_time_frames]
    Zxx_test = Zxx_test[:, :min_time_frames]

    mag_ref = np.abs(Zxx_ref)
    mag_test = np.abs(Zxx_test)
    epsilon = 1e-10
    mag_err_db = 20 * np.log10((np.abs(mag_ref - mag_test) + epsilon) / (mag_ref + epsilon))
    mean_mag_error = np.mean(np.abs(mag_err_db))

    phase_ref = np.angle(Zxx_ref)
    phase_test = np.angle(Zxx_test)
    phase_err = np.unwrap(phase_ref - phase_test, axis=0)
    mean_phase_error = np.mean(np.abs(phase_err))

    # Combine into a score (normalize to reasonable scales)
    score = (
        -snr                 # higher SNR = better → negate to minimize
        + mse * 1000         # scale MSE for weighting
        + mean_mag_error     # already in dB
        + mean_phase_error   # radians
        - corr * 10          # encourage higher correlation
    )

    return {
        'combined_score': score,
        'snr': snr,
        'mse': mse,
        'correlation': corr,
        'spectral_mag_error': mean_mag_error,
        'spectral_phase_error': mean_phase_error
    }
'''


def compute_snr(reference, estimate):
    """
    Computes SNR in dB between reference and estimate.
    """
    noise = reference - estimate
    signal_power = np.mean(reference ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return np.inf
    return 10 * np.log10(signal_power / noise_power)

def compute_freq_error(reference, estimate, fs):
    """
    Computes magnitude error between reference and estimated signals in frequency domain (in dB).
    """
    N = len(reference)
    ref_fft = np.abs(np.fft.rfft(reference, n=N))
    est_fft = np.abs(np.fft.rfft(estimate, n=N))

    # Avoid log(0)
    ref_fft = np.maximum(ref_fft, 1e-12)
    est_fft = np.maximum(est_fft, 1e-12)

    error_db = 20 * np.log10(np.abs(ref_fft - est_fft) / ref_fft)
    return np.mean(np.abs(error_db))


def compute_combined_error_score(snr_db, freq_err_db, phase_err_rad):
    """
    Combines SNR (dB), frequency magnitude error (dB), and phase error (radians)
    into a single scalar score. Lower is better.
    """
    snr_db = float(np.atleast_1d(snr_db)[0])
    freq_err_db = float(np.atleast_1d(freq_err_db)[0])
    phase_err_rad = float(np.atleast_1d(phase_err_rad)[0])

    snr_penalty = -snr_db  # Always factor SNR as a negative contribution
    combined = freq_err_db + phase_err_rad + snr_penalty
    return round(combined, 3)



def summarize_metrics_table(
    dataset: dict,
    metric_fn=compute_time_domain_error,
    sort_by='snr',
    ascending=False,
    round_digits=3,
    csv_path=None
) -> pd.DataFrame:
    """
    Computes and summarizes time-domain error metrics across configurations.

    Parameters:
        dataset (dict): 
            Key = label (e.g., 'radius = 0.1'), 
            Value = tuple(ref_signal, virtual_signal)
        metric_fn (function): Function to compute metrics (e.g., compute_time_domain_error)
        sort_by (str): Metric name to sort results by (e.g., 'snr', 'mse')
        ascending (bool): Sort order (True = ascending, False = descending)
        round_digits (int): Rounding for display
        csv_path (str or None): If provided, saves table to this CSV path

    Returns:
        pd.DataFrame: Metrics table with one row per configuration
    """
    rows = []
    for label, entry in dataset.items():
        ref = entry['ref']
        test = entry['virtual']
        metrics = metric_fn(ref, test)
        metrics["Test"] = label
        rows.append(metrics)

    df = pd.DataFrame(rows)
    df = df.set_index("Test")

    # Round for clarity
    df = df.round(round_digits)

    # Sort by chosen metric
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending)

    # Optional CSV export
    if csv_path:
        df.to_csv(csv_path)
        print(f"📄 Saved summary to {csv_path}")

    return df


def save_signals_to_wav(
    dataset: dict,
    fs: int = 48000,
    output_dir: str = "wav_outputs",
    prefix: str = "test",
    normalize: bool = True
):
    """
    Saves reference and virtual mic signals to WAV files for each test configuration.

    Parameters:
        dataset (dict): 
            Key = label (e.g., 'radius = 0.1'), 
            Value = tuple(ref_signal, virtual_signal)
        fs (int): Sampling rate
        output_dir (str): Directory to save WAV files
        prefix (str): Prefix for filenames (e.g., test name or sweep)
        normalize (bool): If True, normalize each signal to [-1, 1]
    """
    os.makedirs(output_dir, exist_ok=True)

    for label, (ref, vm) in dataset.items():
        safe_label = label.replace(" ", "_").replace("=", "").replace(".", "p")

        if normalize:
            ref = ref / np.max(np.abs(ref) + 1e-10)
            vm = vm / np.max(np.abs(vm) + 1e-10)

        ref_path = os.path.join(output_dir, f"test_files/wav_outputs/{prefix}_{safe_label}_ref.wav")
        vm_path = os.path.join(output_dir, f"test_files/wav_outputs/{prefix}_{safe_label}_vm.wav")

        sf.write(ref_path, ref, fs)
        sf.write(vm_path, vm, fs)

        print(f"💾 Saved: {ref_path}, {vm_path}")

def generate_error_spectrogram(
    ref_signal,
    test_signal,
    fs=48000,
    window='hann',
    n_fft=32768,
    hop_length=512,
    mode='magnitude',
    title="Error Spectrogram",
    cmap='inferno',
    vmin=None,
    vmax=None
):
    """
    Generates a 2D error spectrogram between reference and test signal.

    Parameters:
        ref_signal (np.ndarray): Ground-truth signal
        test_signal (np.ndarray): Virtual mic signal
        fs (int): Sampling rate
        window (str): STFT window
        n_fft (int): FFT size
        hop_length (int): Hop size between frames
        mode (str): 'magnitude' or 'phase'
        title (str): Plot title
        cmap (str): Colormap to use
        vmin, vmax (float): Optional manual color scale limits
    """
    # Match signal lengths
    min_len = min(len(ref_signal), len(test_signal))
    ref = ref_signal[:min_len]
    test = test_signal[:min_len]

    # Compute STFTs
    f, t, Zxx_ref = stft(ref, fs=fs, window=window, nperseg=n_fft, noverlap=n_fft - hop_length)
    _, _, Zxx_test = stft(test, fs=fs, window=window, nperseg=n_fft, noverlap=n_fft - hop_length)

    if mode == 'magnitude':
        mag_ref = np.abs(Zxx_ref)
        mag_test = np.abs(Zxx_test)
        epsilon = 1e-10
        error = 20 * np.log10((np.abs(mag_ref - mag_test) + epsilon) / (mag_ref + epsilon))
    elif mode == 'phase':
        phase_ref = np.angle(Zxx_ref)
        phase_test = np.angle(Zxx_test)
        error = np.unwrap(phase_ref - phase_test, axis=0)
    else:
        raise ValueError("Mode must be 'magnitude' or 'phase'.")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(t, f, np.abs(error), shading='gouraud', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Error (dB)" if mode == 'magnitude' else "Error (radians)")
    plt.tight_layout()



def plot_fir_evaluation_overlay(results_by_mode, fs=48000, n_fft=1024, hop_size=512):

    modes = list(results_by_mode.keys())
    colors = plt.cm.tab10.colors
    linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5))]
    hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

    fontdict = {"fontsize": 14, "fontweight": "bold"}
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    ax_mag, ax_phase = axs[0], axs[1]

    snr_values = []

    for idx, mode in enumerate(modes):
        test_obj = results_by_mode[mode]
        ref_input = test_obj.reference_input[:, 0]  # Mono (Left channel)
        recordings = test_obj.recorded_responses

        min_len = min(min(len(r), len(ref_input)) for r in recordings)
        freqs = np.fft.rfftfreq(n_fft, d=1 / fs)

        mag_errors = []
        phase_errors = []
        snrs = []

        for rec in recordings:
            rec = rec[:min_len]
            ref = ref_input[:min_len]

            _, _, Z_ref = stft(ref, fs=fs, nperseg=n_fft, noverlap=hop_size)
            _, _, Z_rec = stft(rec, fs=fs, nperseg=n_fft, noverlap=hop_size)

            mag_ref = np.abs(Z_ref)
            mag_rec = np.abs(Z_rec)
            mag_err = 20 * np.log10((mag_rec + 1e-8) / (mag_ref + 1e-8))
            mag_error = np.mean(np.abs(mag_err), axis=1)
            mag_errors.append(mag_error)

            phase_ref = angle(Z_ref)
            phase_rec = angle(Z_rec)
            phase_diff = unwrap(phase_rec - phase_ref, axis=0)
            phase_error = np.mean(np.abs(np.angle(np.exp(1j * phase_diff))), axis=1)
            phase_errors.append(phase_error)

            signal_power = np.mean(ref ** 2)
            noise_power = np.mean((ref - rec) ** 2) + 1e-10
            snr = 10 * np.log10(signal_power / noise_power)
            snrs.append(snr)

        mag_avg = np.mean(np.array(mag_errors), axis=0)
        phase_avg = np.mean(np.array(phase_errors), axis=0)
        snr_avg = np.mean(snrs)

        linestyle = linestyles[idx % len(linestyles)]
        color = colors[idx % len(colors)]

        ax_mag.plot(freqs, mag_avg, label=mode, color=color, linestyle=linestyle)
        ax_phase.plot(freqs, phase_avg, label=mode, color=color, linestyle=linestyle)
        snr_values.append(snr_avg)

    ax_mag.set_xscale("log")
    ax_mag.set_title("Magnitude Error (dB)", fontdict=fontdict)
    ax_mag.set_xlabel("Frequency (Hz)", fontdict=fontdict)
    ax_mag.set_ylabel("Error (dB)", fontdict=fontdict)
    ax_mag.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_mag.legend()

    ax_phase.set_xscale("log")
    ax_phase.set_title("Phase Error (radians)", fontdict=fontdict)
    ax_phase.set_xlabel("Frequency (Hz)", fontdict=fontdict)
    ax_phase.set_ylabel("Error (rad)", fontdict=fontdict)
    ax_phase.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_phase.legend()

    plt.tight_layout()
    plt.savefig("graphs/fir_results.png", dpi=600, bbox_inches='tight')



def compute_linear_mag_error_spectrum(reference, estimate, fs, fft_size=32768):
    """
    Computes linear magnitude error spectrum (no log scaling).

    Parameters:
        reference (np.ndarray): Reference signal.
        estimate (np.ndarray): Estimated signal.
        fs (int): Sampling rate (Hz).
        fft_size (int): FFT size for FFT computation.

    Returns:
        freqs (np.ndarray): Frequency bins (Hz).
        mag_error (np.ndarray): Absolute magnitude difference per bin.
    """
    # Pad or truncate to desired FFT size
    ref = np.pad(reference, (0, max(0, fft_size - len(reference))), mode='constant')[:fft_size]
    est = np.pad(estimate, (0, max(0, fft_size - len(estimate))), mode='constant')[:fft_size]

    # FFT magnitude spectra
    ref_mag = np.abs(np.fft.rfft(ref, n=fft_size))
    est_mag = np.abs(np.fft.rfft(est, n=fft_size))
    freqs = np.fft.rfftfreq(fft_size, 1 / fs)

    # Absolute magnitude error
    error = np.abs(ref_mag - est_mag)

    return freqs, error


def compute_freq_error_spectrum(reference, estimate, fs, fft_size=32768, epsilon=1e-12):
    """
    Computes log-magnitude error (in dB) between reference and estimated signal FFTs.
    Avoids division by small values to prevent artificial spikes.
    """
    # Zero-pad or truncate to fixed FFT size
    ref = np.pad(reference, (0, max(0, fft_size - len(reference))), mode='constant')[:fft_size]
    est = np.pad(estimate, (0, max(0, fft_size - len(estimate))), mode='constant')[:fft_size]

    # Frequency axis
    freqs = np.fft.rfftfreq(fft_size, 1 / fs)

    # FFT magnitude (use dB)
    ref_fft = np.abs(np.fft.rfft(ref, n=fft_size))
    est_fft = np.abs(np.fft.rfft(est, n=fft_size))

    # Log-magnitude error in dB
    ref_db = 20 * np.log10(ref_fft )
    est_db = 20 * np.log10(est_fft )
    error_db = np.abs(ref_db - est_db)

    return freqs, error_db

import numpy as np
import matplotlib.pyplot as plt

def analyze_spectral_error(ref, vm, fs, fft_size=32768, epsilon=1e-12):
    """
    Computes and plots:
    - Log-magnitude error (dB)
    - Linear magnitude error
    - Phase error (radians)
    between a reference and virtual mic signal.
    """
    # Pad or truncate to FFT size
    ref = np.pad(ref, (0, max(0, fft_size - len(ref))), mode='constant')[:fft_size]
    vm = np.pad(vm, (0, max(0, fft_size - len(vm))), mode='constant')[:fft_size]

    # FFTs
    ref_fft = np.fft.rfft(ref, n=fft_size)
    vm_fft = np.fft.rfft(vm, n=fft_size)
    freqs = np.fft.rfftfreq(fft_size, d=1/fs)

    # Magnitudes and Phases
    ref_mag = np.abs(ref_fft)
    vm_mag = np.abs(vm_fft)
    ref_phase = np.angle(ref_fft)
    vm_phase = np.angle(vm_fft)

    # Errors
    log_mag_error_db = np.abs(20 * np.log10(ref_mag + epsilon) - 20 * np.log10(vm_mag + epsilon))
    lin_mag_error = np.abs(ref_mag - vm_mag)
    phase_error_rad = np.abs(np.angle(np.exp(1j * (ref_phase - vm_phase))))  # wrapped abs diff

    # Limit to 20 Hz – 20 kHz
    valid = (freqs > 20) & (freqs < 20000)
    freqs = freqs[valid]
    log_mag_error_db = log_mag_error_db[valid]
    lin_mag_error = lin_mag_error[valid]
    phase_error_rad = phase_error_rad[valid]

    # Plotting
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(freqs, log_mag_error_db, label="Log-Mag Error (dB)", color='tab:blue')
    plt.ylabel("dB")
    plt.xscale("log")
    plt.title("Log-Magnitude Error (dB)")
    plt.grid(True, which="both")

    plt.subplot(3, 1, 2)
    plt.plot(freqs, lin_mag_error, label="Linear Magnitude Error", color='tab:green')
    plt.ylabel("Linear Mag Error")
    plt.xscale("log")
    plt.title("Linear Magnitude Error")
    plt.grid(True, which="both")

    plt.subplot(3, 1, 3)
    plt.plot(freqs, phase_error_rad, label="Phase Error (rad)", color='tab:red')
    plt.ylabel("Phase Error (rad)")
    plt.xlabel("Frequency (Hz)")
    plt.xscale("log")
    plt.title("Phase Error (radians)")
    plt.grid(True, which="both")

    plt.tight_layout()

    return freqs, log_mag_error_db, lin_mag_error, phase_error_rad


def plot_virtual_mic_evaluation_from_csv(dataset, ranking_df, fs):
    """
    Plots averaged virtual mic performance across:
    - Frequency Error 
    - Phase Error
    - SNR (from CSV)
    """
    plt.rcParams['font.family'] = 'monospace'

    geometries = sorted(ranking_df["geometry"].unique())
    colors = plt.cm.tab10.colors
    linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5))]
    hatches = ['/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']
    fontdict = {"fontsize": 14, "fontweight": "bold"}
    avg_results = {}

    for geom in geometries:
        top_row = ranking_df[ranking_df["geometry"] == geom].nsmallest(1, 'snr_db').iloc[0]

        label_prefix = (
            f"geometry={top_row.geometry},method={top_row.method},radius={top_row.radius},"
            f"num_mics={top_row.num_mics},mic_array_order={top_row.mic_array_order},"
            f"array=room_fixed,audio_channel_format={top_row.audio_channel_format}"
        )
        label_for_plot = f"{geom:<20} ({top_row.num_mics:>2} mics)"

        freq_errors = []
        phase_errors = []
        freqs_mag = None

        for i in range(5):
            label = f"{label_prefix},ref=room_pos{i}"
            entry = dataset.get(label)
            if entry is None:
                continue

            ref = entry["ref"]
            vm = entry["virtual"]

            freqs, freq_error = compute_freq_error_spectrum(ref, vm, fs)
            _, phase_error = compute_phase_error(ref, vm, fs, n_fft=32768)

            if freqs_mag is None:
                freqs_mag = freqs

            freq_errors.append(np.abs(freq_error))
            phase_errors.append(np.abs(phase_error))

        if freq_errors:
            avg_results[label_for_plot] = {
                "freqs": freqs_mag,
                "avg_freq_error": np.mean(np.stack(freq_errors), axis=0),
                "avg_phase_error": np.mean(np.stack(phase_errors), axis=0),
                "snr_db": top_row["snr_db"]
            }

    keys = sorted(avg_results.keys())
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))

    for idx, label in enumerate(keys):
        color = colors[idx % len(colors)]
        linestyle = linestyles[idx % len(linestyles)]
        data = avg_results[label]

        axs[0].plot(data["freqs"], data["avg_freq_error"],
                    label=label, color=color, linestyle=linestyle)
        axs[1].plot(data["freqs"], data["avg_phase_error"],
                    label=label, color=color, linestyle=linestyle)

    axs[0].set_title("Avg Magnitude Error", fontdict=fontdict)
    axs[0].set_xscale("log")
    axs[0].set_xlabel("Frequency (Hz)", fontdict=fontdict)
    axs[0].set_ylabel("Relative Error (dB)", fontdict=fontdict)
    axs[0].grid(True, which='both', linestyle='--')
    axs[0].legend(title="Geometry (Mics)")

    axs[1].set_title("Avg Phase Error (radians)", fontdict=fontdict)
    axs[1].set_xscale("log")
    axs[1].set_xlabel("Frequency (Hz)", fontdict=fontdict)
    axs[1].set_ylabel("Phase Error (rad)", fontdict=fontdict)
    axs[1].grid(True, which='both', linestyle='--')
    axs[1].legend(title="Geometry (Mics)")

    plt.tight_layout()
    plt.savefig("graphs/virtual_mic_results.png", dpi=600, bbox_inches='tight')




import matplotlib.pyplot as plt

def plot_fft_error_db(freqs, error_db, label="FFT Error", color="blue"):
    """
    Plots the magnitude error between FFTs in dB across frequency.
    
    Parameters:
        freqs (array): Frequency axis in Hz.
        error_db (array): Error in dB at each frequency.
        label (str): Legend label.
        color (str): Line color.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, error_db, label=label, color=color)
    plt.xscale("log")
    plt.xlabel("Frequency (Hz)", fontsize=12, fontweight="bold")
    plt.ylabel("Magnitude Error (dB)", fontsize=12, fontweight="bold")
    plt.title("Frequency-wise Magnitude Error (dB)", fontsize=14, fontweight="bold")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.axhline(1, color="gray", linestyle="--", linewidth=0.8, label="1 dB Error Threshold")
    plt.legend()
    plt.tight_layout()



def plot_time_signal(signal, fs, title="Time Domain Signal"):
    time = np.arange(len(signal)) / fs
    plt.figure(figsize=(10, 4))
    plt.plot(time, signal)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    
def plot_fft(signal, fs, title):
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1/fs)
    mag = 20 * np.log10(np.abs(fft) + 1e-12)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, mag)
    plt.xscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title(title)
    plt.grid(True, which='both')
    plt.tight_layout()

    return freqs, mag

def align_signals(ref, vm):
    from scipy.signal import correlate
    corr = correlate(ref, vm, mode='full')
    lag = np.argmax(corr) - len(vm) + 1
    if lag > 0:
        vm = np.pad(vm, (lag, 0), mode='constant')[:len(ref)]
    else:
        ref = np.pad(ref, (-lag, 0), mode='constant')[:len(vm)]
    return ref, vm

def save_signal_to_wav(filename, signal, fs=48000):
    """
    Save a time-domain signal to a WAV file.

    Args:
        filename (str): Path to output WAV file.
        signal (np.ndarray): The audio signal (float32 or float64).
        fs (int): Sampling rate in Hz (default: 48000).
    """
    from scipy.io.wavfile import write
    # Normalize to int16 range to prevent clipping
    if signal.dtype != np.int16:
        signal = signal / np.max(np.abs(signal))  # normalize to -1.0 to 1.0
        signal = (signal * 32767).astype(np.int16)

    if signal.ndim == 2 and signal.shape[0] == 2:
        signal = signal.T
    
    write(filename, fs, signal)

# ============================================================
# Phase B: validate top-K Phase A random spherical (4) layouts
# ============================================================

def generate_random_spherical_geometry(center, radius, num_mics=4, seed=0):
    """Generate `num_mics` random points uniformly on a sphere of given radius.

    Returns array shape (3, N), consistent with pyroomacoustics MicrophoneArray.
    """
    rng = np.random.default_rng(int(seed))
    u = rng.random(num_mics)
    v = rng.random(num_mics)
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    pts = np.vstack([x, y, z]) * float(radius)
    center = np.asarray(center).reshape(3, 1)
    return pts + center


def compute_geometry_quality_metrics(mic_positions_3xN):
    """Simple geometry metrics for analysis/plots."""
    mp = np.asarray(mic_positions_3xN)
    if mp.shape[0] != 3:
        mp = mp.T
    N = mp.shape[1]

    dists = []
    for i in range(N):
        for j in range(i + 1, N):
            dists.append(np.linalg.norm(mp[:, i] - mp[:, j]))
    dists = np.array(dists) if len(dists) else np.array([0.0])

    X = mp - np.mean(mp, axis=1, keepdims=True)
    C = X @ X.T / max(N, 1)
    evals = np.linalg.eigvalsh(C)
    evals = np.sort(np.maximum(evals, 1e-12))
    spread_eig_ratio = float(evals[-1] / evals[0])

    z_range = float(np.max(mp[2, :]) - np.min(mp[2, :]))

    return {
        "min_pairwise_dist": float(np.min(dists)),
        "mean_pairwise_dist": float(np.mean(dists)),
        "spread_eig_ratio": spread_eig_ratio,
        "z_range": z_range,
    }


def select_top_draws_per_method_from_summary_csv(
    phase_a_summary_csv_path,
    top_k=10,
    sort_col="combined_score",
    geometry_filter=("spherical_uniform", "spherical_random", "spherical"),
    num_mics_filter=4,
):
    """Select top-K draw IDs per method from Phase A summary CSV."""
    df = pd.read_csv(phase_a_summary_csv_path)

    if "draw" not in df.columns:
        print("⚠️ Phase A summary CSV has no 'draw' column; cannot select random draws.")
        return {}

    if geometry_filter is not None and "geometry" in df.columns:
        df = df[df["geometry"].isin(list(geometry_filter))]

    if num_mics_filter is not None and "num_mics" in df.columns:
        df = df[df["num_mics"] == num_mics_filter]

    if sort_col not in df.columns:
        raise KeyError(f"Expected '{sort_col}' column in Phase A CSV: {phase_a_summary_csv_path}")

    top_by_method = {}
    for method, g in df.groupby("method"):
        gg = g.groupby("draw")[sort_col].mean().reset_index().sort_values(sort_col, ascending=True)
        top_by_method[method] = [int(x) for x in gg.head(int(top_k))["draw"].tolist()]
    return top_by_method


def run_phase_b_validation(
    vms,
    room_dims,
    test_signals,
    phase_a_summary_csv_path,
    dataset_checkpoint_path="test_files/bin/checkpoint_phase_b.pkl",
    top_k=10,
    seed=0,
    radius=0.25,
    method_list=None,
    use_reflections=False,
    absorption=0.3,
    max_order_reflections=5,
    eval_radius=1.0,
    include_uniform_4=True,
    mic_jitter_std_m=0.0,
    resume_if_checkpoint_exists=True,
):
    """Phase B: validate Phase A top-K random spherical(4) layouts vs baselines.

    - Uses Phase A summary CSV to choose top-K `draw` IDs per method.
    - For each signal + method, evaluates:
        tetrahedral(4), (optional) spherical_uniform(4), and spherical_random(4) for each selected draw.
    - Evaluates at 5 ref positions (centre + ring) from `get_room_evaluation_positions`.
    - Saves checkpoint FIRST, then writes CSV outputs (so a plotting/CSV crash won't lose the run).
    - If `resume_if_checkpoint_exists` is True and the checkpoint exists, it will skip simulation and only postprocess.
    """
    fs = 48000
    if method_list is None:
        method_list = ['mvdr', 'delay_sum_time', 'delay_sum_freq']

    # If we already have a checkpoint, don't rerun sims; just postprocess
    if resume_if_checkpoint_exists and os.path.exists(dataset_checkpoint_path):
        print(f"🔁 Phase B: found existing checkpoint at {dataset_checkpoint_path} — postprocessing only.")
        room_positions = vms.get_room_style_positions(room_dims)
        include_cols = [
            "signal_name","method","num_mics","geometry","mic_array_order","audio_channel_format","radius","draw",
            "min_pairwise_dist","mean_pairwise_dist","spread_eig_ratio","z_range"
        ]
        postprocess_phase_b_checkpoint(
            dataset_checkpoint_path,
            fs=fs,
            room_dims=room_dims,
            room_positions=room_positions,
            out_dir="test_files/csv/phase_b",
            include_columns=include_cols,
        )
        return {"skipped_simulation": True}

    # Normalize signals to (2, N)
    signals_2n = {name: _coerce_stereo_to_2xN(sig) for name, sig in test_signals.items()}

    room_positions = vms.get_room_style_positions(room_dims)
    mic_array_centre = room_positions['mic_array_centre']
    source_positions = room_positions['source_positions']
    eval_positions = vms.get_room_evaluation_positions(room_positions['ref_mic_pos'], radius=eval_radius)

    top_draws_by_method = select_top_draws_per_method_from_summary_csv(
        phase_a_summary_csv_path,
        top_k=top_k,
        sort_col="combined_score",
        geometry_filter=("spherical_uniform", "spherical_random", "spherical"),
        num_mics_filter=4
    )

    dataset = {}
    meta = {
        "fs": fs,
        "room_dims": room_dims,
        "room_positions": room_positions,
        "phase": "B",
        "use_reflections": bool(use_reflections),
        "absorption": float(absorption),
        "eval_radius": float(eval_radius),
        "radius": float(radius),
        "top_k": int(top_k),
        "method_list": list(method_list),
        "signals": list(signals_2n.keys()),
    }

    def build_configs_for_method(method):
        configs = [("tetrahedral", 4, None)]
        if include_uniform_4:
            configs.append(("spherical_uniform", 4, None))
        for draw_id in top_draws_by_method.get(method, []):
            configs.append(("spherical_random", 4, int(draw_id)))

        # Deduplicate
        uniq = []
        seen = set()
        for c in configs:
            if c not in seen:
                uniq.append(c)
                seen.add(c)
        return uniq

    # Run simulations
    for signal_name, stereo_2n in signals_2n.items():
        for method in method_list:
            configs = build_configs_for_method(method)
            for geom, num_mics, draw_id in tqdm(configs, desc=f"Phase B ({signal_name}) configs"):
                # Mic geometry
                if geom == "spherical_random":
                    mic_positions = generate_random_spherical_geometry(
                        mic_array_centre, radius, num_mics=num_mics, seed=seed + int(draw_id)
                    )
                else:
                    mic_positions = vms.generate_array_geometry(mic_array_centre, geom, radius, num_mics)

                # Optional jitter
                if mic_jitter_std_m and mic_jitter_std_m > 0:
                    rng = np.random.default_rng(seed + (int(draw_id) if draw_id is not None else 0))
                    mic_positions = mic_positions + rng.normal(0.0, mic_jitter_std_m, size=mic_positions.shape)

                array_order = vms.estimate_array_order(geom, num_mics) if hasattr(vms, "estimate_array_order") else None
                geom_metrics = compute_geometry_quality_metrics(mic_positions)

                for i, ref_position in enumerate(eval_positions):
                    base_label = (
                        f"signal_name={signal_name},geometry={geom},method={method},radius={radius},num_mics={num_mics},"
                        f"mic_array_order={array_order},array=room_fixed,audio_channel_format=stereo"
                    )
                    if draw_id is not None:
                        base_label += f",draw={int(draw_id)}"

                    label = base_label + f",ref_mic_position=room_pos{i}"

                    max_order = 0 if not use_reflections else max_order_reflections
                    room = vms.create_simulation_room(room_dims, fs, absorption=absorption, max_order=max_order)

                    add_stereo_signal(room, stereo_2n, fs, source_positions[0], source_positions[1])
                    room.add_microphone_array(pra.MicrophoneArray(mic_positions, fs))
                    room.add_microphone_array(pra.MicrophoneArray(np.array(ref_position).reshape(3, 1), fs))

                    room.compute_rir()
                    room.simulate()

                    ref_signal = room.mic_array.signals[-1]
                    virtual_signal = vms.compute_virtual_microphone(room, method, np.array(ref_position), mic_positions)

                    dataset[label] = {
                        "ref": ref_signal,
                        "virtual": virtual_signal,
                        "audio_channel_format": "stereo",
                        "geometry": geom,
                        "method": method,
                        "radius": float(radius),
                        "num_mics": int(num_mics),
                        "mic_array_order": array_order,
                        "ref_position": ref_position,
                        "src_position": source_positions,
                        "ref_mic_position": f"room_pos{i}",
                        "mic_array_position": "room_fixed",
                        "signal_name": signal_name,
                        "draw": int(draw_id) if draw_id is not None else -1,
                        **geom_metrics,
                    }

                # Save incrementally (protect long runs)
                if len(dataset) and (len(dataset) % 25 == 0):
                    ensure_parent_dir(dataset_checkpoint_path)
                    with open(dataset_checkpoint_path, "wb") as f:
                        pickle.dump({"dataset": dataset, "meta": meta}, f)

    # Save checkpoint BEFORE post-processing
    ensure_parent_dir(dataset_checkpoint_path)
    with open(dataset_checkpoint_path, "wb") as f:
        pickle.dump({"dataset": dataset, "meta": meta}, f)
    print(f"✅ Phase B checkpoint saved to {dataset_checkpoint_path}")

    include_cols = [
        "signal_name","method","num_mics","geometry","mic_array_order","audio_channel_format","radius","draw",
        "min_pairwise_dist","mean_pairwise_dist","spread_eig_ratio","z_range"
    ]
    postprocess_phase_b_checkpoint(
        dataset_checkpoint_path,
        fs=fs,
        room_dims=room_dims,
        room_positions=room_positions,
        out_dir="test_files/csv/phase_b",
        include_columns=include_cols,
    )

    return {"skipped_simulation": False}



# ============================================================
# Utilities (missing in some versions)
# ============================================================

def ensure_parent_dir(path: str) -> None:
    """Ensure the parent directory of a file path exists."""
    if path is None:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def postprocess_phase_b_checkpoint(
    checkpoint_path: str,
    fs: int,
    room_dims,
    room_positions: dict,
    out_dir: str = "test_files/csv/phase_b",
    include_columns=None,
):
    """Load Phase B checkpoint and write summary + ranking CSVs.

    Expected checkpoint format:
        {"dataset": dict, "meta": dict}

    Outputs:
        - {out_dir}/summary_per_position_phase_b.csv
        - {out_dir}/ranking_phase_b.csv

    Returns:
        ranking_df (pd.DataFrame)
    """
    ensure_parent_dir(os.path.join(out_dir, "dummy.txt"))
    os.makedirs(out_dir, exist_ok=True)

    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)

    dataset = ckpt.get("dataset", ckpt)  # tolerate older formats

    records = []
    mac = np.array(room_positions.get("mic_array_centre", [0, 0, 0]), dtype=float)

    for label, data in tqdm(dataset.items(), total=len(dataset), desc="Phase B postprocess", unit="entry"):
        ref = data["ref"]
        vm = data["virtual"]

        snr_db = compute_snr(ref, vm)
        freq_err_db = compute_freq_error(ref, vm, fs)
        _, phase_err = compute_phase_error(ref, vm, fs, n_fft=1024)

        phase_err_scalar = float(np.mean(np.abs(phase_err)))
        combined = compute_combined_error_score(snr_db, freq_err_db, phase_err_scalar)

        ref_pos = np.array(data.get("ref_position", [np.nan, np.nan, np.nan]), dtype=float)
        if ref_pos.shape == (3,) and np.all(np.isfinite(ref_pos)):
            euclid = float(np.linalg.norm(ref_pos - mac))
        else:
            euclid = np.nan

        rec = {
            "label": label,
            "signal_name": data.get("signal_name", "unknown"),
            "method": data.get("method", "unknown"),
            "geometry": data.get("geometry", "unknown"),
            "num_mics": data.get("num_mics", np.nan),
            "radius": data.get("radius", np.nan),
            "mic_array_order": data.get("mic_array_order", np.nan),
            "audio_channel_format": data.get("audio_channel_format", "stereo"),
            "draw": data.get("draw", data.get("draw_id", -1)),
            "ref_mic_position": data.get("ref_mic_position", "room_pos?"),
            "mic_array_position": data.get("mic_array_position", "room_fixed"),
            "euclidean_distance": euclid,
            "min_pairwise_dist": data.get("min_pairwise_dist", np.nan),
            "mean_pairwise_dist": data.get("mean_pairwise_dist", np.nan),
            "spread_eig_ratio": data.get("spread_eig_ratio", np.nan),
            "z_range": data.get("z_range", np.nan),
            "snr_db": float(np.round(snr_db, 3)),
            "freq_error_db": float(np.round(freq_err_db, 3)),
            "phase_error_rad": float(np.round(phase_err_scalar, 3)),
            "combined_score": float(np.round(combined, 6)),
        }
        records.append(rec)


    summary_df = pd.DataFrame(records)

    # Save per-position summary
    summary_csv = os.path.join(out_dir, "summary_per_position_phase_b.csv")
    summary_df.to_csv(summary_csv, index=False)

    # Group/rank across the 5 reference positions
    if include_columns is None:
        group_cols = [
            "signal_name", "method", "geometry", "num_mics", "radius", "mic_array_order",
            "audio_channel_format", "draw",
            "min_pairwise_dist", "mean_pairwise_dist", "spread_eig_ratio", "z_range",
        ]
    else:
        group_cols = [c for c in include_columns if c in summary_df.columns]
        if "signal_name" in summary_df.columns and "signal_name" not in group_cols:
            group_cols.insert(0, "signal_name")

    # Avoid pandas collision: don't group by a column you also aggregate
    if "euclidean_distance" in group_cols:
        group_cols = [c for c in group_cols if c != "euclidean_distance"]


    agg = summary_df.groupby(group_cols, dropna=False).agg(
        snr_db=("snr_db", "mean"),
        freq_error_db=("freq_error_db", "mean"),
        phase_error_rad=("phase_error_rad", "mean"),
        combined_score=("combined_score", "mean"),
        euclidean_distance=("euclidean_distance", "mean"),
    ).reset_index()

    agg = agg.sort_values("combined_score", ascending=True).reset_index(drop=True)
    agg["rank"] = np.arange(1, len(agg) + 1)

    for c in ["snr_db", "freq_error_db", "phase_error_rad", "combined_score", "euclidean_distance"]:
        if c in agg.columns:
            agg[c] = agg[c].astype(float).round(3)

    ranking_csv = os.path.join(out_dir, "ranking_phase_b.csv")
    agg.to_csv(ranking_csv, index=False)

    print(f"✅ Phase B postprocess wrote:\n  - {summary_csv}\n  - {ranking_csv}")
    return agg


# ============================================================
# Phase B plotting helpers
# ============================================================

def plot_best_freq_phase_errors_by_geometry(
    ranking_df: pd.DataFrame,
    out_dir: str = "graphs/phase_b",
    by_method: bool = True,
    show: bool = True,
    save_png: bool = True,
):
    """Plot best (lowest) frequency and phase errors per geometry.

    Uses the *ranked* Phase B table (e.g., ranking_phase_b.csv already loaded to ranking_df)
    that contains mean-over-positions metrics such as:
      - freq_error_db
      - phase_error_rad

    If by_method=True, produces one figure per method for freq and phase.
    If by_method=False, produces one figure (aggregated across methods).
    """
    if ranking_df is None or len(ranking_df) == 0:
        print("⚠️ plot_best_freq_phase_errors_by_geometry: ranking_df is empty.")
        return

    required_cols = {"geometry", "freq_error_db", "phase_error_rad"}
    missing = required_cols - set(ranking_df.columns)
    if missing:
        raise ValueError(f"ranking_df missing required columns: {sorted(missing)}")

    os.makedirs(out_dir, exist_ok=True)

    def _label_row(r):
        g = str(r.get("geometry", "unknown"))
        d = r.get("draw", None)
        # Draw might be float in CSV
        try:
            d_int = int(float(d))
        except Exception:
            d_int = None
        if d_int is None or d_int < 0:
            return f"{g} (baseline)"
        return f"{g} (draw={d_int})"

    def _plot_one(sub: pd.DataFrame, metric: str, title: str, fname: str):
        s = sub.sort_values(metric, ascending=True)
        xlabels = [_label_row(r) for _, r in s.iterrows()]
        y = s[metric].astype(float).values

        plt.figure(figsize=(11, 4))
        plt.bar(range(len(y)), y)
        plt.xticks(range(len(y)), xlabels, rotation=15, ha="right")
        plt.ylabel(metric)
        plt.title(title)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()

        if save_png:
            plt.savefig(os.path.join(out_dir, fname), dpi=200, bbox_inches="tight")
        if show:
            plt.show(block=False)
        else:
            plt.close()

    if by_method and "method" in ranking_df.columns:
        for method in sorted(ranking_df["method"].dropna().unique()):
            dfm = ranking_df[ranking_df["method"] == method].copy()

            # Best per geometry for each metric
            best_freq = dfm.sort_values("freq_error_db", ascending=True).groupby("geometry", as_index=False).first()
            best_phase = dfm.sort_values("phase_error_rad", ascending=True).groupby("geometry", as_index=False).first()

            _plot_one(
                best_freq,
                "freq_error_db",
                title=f"Phase B: Best (lowest) frequency magnitude error by geometry | method={method}",
                fname=f"phase_b_best_freq_error__{method}.png",
            )
            _plot_one(
                best_phase,
                "phase_error_rad",
                title=f"Phase B: Best (lowest) phase error by geometry | method={method}",
                fname=f"phase_b_best_phase_error__{method}.png",
            )
    else:
        best_freq = ranking_df.sort_values("freq_error_db", ascending=True).groupby("geometry", as_index=False).first()
        best_phase = ranking_df.sort_values("phase_error_rad", ascending=True).groupby("geometry", as_index=False).first()

        _plot_one(
            best_freq,
            "freq_error_db",
            title="Phase B: Best (lowest) frequency magnitude error by geometry (all methods)",
            fname="phase_b_best_freq_error__all_methods.png",
        )
        _plot_one(
            best_phase,
            "phase_error_rad",
            title="Phase B: Best (lowest) phase error by geometry (all methods)",
            fname="phase_b_best_phase_error__all_methods.png",
        )

    print(f"✅ Saved Phase B best-metric plots to: {out_dir}")

def plot_phase_b_best_error_spectra(
    checkpoint_path: str,
    ranking_csv_path: str,
    out_dir: str = "graphs/phase_b_spectra",
    fs: int = 48000,
    by_method: bool = True,
    ref_positions=("room_pos0", "room_pos1", "room_pos2", "room_pos3", "room_pos4"),
    f_min: float = 20.0,
    f_max: float = 20000.0,
    fft_size: int = 32768,
    n_fft_phase: int = 1024,
    show: bool = True,
    save_png: bool = True,
):
    """
    Plot Phase B spectrum error curves for the BEST config per geometry (optionally per method).

    Requires:
      - Phase B checkpoint (.pkl) containing raw 'ref' and 'virtual' signals per config/position
      - Phase B ranking CSV (ranking_phase_b.csv) to decide which configs are "best"

    Produces:
      - magnitude error spectrum (dB) averaged across 5 ref positions
      - phase error spectrum (rad) averaged across 5 ref positions
    """

    os.makedirs(out_dir, exist_ok=True)

    rank = pd.read_csv(ranking_csv_path)

    with open(checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)
    dataset = ckpt.get("dataset", ckpt)

    # ---- helper: try to find a checkpoint entry robustly ----
    def _match_draw(d_draw, csv_draw):
        # tolerate float/int/str inconsistencies
        try:
            return int(float(d_draw)) == int(float(csv_draw))
        except Exception:
            return str(d_draw) == str(csv_draw)

    def find_entry(method, geometry, draw, ref_pos_name):
        # Preferred: field-based matching
        for d in dataset.values():
            if d.get("method") != method:
                continue
            if d.get("geometry") != geometry:
                continue

            d_draw = d.get("draw", d.get("draw_id", None))
            if d_draw is not None and not _match_draw(d_draw, draw):
                continue

            # ref position naming can differ between pipelines
            ref_name = d.get("ref_mic_position", None)
            if ref_name not in (ref_pos_name, f"ref={ref_pos_name}"):
                continue

            if "ref" in d and "virtual" in d:
                return d

        # Fallback: key-string matching
        for label, d in dataset.items():
            if (f"method={method}" in str(label) and f"geometry={geometry}" in str(label)
                and f"ref={ref_pos_name}" in str(label) and ("ref" in d and "virtual" in d)):
                # draw might be missing for baselines; try to match if present
                if f"draw={draw}" in str(label) or ("draw" not in str(label) and pd.isna(draw)):
                    return d

        raise KeyError(
            f"Could not find checkpoint entry for method={method}, geometry={geometry}, draw={draw}, ref={ref_pos_name}"
        )

    # Decide how we pick “best”
    group_cols = ["geometry"] + (["method"] if by_method else [])
    best_rows = (
        rank.sort_values("combined_score", ascending=True)
            .groupby(group_cols, as_index=False)
            .first()
    )

    print("\nBest rows chosen for spectrum plots:")
    print(best_rows[group_cols + ["draw", "combined_score", "freq_error_db", "phase_error_rad"]])

    # ---- plot each best config ----
    for _, row in best_rows.iterrows():
        method = row["method"] if by_method else rank["method"].iloc[0]
        geometry = row["geometry"]
        draw = row.get("draw", np.nan)

        mag_err_list = []
        phase_err_list = []
        freqs_mag = None
        freqs_phase = None

        for rp in ref_positions:
            d = find_entry(method, geometry, draw, rp)

            ref = np.asarray(d["ref"]).reshape(-1)
            vm = np.asarray(d["virtual"]).reshape(-1)

            freqs_mag, mag_err = compute_freq_error_spectrum(ref, vm, fs, fft_size=fft_size)
            freqs_phase, phase_err = compute_phase_error(ref, vm, fs, n_fft=n_fft_phase)

            mag_err_list.append(mag_err)
            phase_err_list.append(phase_err)

        mag_err_mean = np.mean(np.vstack(mag_err_list), axis=0)
        phase_err_mean = np.mean(np.vstack(phase_err_list), axis=0)

        # Labels
        if pd.isna(draw):
            draw_tag = "baseline"
        else:
            try:
                draw_tag = f"draw_{int(float(draw))}"
            except Exception:
                draw_tag = f"draw_{draw}"

        name_tag = f"{method}__{geometry}__{draw_tag}"

        # Magnitude error spectrum
        plt.figure(figsize=(12, 4))
        plt.plot(freqs_mag, mag_err_mean, label="Mean over 5 positions")
        plt.xlim(f_min, f_max)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude error (dB)")
        plt.title(f"Phase B magnitude error | {method} | {geometry} | {draw_tag}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        if save_png:
            mag_path = os.path.join(out_dir, f"mag_error__{name_tag}.png")
            plt.savefig(mag_path, dpi=200)
        if show:
            plt.show(block=False)
        else:
            plt.close()

        # Phase error spectrum
        plt.figure(figsize=(12, 4))
        plt.plot(freqs_phase, phase_err_mean, label="Mean over 5 positions")
        plt.xlim(f_min, f_max)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Phase error (rad)")
        plt.title(f"Phase B phase error | {method} | {geometry} | {draw_tag}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        if save_png:
            ph_path = os.path.join(out_dir, f"phase_error__{name_tag}.png")
            plt.savefig(ph_path, dpi=200)
        if show:
            plt.show(block=False)
        else:
            plt.close()

        if save_png:
            print(f"✅ Saved spectra:\n  {mag_path}\n  {ph_path}")

