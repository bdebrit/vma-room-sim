# Import required libraries
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import os
import pickle
import pandas as pd
import soundfile as sf

from itertools import product
from tqdm import tqdm

from scipy.signal import stft
from scipy.signal import chirp
from scipy.signal import correlate, correlation_lags
from sklearn.metrics import mean_squared_error
from numpy import angle, unwrap


# ============================================================
# METRICS PREPROCESS FLAGS (enable one step at a time)
# ============================================================

# STEP 1: Delay-align estimate to reference before metrics
METRICS_ALIGN_SIGNALS = True

# STEP 2: Apply best-fit scalar gain to estimate before metrics
METRICS_GAIN_MATCH = True

# STEP 3: Weight/mask frequency bins by reference magnitude
METRICS_WEIGHT_BY_REF_MAG = False

# Reference magnitude floor (relative to max) for masking in dB
# e.g. -80 dB means ignore bins where |Ref| < max(|Ref|) * 10^(-80/20)
METRICS_REF_MAG_FLOOR_DB = -80.0

# Max lag (seconds) for time alignment search (cross-correlation)
METRICS_ALIGN_MAX_LAG_SEC = 0.02  # 20 ms


def ensure_parent_dir(path: str) -> None:
    """Create parent directory for a file path if needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# ============================================================
# Signal alignment + gain match helpers
# ============================================================

def _estimate_delay_samples(reference, estimate, fs, max_lag_sec=0.02):
    """
    Estimate integer sample delay between reference and estimate
    via cross-correlation. Returns lag in samples.

    Convention (from scipy.signal.correlate):
      If estimate is delayed by +D samples, returned lag is approximately -D.
    """
    ref = np.asarray(reference)
    est = np.asarray(estimate)

    N = min(len(ref), len(est))
    ref = ref[:N]
    est = est[:N]

    max_lag = int(round(max_lag_sec * fs))
    if max_lag < 1:
        return 0

    # Full correlation (FFT method), then restrict lags to [-max_lag, +max_lag]
    corr = correlate(ref, est, mode="full", method="fft")
    lags = correlation_lags(len(ref), len(est), mode="full")

    mask = (lags >= -max_lag) & (lags <= max_lag)
    corr = corr[mask]
    lags = lags[mask]

    lag = int(lags[np.argmax(corr)])
    return lag


def _align_signals(reference, estimate, fs, max_lag_sec=0.02):
    """
    Align estimate to reference using integer lag from cross-correlation.
    Returns (ref_aligned, est_aligned, lag_samples).
    """
    ref = np.asarray(reference)
    est = np.asarray(estimate)

    lag = _estimate_delay_samples(ref, est, fs, max_lag_sec=max_lag_sec)

    # If estimate is delayed by +D, lag ~ -D, so we advance estimate by D = -lag
    if lag < 0:
        # advance estimate
        shift = -lag
        est2 = est[shift:]
        ref2 = ref[:len(est2)]
    elif lag > 0:
        # advance reference
        shift = lag
        ref2 = ref[shift:]
        est2 = est[:len(ref2)]
    else:
        ref2 = ref
        est2 = est

    N = min(len(ref2), len(est2))
    return ref2[:N], est2[:N], lag


def _best_fit_gain(reference, estimate):
    """
    Best-fit scalar gain g that minimises || reference - g*estimate ||^2.
    Returns (estimate_scaled, g).
    """
    ref = np.asarray(reference)
    est = np.asarray(estimate)

    N = min(len(ref), len(est))
    ref = ref[:N]
    est = est[:N]

    denom = float(np.dot(est, est)) + 1e-12
    g = float(np.dot(ref, est)) / denom
    return (g * est), g


def _preprocess_pair(reference, estimate, fs):
    """
    Apply (optional) alignment + (optional) gain match.
    Returns (ref_p, est_p, info_dict).
    """
    ref = np.asarray(reference)
    est = np.asarray(estimate)

    info = {"lag_samples": 0, "gain": 1.0}

    if METRICS_ALIGN_SIGNALS:
        ref, est, lag = _align_signals(ref, est, fs, max_lag_sec=METRICS_ALIGN_MAX_LAG_SEC)
        info["lag_samples"] = lag
    else:
        N = min(len(ref), len(est))
        ref = ref[:N]
        est = est[:N]

    if METRICS_GAIN_MATCH:
        est, g = _best_fit_gain(ref, est)
        info["gain"] = g

    return ref, est, info


# ============================================================
# Ranking + summary
# ============================================================

def summarize_rankings(dataset, fs, room_dims, room_positions,
                      csv_path="test_files/csv/ranking_all_permutations.csv",
                      include_columns=None):
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

        ref = data["ref"]
        virtual = data["virtual"]

        snr = compute_snr(ref, virtual, fs=fs)
        freq_err = compute_freq_error(ref, virtual, fs)
        _, phase_error = compute_phase_error(ref, virtual, fs)

        if isinstance(phase_error, (tuple, list)):
            phase_error = phase_error[0]

        phase_err_scalar = float(np.mean(np.abs(phase_error)))
        combined = compute_combined_error_score(snr, freq_err, phase_err_scalar)

        record["snr_db"] = round(snr, 2)
        record["freq_error_db"] = round(freq_err, 3)
        record["phase_error_rad"] = round(phase_err_scalar, 3)
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
        cols.remove('euclidean_distance')
        cols.insert(radius_idx + 1, 'euclidean_distance')
        grouped = grouped[cols]

    if csv_path:
        summary_path = "test_files/csv/summary_per_position.csv"
        ensure_parent_dir(summary_path)
        ensure_parent_dir(csv_path)
        summary_df.to_csv(summary_path, index=False)
        grouped.to_csv(csv_path, index=False)

    return grouped


# ============================================================
# Simulation helpers
# ============================================================

def add_stereo_signal(room, stereo_signal, fs, left_pos, right_pos):
    """
    Adds two mono sources (left and right) from a stereo signal at specified positions.
    Expected stereo_signal shape: (2, N)
    """
    stereo_signal = np.asarray(stereo_signal)

    if stereo_signal.ndim != 2 or stereo_signal.shape[0] != 2:
        raise ValueError("Expected a stereo signal with shape (2, N)")

    room.add_source(position=left_pos, signal=stereo_signal[0], delay=0.0)
    room.add_source(position=right_pos, signal=stereo_signal[1], delay=0.0)


def _to_stereo_2xN(test_signal, fs, expected_fs=48000):
    """
    Convert input (wav path or ndarray) to stereo with shape (2, N).
    Accepts:
      - WAV path
      - ndarray shape (N,), (1,N), (N,1), (2,N), (N,2)
    """
    if isinstance(test_signal, str) and test_signal.lower().endswith('.wav'):
        signal_data, file_fs = sf.read(test_signal)
        if file_fs != expected_fs:
            raise ValueError(f"Expected {expected_fs} Hz, got {file_fs}")

        if signal_data.ndim == 1:
            # mono
            return np.stack([signal_data, signal_data], axis=0)

        # multichannel (N, C) expected from sf.read
        if signal_data.shape[1] < 2:
            mono = signal_data[:, 0]
            return np.stack([mono, mono], axis=0)

        stereo = signal_data[:, :2].T  # (2, N)
        return stereo

    if isinstance(test_signal, np.ndarray):
        x = np.asarray(test_signal)

        if x.ndim == 1:
            return np.stack([x, x], axis=0)  # (2, N)

        if x.ndim == 2 and x.shape[0] == 2:
            return x  # already (2, N)

        if x.ndim == 2 and x.shape[1] == 2:
            return x.T  # (2, N)

        if x.ndim == 2 and x.shape[0] == 1:
            mono = x.reshape(-1)
            return np.stack([mono, mono], axis=0)

        if x.ndim == 2 and x.shape[1] == 1:
            mono = x.reshape(-1)
            return np.stack([mono, mono], axis=0)

        raise ValueError(f"Unsupported ndarray test_signal shape: {x.shape}")

    raise TypeError("test_signal must be a WAV file path or a NumPy array")


def run_all_room_mic_tests(vms, room_dims, test_signal, dataset_checkpoint_path, use_reflections=False):
    """
    Run all room-style simulations over method, geometry, radius, num_mics.
    Evaluates each configuration at 5 reference mic positions: 1 center + 4 perimeter.
    """

    fs = 48000
    radius_values = [0.1, 0.25, 0.5]
    method_list = ['mvdr', 'delay_sum_time', 'delay_sum_freq']
    audio_format = "stereo"

    geometry_mic_map = {
        'spherical_uniform': [4, 8, 12, 16, 20],
        'tetrahedral': [4],
        'octahedral': [6],
        'icosahedral': [12]
    }

    geometry_num_mic_combinations = [
        (geom, n) for geom, counts in geometry_mic_map.items() for n in counts
    ]
    all_combinations = list(product(geometry_num_mic_combinations, method_list, radius_values))

    room_positions = vms.get_room_style_positions(room_dims)
    mic_array_centre = room_positions['mic_array_centre']
    source_positions = room_positions['source_positions']
    eval_positions = vms.get_room_evaluation_positions(room_positions['ref_mic_pos'], radius=1.0)

    # Convert to stereo (2, N)
    signal_stereo = _to_stereo_2xN(test_signal, fs, expected_fs=fs)

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
            room = vms.create_simulation_room(room_dims, fs, absorption=0.3, max_order=5, use_reflections=use_reflections)

            # ✅ FIX: pass the full stereo (2,N), not only one channel
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

    ensure_parent_dir(dataset_checkpoint_path)
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


# ============================================================
# Test signal generation
# ============================================================

def generate_test_signal(
    signal_type='sine_sweep',
    duration=4.0,
    fs=48000,
    f_start=20,
    f_end=20000,
    apply_fade=True
):
    """
    Generate test signals: sine sweep, log sweep, noise etc.

    Returns:
        signal (np.ndarray): shape (1, N) mono for now (caller can stereo-duplicate)
        t (np.ndarray): time axis
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    if signal_type == 'sine_sweep':
        signal = chirp(t, f0=f_start, f1=f_end, t1=duration, method='linear')
    elif signal_type == 'sine_sweep_log':
        signal = chirp(t, f0=f_start, f1=f_end, t1=duration, method='logarithmic')
    elif signal_type == 'white_noise':
        signal = np.random.randn(len(t))
    else:
        raise ValueError(f"Unsupported signal_type: {signal_type}")

    if apply_fade:
        fade_len = int(0.05 * fs)  # 50ms fade
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = np.linspace(1, 0, fade_len)
        signal[:fade_len] *= fade_in
        signal[-fade_len:] *= fade_out

    return np.array([signal]), t


# ============================================================
# Metrics
# ============================================================

def compute_snr(reference, estimate, fs=48000):
    """
    Compute SNR between reference and estimated signals in dB.
    Optional alignment/gain is controlled by global flags.
    """
    ref, est, _ = _preprocess_pair(reference, estimate, fs)

    noise = ref - est
    snr = 10 * np.log10(np.sum(ref**2) / (np.sum(noise**2) + 1e-12))
    return float(snr)


def compute_freq_error(reference, estimate, fs):
    """
    Mean absolute magnitude ratio error in dB:

        err(f) = 20*log10(|Est(f)| / |Ref(f)|)

    With optional alignment/gain match + optional masking/weighting.
    """
    ref, est, _ = _preprocess_pair(reference, estimate, fs)

    N = min(len(ref), len(est))
    ref = ref[:N]
    est = est[:N]

    ref_mag = np.abs(np.fft.rfft(ref, n=N))
    est_mag = np.abs(np.fft.rfft(est, n=N))

    eps = 1e-12
    err_db = 20.0 * np.log10((est_mag + eps) / (ref_mag + eps))

    # Drop DC bin (often uninformative)
    err_db = err_db[1:]
    ref_mag_use = ref_mag[1:]

    if METRICS_WEIGHT_BY_REF_MAG:
        # Mask bins where ref magnitude is extremely low
        floor_lin = (np.max(ref_mag_use) + 1e-12) * (10.0 ** (METRICS_REF_MAG_FLOOR_DB / 20.0))
        mask = ref_mag_use >= floor_lin
        if np.any(mask):
            w = ref_mag_use[mask]
            w = w / (np.sum(w) + 1e-12)
            return float(np.sum(w * np.abs(err_db[mask])))
        else:
            return float(np.mean(np.abs(err_db)))
    else:
        return float(np.mean(np.abs(err_db)))


def compute_phase_error(reference, estimate, fs, nperseg=1024, noverlap=512):
    """
    Compute phase error vs frequency using STFT.

    Uses wrapped phase difference in [-pi, pi] (more stable than unwrap for error).
    With optional magnitude-weighting across time AND optional ref-bin masking.
    """
    ref, est, _ = _preprocess_pair(reference, estimate, fs)

    N = min(len(ref), len(est))
    ref = ref[:N]
    est = est[:N]

    f, _, Zref = stft(ref, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, _, Zest = stft(est, fs=fs, nperseg=nperseg, noverlap=noverlap)

    # Ensure same time frames
    T = min(Zref.shape[1], Zest.shape[1])
    Zref = Zref[:, :T]
    Zest = Zest[:, :T]

    # Wrapped phase difference (stable)
    phase_diff = np.angle(np.exp(1j * (np.angle(Zref) - np.angle(Zest))))
    abs_diff = np.abs(phase_diff)

    if METRICS_WEIGHT_BY_REF_MAG:
        mag_ref = np.abs(Zref)
        w = mag_ref / (np.sum(mag_ref, axis=1, keepdims=True) + 1e-12)  # per-freq weights over time
        phase_err_per_freq = np.sum(w * abs_diff, axis=1)

        # Mask frequency bins with very low average ref magnitude
        avg_ref_mag = np.mean(mag_ref, axis=1)
        floor_lin = (np.max(avg_ref_mag) + 1e-12) * (10.0 ** (METRICS_REF_MAG_FLOOR_DB / 20.0))
        mask = avg_ref_mag >= floor_lin
        # Return full vector (plotting uses it); downstream averaging can mask if desired
        # We'll still output the full vector; plotting can ignore bins if it wants.
        return f, phase_err_per_freq
    else:
        phase_err_per_freq = np.mean(abs_diff, axis=1)
        return f, phase_err_per_freq


def compute_time_domain_error(reference, estimate):
    """
    Simple time-domain MSE error (with optional alignment/gain via flags).
    """
    ref, est, _ = _preprocess_pair(reference, estimate, fs=48000)
    N = min(len(ref), len(est))
    ref = ref[:N]
    est = est[:N]
    return float(mean_squared_error(ref, est))


def compute_combined_error_score(snr_db, freq_err_db, phase_err_rad):
    """
    Combines SNR (dB), frequency magnitude error (dB), and phase error (radians)
    into a single scalar score. Lower is better.
    """
    snr_db = float(np.atleast_1d(snr_db)[0])
    freq_err_db = float(np.atleast_1d(freq_err_db)[0])
    phase_err_rad = float(np.atleast_1d(phase_err_rad)[0])

    snr_penalty = -snr_db
    combined = freq_err_db + phase_err_rad + snr_penalty
    return round(combined, 3)


def summarize_metrics_table(dataset, metric_fn, sort_by='combined_score', ascending=True, round_digits=3, csv_path=None):
    """
    Summarize dataset into a DataFrame with metrics and rank.
    """
    rows = []
    for label, data in dataset.items():
        ref = data['ref']
        virtual = data['virtual']

        snr = compute_snr(ref, virtual, fs=48000)
        freq_err = compute_freq_error(ref, virtual, fs=48000)
        _, phase_err = compute_phase_error(ref, virtual, fs=48000)

        combined = compute_combined_error_score(snr, freq_err, float(np.mean(np.abs(phase_err))))

        rows.append({
            "label": label,
            "snr_db": round(snr, 2),
            "freq_error_db": round(freq_err, round_digits),
            "phase_error_rad": round(float(np.mean(np.abs(phase_err))), round_digits),
            "combined_score": round(combined, round_digits)
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(by=sort_by, ascending=ascending)
    df["rank"] = range(1, len(df) + 1)

    if csv_path:
        ensure_parent_dir(csv_path)
        df.to_csv(csv_path, index=False)

    return df


# ============================================================
# Plotting
# ============================================================

def plot_virtual_mic_evaluation_from_csv(dataset, ranking_df, fs):
    """
    Plots evaluation curves for top-ranked configurations, grouped by geometry.
    Uses the SAME masking/weighting logic as compute_freq_error (if enabled).
    """
    geometries = ranking_df["geometry"].unique()

    plt.figure(figsize=(14, 8))
    plt.title("Avg Magnitude Error")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Relative Error (dB)")
    plt.xscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)

    for geom in geometries:
        subset = ranking_df[ranking_df["geometry"] == geom]
        best = subset.iloc[0]

        label_prefix = (
            f"geometry={geom},method={best['method']},radius={best['radius']},num_mics={int(best['num_mics'])},"
            f"mic_array_order={best['mic_array_order']},array=room_fixed,audio_channel_format={best['audio_channel_format']}"
        )

        match_label = None
        for k in dataset.keys():
            if k.startswith(label_prefix):
                match_label = k
                break
        if match_label is None:
            continue

        ref = dataset[match_label]["ref"]
        est = dataset[match_label]["virtual"]

        ref_p, est_p, _ = _preprocess_pair(ref, est, fs)

        N = min(len(ref_p), len(est_p))
        ref_p = ref_p[:N]
        est_p = est_p[:N]

        ref_mag = np.abs(np.fft.rfft(ref_p, n=N))
        est_mag = np.abs(np.fft.rfft(est_p, n=N))
        freqs = np.fft.rfftfreq(N, d=1.0/fs)

        eps = 1e-12
        err_db = 20.0 * np.log10((est_mag + eps) / (ref_mag + eps))

        # drop DC for display
        freqs_plot = freqs[1:]
        err_plot = np.abs(err_db[1:])
        ref_mag_use = ref_mag[1:]

        if METRICS_WEIGHT_BY_REF_MAG:
            floor_lin = (np.max(ref_mag_use) + 1e-12) * (10.0 ** (METRICS_REF_MAG_FLOOR_DB / 20.0))
            mask = ref_mag_use >= floor_lin
            freqs_plot = freqs_plot[mask]
            err_plot = err_plot[mask]

        plt.plot(freqs_plot, err_plot, label=f"{geom} ({int(best['num_mics'])} mics)")

    plt.legend(title="Geometry (Mics)")
    plt.tight_layout()

    # Phase error plot
    plt.figure(figsize=(14, 6))
    plt.title("Avg Phase Error (radians)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase Error (rad)")
    plt.xscale("log")
    plt.grid(True, which="both", ls="--", alpha=0.5)

    for geom in geometries:
        subset = ranking_df[ranking_df["geometry"] == geom]
        best = subset.iloc[0]

        label_prefix = (
            f"geometry={geom},method={best['method']},radius={best['radius']},num_mics={int(best['num_mics'])},"
            f"mic_array_order={best['mic_array_order']},array=room_fixed,audio_channel_format={best['audio_channel_format']}"
        )

        match_label = None
        for k in dataset.keys():
            if k.startswith(label_prefix):
                match_label = k
                break
        if match_label is None:
            continue

        ref = dataset[match_label]["ref"]
        est = dataset[match_label]["virtual"]

        freqs, phase_err = compute_phase_error(ref, est, fs)

        # drop DC
        freqs_plot = freqs[1:]
        phase_plot = phase_err[1:]

        if METRICS_WEIGHT_BY_REF_MAG:
            # optional mask by avg magnitude inside compute_phase_error: we keep it simple here
            pass

        plt.plot(freqs_plot, phase_plot, label=f"{geom} ({int(best['num_mics'])} mics)")

    plt.legend(title="Geometry (Mics)")
    plt.tight_layout()


# ============================================================
# WAV export
# ============================================================

def save_signal_to_wav(filename, signal, fs=48000):
    """
    Save mono or stereo signal to WAV.

    Accepts:
      (N,) mono
      (1,N) mono
      (2,N) stereo
      (N,1) mono
      (N,2) stereo

    Writes:
      (N,) mono
      (N,2) stereo
    """
    import numpy as np
    import os
    from scipy.io.wavfile import write

    def ensure_parent_dir(path: str) -> None:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    x = np.asarray(signal)

    if x.ndim == 2 and x.shape[0] == 1:
        x = x.reshape(-1)
    elif x.ndim == 2 and x.shape[1] == 1:
        x = x.reshape(-1)
    elif x.ndim == 2 and x.shape[0] == 2:
        x = x.T
    elif x.ndim == 2 and x.shape[1] == 2:
        pass
    elif x.ndim == 1:
        pass
    else:
        raise ValueError(f"Unsupported signal shape for WAV: {x.shape}")

    if x.dtype != np.int16:
        peak = float(np.max(np.abs(x))) if x.size else 0.0
        if peak > 0:
            x = x / peak
        x = (x * 32767.0).astype(np.int16)

    ensure_parent_dir(filename)
    write(filename, fs, x)
