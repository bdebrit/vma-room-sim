from analysis_utils import *
from virtual_mic_simulation import VirtualMicSimulation
import os
import pandas as pd

# ============================================================
# Phase B: Validation run + robust resume/postprocess
# ============================================================

PHASE_B_RESUME_IF_CHECKPOINT_EXISTS = True

PHASE_A_SUMMARY_CSV = "test_files/csv/summary_per_position.csv"
PHASE_B_CHECKPOINT = "test_files/bin/checkpoint_phase_b.pkl"
PHASE_B_OUT_DIR = "test_files/csv/phase_b"

PHASE_B_TOP_K_DRAWS = 8
PHASE_B_RADIUS = 0.25
PHASE_B_METHODS = ["mvdr", "delay_sum_time", "delay_sum_freq"]

PHASE_B_INCLUDE_COLUMNS = [
    "signal_name",
    "method",
    "geometry",
    "num_mics",
    "radius",
    "draw",
    "min_pairwise_dist",
    "mean_pairwise_dist",
    "spread_eig_ratio",
    "z_range",
]

# ============================================================
# Phase B spectrum plots (requires checkpoint + ranking CSV)
# ============================================================

PLOT_PHASE_B_SPECTRA = True
PHASE_B_SPECTRA_OUT_DIR = "graphs/phase_b_spectra"

if __name__ == "__main__":
    fs = 48000
    room_dims = [12, 10, 4.0]

    sweep_duration = 5.0
    sweep_freq_range = (20, 20000)

    os.makedirs("test_files/bin", exist_ok=True)
    os.makedirs("test_files/csv", exist_ok=True)
    os.makedirs(PHASE_B_OUT_DIR, exist_ok=True)
    os.makedirs("test_files/wav_inputs", exist_ok=True)
    os.makedirs("graphs", exist_ok=True)

    # Generate MONO sweep, then duplicate to stereo (2, N)
    mono = generate_test_signal(
        signal_type="sine_sweep_log",
        duration=sweep_duration,
        fs=fs,
        f_start=sweep_freq_range[0],
        f_end=sweep_freq_range[1],
        stereo=False,
        apply_fade=True,
        preview=False
    )[0]

    stereo_2n = np.stack([mono, mono], axis=0)  # (2, N)

    save_signal_to_wav("test_files/wav_inputs/test_sweep.wav", stereo_2n.T, fs)

    vms = VirtualMicSimulation(PHASE_B_CHECKPOINT, room_dims)
    room_positions = vms.get_room_style_positions(room_dims)

    print("=== Phase B: validation run (4-mic random spherical vs baselines) ===")
    print(f"Phase A summary: {PHASE_A_SUMMARY_CSV}")
    print(f"Phase B checkpoint: {PHASE_B_CHECKPOINT}")

    # ------------------------------------------------------------
    # RESUME PATH: if checkpoint exists, do NOT rerun simulation
    # ------------------------------------------------------------
    if PHASE_B_RESUME_IF_CHECKPOINT_EXISTS and os.path.exists(PHASE_B_CHECKPOINT):
        print(f"🔁 Found existing Phase B checkpoint: {PHASE_B_CHECKPOINT}")

        ranking_df = postprocess_phase_b_checkpoint(
            checkpoint_path=PHASE_B_CHECKPOINT,
            fs=fs,
            room_dims=room_dims,
            room_positions=room_positions,
            out_dir=PHASE_B_OUT_DIR,
            include_columns=PHASE_B_INCLUDE_COLUMNS,
        )
        print(f"✅ Phase B postprocess complete. Ranking CSV: {PHASE_B_OUT_DIR}/ranking_phase_b.csv")

        # Spectrum plots (magnitude error + phase error vs frequency)
        if PLOT_PHASE_B_SPECTRA:
            try:
                ranking_csv = os.path.join(PHASE_B_OUT_DIR, "ranking_phase_b.csv")
                plot_phase_b_best_error_spectra(
                    checkpoint_path=PHASE_B_CHECKPOINT,
                    ranking_csv_path=ranking_csv,
                    out_dir=PHASE_B_SPECTRA_OUT_DIR,
                    fs=fs,
                    by_method=True,
                    show=True,
                    save_png=True,
                )
            except Exception as e:
                print(f"⚠️ Phase B spectrum plotting failed: {e}")

        raise SystemExit(0)

    # ------------------------------------------------------------
    # RUN PATH: run Phase B simulation (slow), then postprocess
    # ------------------------------------------------------------
    test_signals = {"sweep": stereo_2n}

    run_phase_b_validation(
        vms=vms,
        room_dims=room_dims,
        test_signals=test_signals,
        phase_a_summary_csv_path=PHASE_A_SUMMARY_CSV,
        dataset_checkpoint_path=PHASE_B_CHECKPOINT,
        top_k=PHASE_B_TOP_K_DRAWS,
        radius=PHASE_B_RADIUS,
        method_list=PHASE_B_METHODS,
        use_reflections=False,
        absorption=0.3,
        max_order_reflections=5,
        eval_radius=1.0,
        include_uniform_4=True,
        mic_jitter_std_m=0.0,
        resume_if_checkpoint_exists=False,
    )

    print("✅ Phase B run complete.")

    # Postprocess to create ranking CSV (and optionally plot spectra)
    try:
        ranking_df = postprocess_phase_b_checkpoint(
            checkpoint_path=PHASE_B_CHECKPOINT,
            fs=fs,
            room_dims=room_dims,
            room_positions=room_positions,
            out_dir=PHASE_B_OUT_DIR,
            include_columns=PHASE_B_INCLUDE_COLUMNS,
        )
        print(f"✅ Phase B postprocess complete. Ranking CSV: {PHASE_B_OUT_DIR}/ranking_phase_b.csv")
    except Exception as e:
        print(f"⚠️ Phase B postprocess failed: {e}")
        raise

    if PLOT_PHASE_B_SPECTRA:
        try:
            ranking_csv = os.path.join(PHASE_B_OUT_DIR, "ranking_phase_b.csv")
            plot_phase_b_best_error_spectra(
                checkpoint_path=PHASE_B_CHECKPOINT,
                ranking_csv_path=ranking_csv,
                out_dir=PHASE_B_SPECTRA_OUT_DIR,
                fs=fs,
                by_method=True,
                show=True,
                save_png=True,
            )
        except Exception as e:
            print(f"⚠️ Phase B spectrum plotting failed: {e}")

    print("✅ Phase B complete.")
