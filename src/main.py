from analysis_utils import *
from virtual_mic_simulation import VirtualMicSimulation
import pickle
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    # === Config ===
    fs = 48000
    room_dims = [12, 10, 4.0]  # [X, Y, Z] in meters
    num_mics = 20
    sweep_duration = 5.0
    sweep_freq_range = (20, 20000)

    # Paths are relative to the directory you run from (repo root / directory above src)
    dataset_checkpoint_path = "test_files/bin/checkpoint_room_results.pkl"
    output_sweep_path = "test_files/wav_inputs/test_sweep.wav"

    mic_geometry = "spherical_uniform"

    # === Generate and Save Test Signal ===
    mono = generate_test_signal(
        signal_type="sine_sweep_log",
        duration=sweep_duration,
        fs=fs,
        f_start=sweep_freq_range[0],
        f_end=sweep_freq_range[1],
        apply_fade=True,
    )[0].reshape(-1)  # (N,)

    # Use (2, N) stereo convention for the simulation code
    stereo = np.stack([mono, mono], axis=0)  # (2, N) two separate channels (identical)

    # Save as a standard WAV layout (N, 2) for external tools/players
    stereo_wav = stereo.T  # (N, 2)
    save_signal_to_wav(output_sweep_path, stereo_wav, fs)

    # === Virtual Mic Setup ===
    vms = VirtualMicSimulation(dataset_checkpoint_path, room_dims)
    room_positions = vms.get_room_style_positions(room_dims)
    eval_positions = vms.get_room_evaluation_positions(room_positions["ref_mic_pos"], radius=1.0)
    mic_positions = vms.generate_array_geometry(
        room_positions["mic_array_centre"], mic_geometry, 0.25, num_mics
    )

    vms.plot_room_configuration(
        room_positions=room_positions,
        eval_positions=eval_positions,
        mic_positions=mic_positions,
        room_dims=room_dims,
    )

    # === Run Tests (FIR disabled, reflections disabled) ===
    summary_df, ranking_df = run_all_room_mic_tests(
        vms, room_dims, stereo, dataset_checkpoint_path, use_reflections=False
    )

    # === Load Results and Plot Virtual Mic Evaluation ===
    with open(dataset_checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)
        dataset = checkpoint["dataset"]

    plot_virtual_mic_evaluation_from_csv(dataset, ranking_df, fs)

    plt.show()
