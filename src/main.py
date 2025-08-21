from analysis_utils import *
from virtual_mic_simulation import VirtualMicSimulation
from virtual_mic_test import VirtualMicTest
import pickle
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # === Config ===
    fs = 48000
    room_dims = [12, 10, 4.0]  # [X, Y, Z] in meters
    num_mics = 20
    sweep_duration = 5.0
    sweep_freq_range = (20, 20000)
    sweep_time_range = (0, 5)
    dataset_checkpoint_path = "../test_files/bin/checkpoint_room_results.pkl"
    output_sweep_path = '../test_files/wav_inputs/test_sweep.wav'
    mic_geometry = 'spherical_uniform'
    modes = ['single', 'multi', 'avg']

    # === Generate and Save Test Signal ===
    signal = generate_test_signal(
        signal_type='sine_sweep_log',
        duration=sweep_duration,
        fs=fs,
        f_start=sweep_freq_range[0],
        f_end=sweep_freq_range[1],
        apply_fade=True
    )[0]
    save_signal_to_wav(output_sweep_path, signal)

    # === Virtual Mic Setup ===
    vms = VirtualMicSimulation(dataset_checkpoint_path, room_dims)
    room_positions = vms.get_room_style_positions(room_dims)
    eval_positions = vms.get_room_evaluation_positions(room_positions['ref_mic_pos'], radius=1.0)
    mic_positions = vms.generate_array_geometry(room_positions['mic_array_centre'], mic_geometry, 0.25, num_mics)

    vms.plot_room_configuration(
        room_positions=room_positions,
        eval_positions=eval_positions,
        mic_positions=mic_positions,
        room_dims=room_dims
    )

    # === Run Tests ===
    summary_df, ranking_df = run_all_room_mic_tests(vms, room_dims, signal, dataset_checkpoint_path)

    # === Load Results and Plot Virtual Mic Evaluation ===
    with open(dataset_checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)
        dataset = checkpoint["dataset"]

    plot_virtual_mic_evaluation_from_csv(dataset, ranking_df, fs)

    # === FIR Evaluation for Each Mode ===
    results_by_mode = {}
    for mode in modes:
        vms.mode = mode
        results = VirtualMicTest.from_checkpoint_with_fir_mode(
            dataset_checkpoint_path=dataset_checkpoint_path,
            mode=mode,
            numtaps=512,
            start_time=sweep_time_range[0],
            end_time=sweep_time_range[1],
            start_freq=sweep_freq_range[0],
            end_freq=sweep_freq_range[1]
        )

        metrics_df = results.evaluate_metrics_for_all_positions_window(mode=mode)
        avg_combined = metrics_df["Combined Score"].mean()
        results.combined_score = avg_combined

        results_by_mode[mode] = results
        print(metrics_df)

    # === Plot FIR Results ===
    plot_fir_evaluation_overlay(results_by_mode, fs)

    plt.show()
