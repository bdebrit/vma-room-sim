from analysis_utils import *
from virtual_mic_simulation import VirtualMicSimulation
import pickle
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # === Config ===
    fs = 48000
    room_dims = [12, 10, 4.0]  # [X, Y, Z] in meters

    # Phase A settings
    PHASE_A_NUM_DRAWS = 200
    PHASE_A_SEED = 0
    MIC_RADIUS = 0.25
    USE_REFLECTIONS = False  # Phase A: disable reflections (max_order=0)
    METHODS = ['mvdr', 'delay_sum_time', 'delay_sum_freq']

    # Paths are relative to repo root (directory above src)
    dataset_checkpoint_path = "test_files/bin/checkpoint_phase_a.pkl"
    output_sweep_path = "test_files/wav_inputs/test_sweep.wav"

    # === Generate and Save Test Signal ===
    signal = generate_test_signal(
        signal_type='sine_sweep_log',
        duration=5.0,
        fs=fs,
        f_start=20,
        f_end=20000,
        apply_fade=True
    )[0]  # mono (N,)
    # Save as stereo wav (N,2) for convenience; pipeline will coerce internally anyway
    stereo = np.column_stack([signal, signal])
    save_signal_to_wav(output_sweep_path, stereo, fs)

    # === Virtual Mic Setup ===
    vms = VirtualMicSimulation(dataset_checkpoint_path, room_dims)
    room_positions = vms.get_room_style_positions(room_dims)
    eval_positions = vms.get_room_evaluation_positions(room_positions['ref_mic_pos'], radius=1.0)

    # Plot a representative configuration (tetrahedral at chosen radius)
    mic_positions = vms.generate_array_geometry(room_positions['mic_array_centre'], 'tetrahedral', MIC_RADIUS, 4)
    vms.plot_room_configuration(
        room_positions=room_positions,
        eval_positions=eval_positions,
        mic_positions=mic_positions,
        room_dims=room_dims
    )

    # === Phase A Monte Carlo ===
    summary_df, ranking_df = run_phase_a_monte_carlo(
        vms=vms,
        room_dims=room_dims,
        test_signal=stereo,
        dataset_checkpoint_path=dataset_checkpoint_path,
        num_draws=PHASE_A_NUM_DRAWS,
        seed=PHASE_A_SEED,
        radius=MIC_RADIUS,
        method_list=METHODS,
        use_reflections=USE_REFLECTIONS
    )

    # === Load Results and Plot Virtual Mic Evaluation ===
    with open(dataset_checkpoint_path, "rb") as f:
        checkpoint = pickle.load(f)
        dataset = checkpoint["dataset"]

    plot_virtual_mic_evaluation_from_csv(dataset, ranking_df, fs)
    plt.show()
