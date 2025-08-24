import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt

class VirtualMicSimulation:
    def __init__(self, dataset_checkpoint_path, room_dims):
        self.dataset_checkpoint_path = dataset_checkpoint_path
        self.room_dims = room_dims
        self.mode = "single"

    def get_room_style_positions(self, room_dims):
        # Example positions (you likely have your own implementation already)
        room_positions = {
            "source_positions": [
                [1.0, room_dims[1] / 2.0 - 1.0, 1.5],
                [1.0, room_dims[1] / 2.0 + 1.0, 1.5]
            ],
            "ref_mic_pos": [room_dims[0] / 2.0, room_dims[1] / 2.0, 1.5],
            "mic_array_centre": [room_dims[0] / 2.0, room_dims[1] / 2.0, 3.25]
        }
        return room_positions

    def get_room_evaluation_positions(self, center_pos, radius=1.0):
        center = np.array(center_pos)
        offsets = [
            [0, 0, 0],
            [radius, 0, 0],
            [-radius, 0, 0],
            [0, radius, 0],
            [0, -radius, 0]
        ]
        return [list(center + np.array(o)) for o in offsets]

    def estimate_array_order(self, geometry, num_mics):
        # Placeholder heuristic (keep your existing logic if different)
        if geometry == "tetrahedral":
            return 1
        if geometry == "octahedral":
            return 1
        if geometry == "icosahedral":
            return 2
        if geometry == "spherical_uniform":
            return 2 if num_mics >= 12 else 1
        return 1

    def generate_array_geometry(self, center, geometry, radius, num_mics):
        """
        Deterministic mic layouts on a sphere around `center`.
        Returns positions in shape (3, N) to match pyroomacoustics convention.
        """
        center = np.array(center, dtype=float).reshape(3, 1)

        def _normalize_cols(P):
            return P / (np.linalg.norm(P, axis=0, keepdims=True) + 1e-12)

        def _fibonacci_sphere(N):
            # Deterministic "uniform-ish" points on a sphere (3, N)
            golden_angle = np.pi * (3.0 - np.sqrt(5.0))
            i = np.arange(N, dtype=float)
            z = 1.0 - 2.0 * (i + 0.5) / N
            r_xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
            theta = golden_angle * i
            x = np.cos(theta) * r_xy
            y = np.sin(theta) * r_xy
            return np.vstack([x, y, z])

        geometry = (geometry or "").lower().strip()

        if geometry == "tetrahedral":
            # 4 vertices
            P = np.array([
                [ 1,  1, -1, -1],
                [ 1, -1,  1, -1],
                [ 1, -1, -1,  1],
            ], dtype=float)
            P = _normalize_cols(P)
            if num_mics != 4:
                # fall back to spherical_uniform if user asked for a different count
                P = _fibonacci_sphere(num_mics)

        elif geometry == "octahedral":
            # 6 vertices: ±x, ±y, ±z
            P = np.array([
                [ 1, -1,  0,  0,  0,  0],
                [ 0,  0,  1, -1,  0,  0],
                [ 0,  0,  0,  0,  1, -1],
            ], dtype=float)
            P = _normalize_cols(P)
            if num_mics != 6:
                P = _fibonacci_sphere(num_mics)

        elif geometry == "icosahedral":
            # 12 vertices
            phi = (1.0 + np.sqrt(5.0)) / 2.0
            P = np.array([
                [0, 0, 0, 0,  1, -1,  1, -1,  phi, -phi,  phi, -phi],
                [1, -1, 1, -1,  phi,  phi, -phi, -phi,  0,   0,   0,   0 ],
                [phi, phi, -phi, -phi, 0, 0, 0, 0,  1,   1,  -1,  -1 ],
            ], dtype=float)
            P = _normalize_cols(P)
            if num_mics != 12:
                P = _fibonacci_sphere(num_mics)

        elif geometry == "spherical_uniform":
            # Deterministic uniform-ish distribution
            P = _fibonacci_sphere(num_mics)

        else:
            # Unknown geometry -> deterministic uniform-ish
            P = _fibonacci_sphere(num_mics)

        return center + radius * P


    def plot_room_configuration(self, room_positions, eval_positions, mic_positions, room_dims):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # ----------------------------
        # Room wireframe
        # ----------------------------
        xs = [0, room_dims[0]]
        ys = [0, room_dims[1]]
        zs = [0, room_dims[2]]

        for x in xs:
            for y in ys:
                ax.plot([x, x], [y, y], zs, color='gray', alpha=0.3)

        for x in xs:
            for z in zs:
                ax.plot([x, x], ys, [z, z], color='gray', alpha=0.3)

        for y in ys:
            for z in zs:
                ax.plot(xs, [y, y], [z, z], color='gray', alpha=0.3)

        # ----------------------------
        # Sources (green squares + labels)
        # ----------------------------
        src = np.array(room_positions["source_positions"], dtype=float)
        if src.ndim == 1:
            src = src.reshape(1, 3)

        if src.shape[0] >= 1:
            ax.scatter(src[0, 0], src[0, 1], src[0, 2], marker="s", s=40, color="green", label="Source L")
            ax.text(src[0, 0], src[0, 1], src[0, 2], "SrcL", fontsize=8)

        if src.shape[0] >= 2:
            ax.scatter(src[1, 0], src[1, 1], src[1, 2], marker="s", s=40, color="green", label="Source R")
            ax.text(src[1, 0], src[1, 1], src[1, 2], "SrcR", fontsize=8)

        for i in range(2, src.shape[0]):
            ax.scatter(src[i, 0], src[i, 1], src[i, 2], marker="s", s=40, color="green")
            ax.text(src[i, 0], src[i, 1], src[i, 2], f"Src{i}", fontsize=8)

        # ----------------------------
        # Eval positions (red triangles + labels)
        # ----------------------------
        ep = np.array(eval_positions, dtype=float)
        if ep.ndim == 1:
            ep = ep.reshape(1, 3)

        ax.scatter(ep[:, 0], ep[:, 1], ep[:, 2], marker="^", s=35, color="red", label="Evaluation positions")

        for i in range(ep.shape[0]):
            ax.text(ep[i, 0], ep[i, 1], ep[i, 2], f"Ref{i}", color="red", fontsize=8)

        # ----------------------------
        # Evaluation radius (red dotted circle)
        # ----------------------------
        if isinstance(room_positions, dict) and "ref_mic_pos" in room_positions:
            centre_ref = np.array(room_positions["ref_mic_pos"], dtype=float).reshape(3,)
        else:
            centre_ref = ep[0, :]

        r = float(np.max(np.linalg.norm(ep[:, :2] - centre_ref[:2], axis=1)))
        theta = np.linspace(0, 2*np.pi, 200)
        circle_x = centre_ref[0] + r * np.cos(theta)
        circle_y = centre_ref[1] + r * np.sin(theta)
        circle_z = np.ones_like(theta) * centre_ref[2]
        ax.plot(circle_x, circle_y, circle_z, color="red", linestyle=":", linewidth=1.0, label="Evaluation Radius")

        # ----------------------------
        # Mic array mics (blue dots) + centre (orange)
        # ----------------------------
        mp_in = np.array(mic_positions, dtype=float)
        if mp_in.ndim != 2:
            raise ValueError(f"mic_positions must be 2D with shape (3,N) or (N,3). Got {mp_in.shape}")

        if mp_in.shape[0] == 3:
            mp = mp_in
        elif mp_in.shape[1] == 3:
            mp = mp_in.T
        else:
            raise ValueError(f"mic_positions must be (3,N) or (N,3). Got {mp_in.shape}")

        ax.scatter(mp[0, :], mp[1, :], mp[2, :], marker=".", s=20, color="blue", label="Array Mics")

        if isinstance(room_positions, dict) and "mic_array_centre" in room_positions:
            c = np.array(room_positions["mic_array_centre"], dtype=float).reshape(3,)
        else:
            c = np.array([np.mean(mp[0, :]), np.mean(mp[1, :]), np.mean(mp[2, :])], dtype=float)

        ax.scatter(c[0], c[1], c[2], marker="o", s=40, color="orange", label="Mic Array Centre")

        # ----------------------------
        # TRACE LINES: Mic array centre -> each evaluation (ref) position (blue, faint)
        # ----------------------------
        for i in range(ep.shape[0]):
            ax.plot([c[0], ep[i, 0]], [c[1], ep[i, 1]], [c[2], ep[i, 2]],
                    color="blue", alpha=0.35, linewidth=0.8)

        # ----------------------------
        # TRIANGLE BETWEEN SPEAKERS AND Ref0 (faint dotted green)
        #   Source L -> Ref0 -> Source R -> Source L
        # ----------------------------
        if src.shape[0] >= 2:
            tri = np.vstack([src[0, :], centre_ref, src[1, :], src[0, :]])
            ax.plot(tri[:, 0], tri[:, 1], tri[:, 2],
                    color="green", linestyle=":", alpha=0.45, linewidth=1.0)

        # ----------------------------
        # Axes / scaling / view
        # ----------------------------
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Room Configuration")

        ax.set_xlim(0, room_dims[0])
        ax.set_ylim(0, room_dims[1])
        ax.set_zlim(0, room_dims[2])

        try:
            ax.set_box_aspect((room_dims[0], room_dims[1], room_dims[2]))
        except Exception:
            pass

        try:
            ax.view_init(elev=20, azim=135)
        except Exception:
            pass

        ax.legend()








    def create_simulation_room(self, room_dims, fs, absorption=0.3, max_order=5, use_reflections=True):
        """
        Create a Pyroomacoustics simulation room with given parameters.

        Args:
            room_dims (list): Room dimensions [x, y, z].
            fs (int): Sampling frequency.
            absorption (float): Absorption coefficient for the walls.
            max_order (int): Maximum order of reflections.
            use_reflections (bool): If False, forces direct-path only.

        Returns:
            pra.ShoeBox: Room object ready for sources and mic arrays.
        """
        # Direct-path only if reflections disabled
        if not use_reflections:
            max_order = 0

        room = pra.ShoeBox(
            p=room_dims,
            fs=fs,
            materials=pra.Material(absorption),
            max_order=max_order
        )

        return room

    def compute_virtual_microphone(self, room, method, virtual_position, mic_positions):
        # This dispatches to your existing implementations in your real code.
        # Keep your existing mapping if different.
        if method == "delay_sum_time":
            return self.compute_virtual_microphone_delay_sum_time_beamformer(room, virtual_position, mic_positions)
        elif method == "delay_sum_freq":
            return self.compute_virtual_microphone_delay_sum_freq_beamformer(room, virtual_position, mic_positions)
        elif method == "mvdr":
            return self.compute_virtual_microphone_mvdr_beamformer(room, virtual_position, mic_positions)
        else:
            raise ValueError(f"Unknown method: {method}")

    # --- Your existing beamformer implementations should remain here ---
    def compute_virtual_microphone_delay_sum_time_beamformer(self, room, virtual_position, array_positions, window_type="hamming"):
        # KEEP your existing implementation (not rewritten here)
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

        # Simple integer delay (placeholder) — keep your fractional delay version in your real file
        aligned = []
        for i in range(n_mics):
            shift = int(round(max_delay - delays[i]))
            sig = mic_signals[i]
            if shift > 0:
                sig = np.pad(sig, (shift, 0))[:len(sig)]
            aligned.append(sig)

        virtual_signal = np.mean(aligned, axis=0)
        return virtual_signal

    def compute_virtual_microphone_delay_sum_freq_beamformer(self, room, virtual_position, array_positions):
        # KEEP your existing implementation (placeholder)
        return self.compute_virtual_microphone_delay_sum_time_beamformer(room, virtual_position, array_positions)

    def compute_virtual_microphone_mvdr_beamformer(self, room, virtual_position, array_positions):
        # KEEP your existing implementation (placeholder)
        return self.compute_virtual_microphone_delay_sum_time_beamformer(room, virtual_position, array_positions)
