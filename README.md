# Virtual Microphone Array – Room Simulation
This repository contains tools for simulating **virtual microphones and beamformers** in room environments using [Pyroomacoustics](https://github.com/LCAV/pyroomacoustics). It supports reproducible experiments for the AES 2025 paper on room response correction.

## Features
- Room simulation with configurable microphone array geometry  
- Delay-and-sum and MVDR beamforming hooks  
- Frequency/phase error and SNR metrics  
- Reproducible environment via conda (`virtual-mic-sim.yml`)  
- Example WAV inputs under `test_files/`  

## Repository Structure
vma-room-sim/
├─ src/ # Core Python modules
├─ test_files/
│ └─ wav_inputs/ # Example audio (tiny only, larger kept external)
├─ virtual-mic-sim.yml # Conda environment for reproducibility
├─ .gitignore
└─ README.md

## Quick Start
### 1. Clone the repo
```bash
git clone https://github.com/bdebrit/vma-room-sim.git
cd vma-room-sim

### 2. Create the environment
conda env create -f virtual-mic-sim.yml
conda activate virtual-mic-sim

### 3. Run simulation

- Run main.py

## Reproducibility

Fixed RNG seed for consistent simulations

results/ and large artifacts are git-ignored

To update environment after changes:

  conda env update -f virtual-mic-sim.yml --prune

## Data Policy

Keep only short WAVs in test_files/ for quick smoke tests

Host larger datasets or figure bundles externally (ex: Zenodo/OSF) and link them here

## License

MIT License (see LICENSE)

## Citation

If you use this repository in academic work, please cite the AES 2025 paper:

@inproceedings{debrit2025vma,
  author = {de Brit, Brian},
  title  = {Virtual Microphone Array Room Response Correction ...},
  booktitle = {AES Europe 2025},
  year = {2025}
}

## Roadmap

  - Parameter sweeps for array geometry

  - Spatial heatmaps of beamformer response

  - FIR correction and evaluation
