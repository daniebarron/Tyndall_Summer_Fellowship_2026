# Tyndall Summer Fellowship 2026

Daniel Barron, supervised by Dr. Michael Nolan (Tyndall National Institute, Cork).

Two computational projects that share one theme: DFT is accurate but slow, and machine learning can extend what we can learn from it.

1. **PFAS spectroscopy** (`orca_work/`): DFT-computed IR and 19F NMR spectra for 68 PFAS molecules, and machine-learning models built on them for spectral matching, functional-group classification and spectrum prediction.
2. **Nitride oxidation** (`Matlantis_Round_2/`): oxidation of ZrN and VN surfaces with the Matlantis neural network potential (PFP), reaching timescales well beyond DFT-based ab initio MD, benchmarked against Ji et al., *J. Phys. Chem. C* 2025, 129, 11173-11182.

> Raw wavefunction files, ASE trajectories (`*.traj`) and the reference VASP data are **not** in this repository (see [What is not included](#what-is-not-included)).

---

## 1. PFAS spectroscopy

### Methods

| Step | Level of theory / settings |
|---|---|
| Geometry optimisation + frequencies | ORCA 6.1.1, PBE0-D3BJ / def2-SVP, RIJCOSX (def2/J), TightSCF |
| Optimisation convergence | Deliberately loose for speed across 68 molecules: TolE 1e-4, TolRMSG 1e-2, TolMaxG 3e-2, TolRMSD 1e-3, TolMaxD 3e-3 |
| IR spectra | Harmonic frequencies scaled by 0.9614, Lorentzian broadening (FWHM 20 cm^-1), 500-4000 cm^-1 on 3501 points |
| NMR | Single points on the optimised geometries, PBE0 / def2-TZVP, GIAO |
| NMR references | 19F relative to CFCl3 (sigma_ref = 179.716 ppm), 1H and 13C relative to TMS, computed at the same level |
| Starting structures | 3D coordinates from verified SMILES with RDKit (MMFF) |

The 68 molecules and their SMILES are in `orca_work/PFAS_smiles.csv`. IR scaling follows Merrick et al. (2007). IR results for TFA, PFBA and PFOA are compared with experimental NIST spectra in `orca_work/ir_checks/`.

### Pipeline (run order)

All scripts have a configuration block at the top. **Paths are Windows paths from my machine and must be edited before running.**

| # | Script | What it does |
|---|---|---|
| 1 | `pfas_automation.py` | SMILES CSV to ORCA opt+freq inputs (one folder per molecule) and `run_all.bat` |
| 2 | `audit_calculations.py`, `generate_freq_restarts.py`, `check_imaginary_freqs.py` | Find failed jobs, restart frequency calculations that crashed, flag imaginary frequencies (`pfas_calculations/check_imaginary_freqs_results.csv`) |
| 3 | `pfas_out_files/verify_orca_smiles.py`, `check_orca_bonds.py` | Check optimised structures against the intended SMILES |
| 4 | `build_pfas_ir_dataset.py` | Parse frequencies and intensities, broaden, write `ml_dataset/pfas_ir_dataset.npz` |
| 5 | `generate_nmr_inputs.py` | NMR inputs and SLURM scripts (`nmr_inputs/`, run on the Ada cluster) |
| 6 | `collect_nmr_outputs.py` | Gather NMR outputs into `nmr_outputs/` |
| 7 | `build_pfas_nmr_dataset.py`, `build_pfas_combined_dataset.py`, `build_pfas_nmr_spectra.py` | Shieldings to shifts, combined IR+NMR dataset, broadened 19F spectra |
| 8 | `pfas_ml_analysis_new.py` | Tasks 1-3 (below), IR only vs IR + NMR |
| 9 | `pfas_tasks_4_5.py` | Tasks 4-5 |
| 10 | `pfas_nmr_spectra_ml.py` | Tasks A-C on 19F NMR spectra |
| 11 | `pfas_ml_analysis_predict.py` | Predict the IR spectrum of a new molecule from its SMILES |

`pfas_ml_analysis.py` is the earlier IR-only version of step 8. `*_new3` scripts and folders (`ml_dataset_new3/`, `pfas_new_*`) are the same pipeline run on the three held-out molecules.

### Machine-learning tasks

All results use leave-one-out cross-validation on the 68 molecules.

| Task | Question | Method |
|---|---|---|
| 1 | Can a spectrum identify its molecule? | Cosine-similarity library matching |
| 2 | Can functional groups be read from the spectrum? | Random forest on spectra |
| 3 | Can the IR spectrum be predicted from the structure? | RDKit descriptors, Ridge regression on PCA of the spectra |
| 4 | Can NMR features be predicted from IR? | Ridge regression on IR PCA |
| 5 | Does adding NMR improve identification? | Score fusion of IR and NMR similarity |
| A / B / C | Descriptors to 19F spectrum / IR to 19F spectrum / 19F spectrum to IR | Ridge + PCA |

### Key results

- **Task 3:** predicted IR spectra reach a mean Pearson r of 0.91 (cosine similarity 0.92) from 18 RDKit descriptors alone.
- **Task 5:** IR-only retrieval ranks the correct molecule first for all 68 molecules. This is a closed-world result (the query molecule's own spectrum type is in the library), so it shows the spectra are distinguishable, not that unseen molecules would be identified perfectly. 19F NMR features alone give top-1 of 16%.
- **IR and 19F NMR carry different information.** IR predicts mean 19F shift poorly (Task 4, r = 0.16), and adding NMR features did not generally improve Tasks 1-3. Predicting IR from 19F NMR (Task C, r = 0.84) works better than the reverse (Task B, r = 0.65).
- Full logs: `orca_work/ml_dataset/pfas_ml_results.txt` and `pfas_tasks45_results.txt`; figures are in the same folder.

### Held-out molecules

6:2 FTSA, protonated 6:2 FTAB and DONA (3H interior) were computed separately and are kept out of the training set (`pfas_new/`, `ml_dataset_new3/`, `predicted_IR_new_molecules.png`).

### Calculation status

Two jobs showed imaginary frequencies in the check (`8-2_FTCA`: one at -5.6 cm^-1; `8-2_diPAP`: three, down to -43.6 cm^-1) and were re-optimised (`pfas_reopt/`, `8-2_diPAP_corrected.inp`). Ten original output files appear as `ERROR` in the check CSV because their frequency step crashed (no frequency block, no normal termination). Those frequencies were recomputed in restart runs (`generate_freq_restarts.py`, files named `*_freq.out`), and these are the files used in `pfas_out_files/`.

---

## 2. Nitride oxidation with Matlantis (PFP)

### Method

- Potential: PFP v9.0.0 (`EstimatorCalcMode.PBE`, to match the PBE functional in the VASP reference), via ASE.
- Iterative oxidation, following Ji et al.: each round adds O atoms 1.5 A above the surface (6 for ZrN, 10 for VN) with a downward velocity, runs NVT MD, removes everything not bound to the surface, and repeats for up to 40 rounds, then a final FIRE relaxation (fmax = 0.05 eV/A).
- MD: Langevin thermostat (friction 0.001 fs^-1), 2.5 fs timestep, **8000 steps = 20 ps per round**, so up to 800 ps per seed. (Ji et al. use 6.25 ps rounds with a Nose-Hoover chain.)
- Five independent random seeds per system and temperature.
- Starting slabs with atoms closer than 1.5 A are pre-relaxed with FIRE, with the bottom 20% of the slab fixed. Polar VN(111) slabs can break up under PFP, so `get_best_structure.py` and `analyse_run.py` identify the last intact structure of each seed (`is_healthy` check).

### Systems

| Surface | Temperatures (K) | O atoms per round |
|---|---|---|
| ZrN(100) | 295, 363, 1023 | 6 |
| ZrN(110) | 295, 363, 1023 | 6 |
| VN(111) | 295, 363, 873 | 10 |

Each `Matlantis_Round_2/Matlantis/<surface>/<T>K/` folder holds the MD and relaxation logs, an oxidation plot, the best structure (`best_structure.png/pdf`) and `seed_1` ... `seed_5` with per-seed logs and `seed_info.json`.

### Scripts

| File | Purpose |
|---|---|
| `scripts/Personal_scripts/<system><T>.ipynb` | Oxidation run for one system/temperature (e.g. `vn111363` = VN(111), 363 K). Same code, different parameters |
| `scripts/Personal_scripts/Etch-run_Ji-protocol_5seeds.ipynb` | General five-seed version of the protocol |
| `scripts/Personal_scripts/analyse_run.py` | Per-run, per-seed and summary analysis, with Ji et al. reference values built in |
| `scripts/Personal_scripts/get_best_structure.py` | Last healthy structure before slab breakdown |
| `scripts/Personal_scripts/view_local.py`, `View_trajectory.ipynb` | Trajectory viewing |
| `scripts/*.ipynb`, `neb_info.py` | Matlantis group template notebooks (relaxation, NPT/NVT MD, etch, NEB), included for reference and not my own work |
| `convert.py`, `make_reactant.py` | Convert reference VASP structures to ASE `.traj`, create the single-O reactant |

### Results

*TODO: add the summary table from `python analyse_run.py --summary-seeds <Matlantis dir>` (PFP mean +/- spread over five seeds against the Ji et al. reference values) and one or two sentences on the main finding: PFP reproduces the qualitative oxidation trends but binds oxygen more strongly than DFT-aiMD, and polar VN(111) is the hardest case.*

---

## What is not included

- ORCA wavefunction and scratch files (`*.gbw`, `*.densities`, ...), see `.gitignore`. Full raw data is on Tyndall storage.
- ASE trajectories (`*.traj`) from the Matlantis runs.
- The reference VASP aiMD data from Ji et al.

## Software

ORCA 6.1.1 (versions taken from the output headers); Python with `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `rdkit` for the PFAS pipeline; `ase`, `pfp-api-client` and the Matlantis environment for the nitride work. See `requirements.txt`.

## Acknowledgements

Dr. Michael Nolan for supervision; Ji Liu for the reference VASP simulations and data used in the nitride comparison; the Ada HPC cluster; Matlantis for access to PFP.

## Reference

Ji et al., *J. Phys. Chem. C* **2025**, 129, 11173-11182.
