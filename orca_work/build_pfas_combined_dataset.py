"""
build_pfas_combined_dataset.py
==============================
Merges the IR dataset (pfas_ir_dataset.npz) and NMR dataset (pfas_nmr_dataset.npz)
into a single aligned feature matrix for ML.

NMR features used:
    - Mean 19F shift (1 feature) - most informative for PFAS
    - Std  19F shift (1 feature) - encodes chain environment diversity
    - Min/max 19F shift (2 features) - chain-end vs chain-middle F
    - n_F count (1 feature)
    - Mean 1H shift (1 feature, NaN->0 for perfluorinated)
    - n_H count (1 feature)
    - Mean 13C shift (1 feature)
    - n_C count (1 feature)
    Total: 9 NMR features per molecule

IR features: full 3501-point spectra (500-4000 cm⁻¹) - same as before,
             or PCA-reduced if you prefer (set USE_PCA=True).

Outputs
-------
    ml_dataset/pfas_combined_dataset.npz
        mol_names    : (N,)       molecule names (intersection of IR ∩ NMR)
        X_ir         : (N, 3501)  IR spectra
        X_nmr        : (N, 9)     NMR feature vector
        X_combined   : (N, 3510)  concatenated IR + NMR
        nmr_feature_names : (9,)  names of the 9 NMR features

Usage
-----
    python build_pfas_combined_dataset.py
"""

import os
import numpy as np
import pandas as pd

# PATHS

ML_DIR       = r"C:\Users\danie\Downloads\orca_work\ml_dataset"
IR_NPZ       = os.path.join(ML_DIR, "pfas_ir_dataset.npz")
NMR_NPZ      = os.path.join(ML_DIR, "pfas_nmr_dataset.npz")
OUT_NPZ      = os.path.join(ML_DIR, "pfas_combined_dataset.npz")

# LOAD DATASETS

def load_ir(path):
    d = np.load(path, allow_pickle=True)
    # pfas_ir_dataset.npz keys: wavenumbers, spectra, smiles, filenames, csv_names
    names   = d["csv_names"].astype(str)
    spectra = d["spectra"]
    return names, spectra


def load_nmr(path):
    d = np.load(path, allow_pickle=True)
    names    = d["mol_names"].astype(str)
    F_shifts = d["F_shifts"]   # (N, max_F)
    H_shifts = d["H_shifts"]   # (N, max_H)
    C_shifts = d["C_shifts"]   # (N, max_C)
    n_F      = d["n_F"]
    n_H      = d["n_H"]
    n_C      = d["n_C"]
    return names, F_shifts, H_shifts, C_shifts, n_F, n_H, n_C

# BUILD NMR FEATURE VECTOR

def build_nmr_features(F_shifts, H_shifts, C_shifts, n_F, n_H, n_C):
    """
    Collapse per-atom shieldings into a fixed-length feature vector per molecule.
    NaN atoms (padding) are excluded from statistics via np.nanmean etc.
    Missing nuclei (n=0) get 0.0 - the model can learn this as 'absent'.
    """
    N = F_shifts.shape[0]
    features = np.zeros((N, 9), dtype=np.float32)

    for i in range(N):
        f = F_shifts[i, :n_F[i]] if n_F[i] > 0 else np.array([np.nan])
        h = H_shifts[i, :n_H[i]] if n_H[i] > 0 else np.array([np.nan])
        c = C_shifts[i, :n_C[i]] if n_C[i] > 0 else np.array([np.nan])

        features[i, 0] = np.nanmean(f) if n_F[i] > 0 else 0.0   # mean 19F
        features[i, 1] = np.nanstd(f)  if n_F[i] > 1 else 0.0   # std  19F
        features[i, 2] = np.nanmin(f)  if n_F[i] > 0 else 0.0   # min  19F
        features[i, 3] = np.nanmax(f)  if n_F[i] > 0 else 0.0   # max  19F
        features[i, 4] = float(n_F[i])                            # count F
        features[i, 5] = np.nanmean(h) if n_H[i] > 0 else 0.0   # mean 1H
        features[i, 6] = float(n_H[i])                            # count H
        features[i, 7] = np.nanmean(c) if n_C[i] > 0 else 0.0   # mean 13C
        features[i, 8] = float(n_C[i])                            # count C

    feature_names = np.array([
        "F_mean_ppm", "F_std_ppm", "F_min_ppm", "F_max_ppm", "n_F",
        "H_mean_ppm", "n_H",
        "C_mean_ppm", "n_C",
    ])
    return features, feature_names

# ALIGN DATASETS ON MOLECULE NAMES

def normalise_name(name):
    """
    Reduce a molecule name to a lowercase alphanumeric key for fuzzy matching.
    Strips all punctuation, spaces, underscores, parentheses so that:
      'PFOA (Perfluorooctanoic acid)'  -> 'pfooperfluorooctanoicacid'
      'PFOA_Perfluorooctanoic_acid'    -> 'pfooperfluorooctanoicacid'
    """
    import re
    return re.sub(r'[^a-z0-9]', '', name.lower())


def align_datasets(ir_names, ir_spectra, nmr_names, nmr_features):
    """
    Align IR and NMR datasets by normalising molecule names to a common key,
    then taking the intersection. Handles mismatched formatting between datasets
    (e.g. IR uses 'PFOA (Perfluorooctanoic acid)', NMR uses 'PFOA_Perfluorooctanoic_acid').

    For cases where one name is a prefix of the other (e.g. '82dipap' vs
    '82dipapfluorotelomephosphatediester'), we match on prefix to catch them.
    """
    ir_norm  = {normalise_name(n): (i, n) for i, n in enumerate(ir_names)}
    nmr_norm = {normalise_name(n): (i, n) for i, n in enumerate(nmr_names)}

    # Exact matches first
    matched_ir  = {}  # ir_key  -> nmr_key
    matched_nmr = {}  # nmr_key -> ir_key

    for ik in ir_norm:
        if ik in nmr_norm:
            matched_ir[ik] = ik
            matched_nmr[ik] = ik

    # Prefix matches for anything not yet matched
    unmatched_ir  = [k for k in ir_norm  if k not in matched_ir]
    unmatched_nmr = [k for k in nmr_norm if k not in matched_nmr]

    for ik in unmatched_ir:
        for nk in unmatched_nmr:
            if ik.startswith(nk) or nk.startswith(ik):
                matched_ir[ik]  = nk
                matched_nmr[nk] = ik
                break

    only_ir  = sorted(k for k in ir_norm  if k not in matched_ir)
    only_nmr = sorted(k for k in nmr_norm if k not in matched_nmr)

    if only_ir:
        names = [ir_norm[k][1] for k in only_ir]
        print(f"  In IR but not NMR ({len(only_ir)}): {names}")
    if only_nmr:
        names = [nmr_norm[k][1] for k in only_nmr]
        print(f"  In NMR but not IR ({len(only_nmr)}): {names}")

    common_ir_keys = sorted(matched_ir.keys())
    print(f"  Intersection: {len(common_ir_keys)} molecules")

    # Use IR name as the canonical name in output
    mol_names   = np.array([ir_norm[ik][1]                       for ik in common_ir_keys])
    aligned_ir  = np.array([ir_spectra[ir_norm[ik][0]]           for ik in common_ir_keys])
    aligned_nmr = np.array([nmr_features[nmr_norm[matched_ir[ik]][0]] for ik in common_ir_keys])
    return mol_names, aligned_ir, aligned_nmr

# MAIN

def main():
    print("Loading IR dataset...")
    ir_names, ir_spectra = load_ir(IR_NPZ)
    print(f"  {len(ir_names)} molecules, spectra shape {ir_spectra.shape}")

    print("Loading NMR dataset...")
    nmr_names, F_shifts, H_shifts, C_shifts, n_F, n_H, n_C = load_nmr(NMR_NPZ)
    print(f"  {len(nmr_names)} molecules")

    print("Building NMR feature vectors...")
    nmr_features, feat_names = build_nmr_features(
        F_shifts, H_shifts, C_shifts, n_F, n_H, n_C
    )

    print("Aligning datasets...")
    mol_names, X_ir, X_nmr = align_datasets(
        ir_names, ir_spectra, nmr_names, nmr_features
    )

    # Normalise NMR features (z-score) before concatenation
    # IR spectra are already on [0,1] from Lorentzian broadening
    nmr_mean = X_nmr.mean(axis=0)
    nmr_std  = X_nmr.std(axis=0)
    nmr_std[nmr_std == 0] = 1.0   # avoid divide-by-zero for constant features
    X_nmr_scaled = (X_nmr - nmr_mean) / nmr_std

    X_combined = np.concatenate([X_ir, X_nmr_scaled], axis=1)

    print(f"\nCombined feature matrix: {X_combined.shape}")
    print(f"  IR features:  {X_ir.shape[1]}")
    print(f"  NMR features: {X_nmr.shape[1]}  ({', '.join(feat_names)})")

    np.savez(
        OUT_NPZ,
        mol_names         = mol_names,
        X_ir              = X_ir,
        X_nmr             = X_nmr,
        X_nmr_scaled      = X_nmr_scaled,
        X_combined        = X_combined,
        nmr_feature_names = feat_names,
        nmr_mean          = nmr_mean,
        nmr_std           = nmr_std,
    )
    print(f"\nSaved combined dataset -> {OUT_NPZ}")

    # Quick CSV preview
    df = pd.DataFrame(X_nmr, columns=feat_names, index=mol_names)
    df.index.name = "molecule"
    preview_path = os.path.join(ML_DIR, "pfas_nmr_features.csv")
    df.to_csv(preview_path, float_format="%.3f")
    print(f"Saved NMR feature table -> {preview_path}")

    print("\n--- NMR feature summary ---")
    print(df.describe().round(2).to_string())


if __name__ == "__main__":
    main()
