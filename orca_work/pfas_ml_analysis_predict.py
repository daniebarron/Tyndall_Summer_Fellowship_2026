"""
pfas_ml_analysis_predict.py
============================
Extended version of pfas_ml_analysis_new.py with a --predict mode.

Normal mode (no --predict flag):
    Runs Tasks 1, 2, 3 exactly as before (LOO-CV on the 68 training molecules).

Prediction mode (--predict):
    1. Retrains the Task 3 Ridge+PCA model on ALL 68 molecules.
    2. Prompts you to enter one or more SMILES strings for new molecules.
    3. Predicts the IR spectrum for each and saves a plot.
    4. Optionally overlays the actual DFT spectrum if you supply a .out file.

Usage
-----
    # Normal evaluation (Tasks 1-3, LOO-CV):
    python pfas_ml_analysis_predict.py

    # Predict for new molecules interactively:
    python pfas_ml_analysis_predict.py --predict

    # Predict with a specific SMILES and name (non-interactive):
    python pfas_ml_analysis_predict.py --predict ^
        --smiles "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)CCS(=O)(=O)O" ^
        --mol_name "6:2 FTSA"

    # Predict and compare against DFT .out file:
    python pfas_ml_analysis_predict.py --predict ^
        --smiles "FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)CCS(=O)(=O)O" ^
        --mol_name "6:2 FTSA" ^
        --out_file "6_2_FTSA_ir.out"
"""

import os
import re
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import balanced_accuracy_score

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    print("WARNING: RDKit not available — Task 3 and prediction mode will be skipped")

# =============================================================================
# PATHS  — update these to match your machine
# =============================================================================

NPZ_PATH   = r"C:\Users\danie\Downloads\orca_work\ml_dataset\pfas_ir_dataset.npz"
OUTPUT_DIR = r"C:\Users\danie\Downloads\orca_work\ml_dataset"

# Scale factor applied to raw ORCA frequencies when building the IR dataset
SCALE_FACTOR = 0.9614
FREQ_MIN     = 500.0
FREQ_MAX     = 4000.0
# Lorentzian broadening FWHM used when building the dataset (cm-1)
LORENTZ_FWHM = 20.0

# =============================================================================
# DESCRIPTOR FUNCTION  (identical to pfas_ml_analysis_new.py)
# =============================================================================

def get_descriptors(smi):
    """
    Compute the 18 PFAS-relevant RDKit descriptors used in Task 3.
    Returns a dict, or None if the SMILES is invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
    except Exception:
        return None

    d = {}
    d["mol_weight"]  = Descriptors.MolWt(mol)
    d["n_atoms"]     = mol.GetNumAtoms()
    d["n_heavy"]     = mol.GetNumHeavyAtoms()
    d["n_F"]         = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
    d["n_C"]         = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
    d["n_O"]         = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
    d["n_S"]         = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16)
    d["n_N"]         = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
    d["n_P"]         = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 15)
    d["n_Cl"]        = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 17)
    d["frac_F"]      = d["n_F"] / max(d["n_heavy"], 1)
    d["n_rings"]     = rdMolDescriptors.CalcNumRings(mol)
    d["n_rot_bonds"] = rdMolDescriptors.CalcNumRotatableBonds(mol)
    d["tpsa"]        = Descriptors.TPSA(mol)
    d["logp"]        = Descriptors.MolLogP(mol)
    cf2_pat          = Chem.MolFromSmarts("[#6]([F])([F])")
    cf3_pat          = Chem.MolFromSmarts("[#6]([F])([F])[F]")
    d["n_CF2"]       = len(mol.GetSubstructMatches(cf2_pat)) if cf2_pat else 0
    d["n_CF3"]       = len(mol.GetSubstructMatches(cf3_pat)) if cf3_pat else 0
    d["n_CF2_chain"] = d["n_CF2"] - d["n_CF3"]
    return d


# =============================================================================
# ORCA .out PARSER  (for optional DFT comparison)
# =============================================================================

def parse_orca_ir(out_path, wn_grid, fwhm=LORENTZ_FWHM, scale=SCALE_FACTOR):
    """
    Parse vibrational frequencies and IR intensities from an ORCA .out file,
    then broaden onto wn_grid using a Lorentzian with the given FWHM.
    Returns a normalised spectrum array (same length as wn_grid), or None on failure.

    Uses the same line-by-line approach as build_pfas_ir_dataset.py, which is
    robust to blank lines and header variations inside the IR SPECTRUM block.
    """
    try:
        with open(out_path, encoding="utf-8", errors="ignore") as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        print(f"  [WARN] .out file not found: {out_path}")
        return None

    freqs, intensities = [], []
    in_ir_block = False
    found_data  = False

    for line in lines:
        if "IR SPECTRUM" in line and "YES" not in line:
            in_ir_block = True
            found_data  = False
            continue
        if not in_ir_block:
            continue
        stripped = line.strip()
        if stripped.startswith("-") or stripped.startswith("Mode") or stripped.startswith("cm"):
            continue
        if stripped == "":
            if found_data:
                break
            continue
        m = re.match(r"\s*(\d+):\s+([\d.]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)", line)
        if m:
            freq      = float(m.group(2)) * scale
            intensity = float(m.group(4))
            if freq > 0:
                freqs.append(freq)
                intensities.append(intensity)
                found_data = True

    if not freqs:
        print(f"  [WARN] No frequencies/intensities parsed from {out_path}")
        return None

    freqs       = np.array(freqs)
    intensities = np.array(intensities)

    # Lorentzian broadening onto grid
    gamma    = fwhm / 2.0
    spectrum = np.zeros(len(wn_grid))
    for freq, inten in zip(freqs, intensities):
        if FREQ_MIN <= freq <= FREQ_MAX:
            spectrum += inten * gamma / (np.pi * ((wn_grid - freq) ** 2 + gamma ** 2))

    # Normalise to [0, 1]
    peak = spectrum.max()
    if peak > 0:
        spectrum /= peak
    return spectrum.astype(np.float32)


# =============================================================================
# TRAIN FULL MODEL ON 68 MOLECULES
# =============================================================================

def train_model(X_spec, X_desc, N_PC=15, alpha=10.0):
    """
    Fit the Ridge+PCA model on the full training set.
    Returns (pca, scaler, ridge_models) where ridge_models is a list of N_PC
    fitted Ridge regressors, one per PC.
    """
    pca    = PCA(n_components=N_PC, random_state=42)
    Y_pca  = pca.fit_transform(X_spec)

    scaler = StandardScaler()
    Xsc    = scaler.fit_transform(X_desc)

    ridges = []
    for pc in range(N_PC):
        r = Ridge(alpha=alpha)
        r.fit(Xsc, Y_pca[:, pc])
        ridges.append(r)

    return pca, scaler, ridges


def predict_spectrum(smi, pca, scaler, ridges, desc_columns):
    """
    Predict the IR spectrum for a single SMILES string.
    Returns the predicted spectrum (normalised), or None if descriptors fail.
    """
    d = get_descriptors(smi)
    if d is None:
        return None

    # Ensure same column order as training
    row   = np.array([d.get(col, 0.0) for col in desc_columns], dtype=np.float64)
    row_sc = scaler.transform(row.reshape(1, -1))

    pca_coords = np.array([[r.predict(row_sc)[0] for r in ridges]])
    spectrum   = pca.inverse_transform(pca_coords)[0]
    spectrum   = np.clip(spectrum, 0, None)
    peak       = spectrum.max()
    if peak > 0:
        spectrum /= peak
    return spectrum.astype(np.float32)


def find_nearest_neighbours(pred_spec, X_train, names_train, n=3):
    """Return the top-n nearest training molecules by cosine similarity."""
    sims = cosine_similarity(pred_spec.reshape(1, -1), X_train)[0]
    top  = np.argsort(sims)[::-1][:n]
    return [(names_train[i], float(sims[i])) for i in top]


# =============================================================================
# PREDICTION MODE
# =============================================================================

def run_prediction_mode(args, data, wn, names, smiles_train,
                        X_spec, X_desc, desc_columns):
    """
    Train on all 68, then predict IR spectrum for one or more new molecules.
    """
    print("\n" + "=" * 60)
    print("PREDICTION MODE — Training on all 68 molecules...")
    print("=" * 60)

    N_PC  = 15
    alpha = 10.0

    pca, scaler, ridges = train_model(X_spec, X_desc, N_PC=N_PC, alpha=alpha)

    var_exp = pca.explained_variance_ratio_.cumsum()
    print(f"  PCA: {N_PC} components explain {var_exp[-1]*100:.1f}% of spectral variance")
    print(f"  Model trained. Ready to predict.\n")

    # ── Collect molecules to predict ─────────────────────────────────────────
    molecules = []  # list of (name, smiles, out_file_or_None)

    if args.smiles:
        # Non-interactive: passed on command line
        mol_name = args.mol_name if args.mol_name else "New molecule"
        molecules.append((mol_name, args.smiles, args.out_file))
    else:
        # Interactive: prompt until user is done
        print("Enter molecules to predict (leave SMILES blank to finish):\n")
        while True:
            smi  = input("  SMILES: ").strip()
            if not smi:
                break
            name = input("  Name (optional, press Enter to skip): ").strip()
            if not name:
                name = f"Molecule {len(molecules)+1}"
            out  = input("  ORCA .out file path (optional, press Enter to skip): ").strip()
            if not out:
                out = None
            molecules.append((name, smi, out))
            print()

    if not molecules:
        print("No molecules entered. Exiting prediction mode.")
        return

    # ── Predict each molecule ─────────────────────────────────────────────────
    print(f"\nPredicting {len(molecules)} molecule(s)...\n")

    results = []
    for mol_name, smi, out_file in molecules:
        print(f"  {mol_name}")
        print(f"    SMILES: {smi}")

        # Validate SMILES
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"    [ERROR] Invalid SMILES — skipping\n")
            continue

        # Predict
        pred_spec = predict_spectrum(smi, pca, scaler, ridges, desc_columns)
        if pred_spec is None:
            print(f"    [ERROR] Descriptor calculation failed — skipping\n")
            continue

        # Nearest neighbours in training set
        nn = find_nearest_neighbours(pred_spec, X_spec, names, n=3)
        print(f"    Top-3 nearest training molecules (cosine sim):")
        for nn_name, nn_sim in nn:
            print(f"      {nn_sim:.4f}  {nn_name}")

        # Parse DFT .out if provided
        dft_spec = None
        if out_file and os.path.isfile(out_file):
            dft_spec = parse_orca_ir(out_file, wn)
            if dft_spec is not None:
                cos_sim  = cosine_similarity(pred_spec.reshape(1,-1),
                                             dft_spec.reshape(1,-1))[0,0]
                pearson  = np.corrcoef(pred_spec, dft_spec)[0,1]
                print(f"    Comparison vs DFT:")
                print(f"      Cosine similarity : {cos_sim:.4f}")
                print(f"      Pearson r         : {pearson:.4f}")
        elif out_file:
            print(f"    [WARN] .out file not found: {out_file}")

        results.append((mol_name, smi, pred_spec, dft_spec, nn))
        print()

    # ── Plot ──────────────────────────────────────────────────────────────────
    if not results:
        print("No valid predictions to plot.")
        return

    n_mols  = len(results)
    fig, axes = plt.subplots(n_mols, 1,
                             figsize=(13, 4 * n_mols),
                             squeeze=False)

    for ax, (mol_name, smi, pred_spec, dft_spec, nn) in zip(axes[:, 0], results):
        nn_name, nn_sim = nn[0]
        nn_idx  = names.index(nn_name)
        nn_spec = X_spec[nn_idx]

        # Nearest neighbour (shaded, background)
        ax.fill_between(wn, nn_spec, alpha=0.18, color="grey")
        ax.plot(wn, nn_spec, color="grey", linewidth=0.8, alpha=0.6,
                label=f"Nearest in DB: {nn_name.split('(')[0].strip()} (cos={nn_sim:.3f})")

        # ML prediction
        ax.plot(wn, pred_spec, color="#E63946", linewidth=1.8,
                label="ML predicted")

        # DFT (if available)
        if dft_spec is not None:
            cos_sim = cosine_similarity(pred_spec.reshape(1,-1),
                                        dft_spec.reshape(1,-1))[0,0]
            pearson = np.corrcoef(pred_spec, dft_spec)[0,1]
            ax.plot(wn, dft_spec, color="#2196F3", linewidth=1.5,
                    linestyle="--",
                    label=f"DFT (actual)  cos={cos_sim:.3f}  r={pearson:.3f}")

        ax.set_title(mol_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=9)
        ax.set_ylabel("Normalised absorbance", fontsize=9)
        ax.set_xlim(500, 4000)
        ax.set_ylim(-0.02, 1.12)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25, linewidth=0.5)

    fig.suptitle(
        f"ML-Predicted IR Spectra (Ridge+PCA, {N_PC} PCs, α={alpha})\n"
        f"Trained on 68 PFAS molecules  |  LOO-CV Pearson r ≈ 0.913",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_png = os.path.join(OUTPUT_DIR, "predicted_IR_new_molecules.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved → {out_png}")


# =============================================================================
# TASKS 1–3  (unchanged from pfas_ml_analysis_new.py)
# =============================================================================

def run_tasks(data, wn, names, smiles, X_spec, X_desc, desc_columns,
              X_nmr_raw=None, nmr_feat_names=None):

    results_lines = []
    comparison    = {}

    def rlog(msg=""):
        print(msg)
        results_lines.append(msg)

    # ── Task 1 ────────────────────────────────────────────────────────────────
    rlog()
    rlog("=" * 60)
    rlog("TASK 1 — Spectral Library Matching (Cosine Similarity)")
    rlog("=" * 60)

    sim = cosine_similarity(X_spec)
    nn_sims = []
    for i in range(len(X_spec)):
        s = sim[i].copy(); s[i] = -1
        nn_sims.append(s.max())
    mean_sim = np.mean(nn_sims)
    rlog(f"  Mean NN cosine similarity : {mean_sim:.4f}")
    rlog(f"  Min / Max : {np.min(nn_sims):.4f} / {np.max(nn_sims):.4f}")
    comparison["Task 1 — Library matching"] = {
        "Mean NN cosine sim": {"IR only": f"{mean_sim:.4f}"}
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    np.fill_diagonal(sim, 1)
    im = axes[0].imshow(sim, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    axes[0].set_title("Pairwise Cosine Similarity Matrix", fontsize=10)
    short = [n.split("(")[0].strip()[:12] for n in names]
    axes[0].set_xticks(range(len(names))); axes[0].set_xticklabels(short, rotation=90, fontsize=3.5)
    axes[0].set_yticks(range(len(names))); axes[0].set_yticklabels(short, fontsize=3.5)
    plt.colorbar(im, ax=axes[0])
    axes[1].hist(nn_sims, bins=20, color="steelblue", edgecolor="white")
    axes[1].axvline(mean_sim, color="red", linestyle="--", label=f"Mean={mean_sim:.3f}")
    axes[1].set_xlabel("Cosine similarity to nearest neighbour")
    axes[1].set_ylabel("Count"); axes[1].set_title("NN Similarity Distribution (LOO)")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "task1_library_matching.png"), dpi=150, bbox_inches="tight")
    plt.close()
    rlog("  Saved: task1_library_matching.png")

    # ── Task 2 ────────────────────────────────────────────────────────────────
    rlog()
    rlog("=" * 60)
    rlog("TASK 2 — Functional Group Classification (RF + PCA, LOO)")
    rlog("=" * 60)

    # Assign functional group labels from SMILES
    def assign_fg(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return {}
        fg = {}
        fg["carboxylic_acid"] = int(mol.HasSubstructMatch(Chem.MolFromSmarts("C(=O)[OH]")))
        fg["sulfonic_acid"]   = int(mol.HasSubstructMatch(Chem.MolFromSmarts("S(=O)(=O)[OH]")))
        fg["sulfonamide"]     = int(mol.HasSubstructMatch(Chem.MolFromSmarts("S(=O)(=O)N")))
        fg["alcohol"]         = int(mol.HasSubstructMatch(Chem.MolFromSmarts("[CX4][OH]")))
        fg["ether"]           = int(mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]O[#6]")))
        return fg

    fg_list = [assign_fg(s) for s in smiles]
    all_fg  = sorted({k for d in fg_list for k in d})
    Y_fg    = np.array([[d.get(k, 0) for k in all_fg] for d in fg_list])

    N_PC_fg = 20
    pca_fg  = PCA(n_components=N_PC_fg, random_state=42)
    X_pca   = pca_fg.fit_transform(X_spec)
    loo     = LeaveOneOut()

    fg_scores = {}
    for fi, fg in enumerate(all_fg):
        y = Y_fg[:, fi]
        if y.sum() < 2 or (len(y) - y.sum()) < 2:
            continue
        clf  = RandomForestClassifier(n_estimators=200, random_state=42)
        pred = cross_val_predict(clf, X_pca, y, cv=loo)
        fg_scores[fg] = balanced_accuracy_score(y, pred)
        rlog(f"  {fg:20s}: balanced accuracy = {fg_scores[fg]:.3f}")

    comparison["Task 2 — FG classification"] = {
        fg: {"IR only": f"{sc:.4f}"} for fg, sc in fg_scores.items()
    }

    fig, ax = plt.subplots(figsize=(9, 4))
    fgs = list(fg_scores.keys())
    ax.barh(fgs, [fg_scores[f] for f in fgs], color="steelblue", edgecolor="white")
    ax.axvline(0.5, color="red", linestyle="--", linewidth=0.8, label="Random (0.5)")
    ax.set_xlabel("Balanced accuracy (LOO-CV)")
    ax.set_title("Task 2: Functional Group Classification (RF + PCA, LOO)")
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "task2_functional_groups.png"), dpi=150, bbox_inches="tight")
    plt.close()
    rlog("  Saved: task2_functional_groups.png")

    # ── Task 3 ────────────────────────────────────────────────────────────────
    rlog()
    rlog("=" * 60)
    rlog("TASK 3 — Spectrum Prediction from Molecular Descriptors (LOO-CV)")
    rlog("=" * 60)

    N_PC  = 15
    pca3  = PCA(n_components=N_PC, random_state=42)
    Y_pca = pca3.fit_transform(X_spec)
    var_exp = pca3.explained_variance_ratio_.cumsum()
    rlog(f"  PCA: {N_PC} components → {var_exp[-1]*100:.1f}% variance explained")

    def eval_prediction(X_input, label):
        sc   = StandardScaler()
        Xsc  = sc.fit_transform(X_input)
        loo  = LeaveOneOut()
        Y_p  = np.zeros_like(Y_pca)
        for pc in range(N_PC):
            Y_p[:, pc] = cross_val_predict(Ridge(alpha=10.0), Xsc, Y_pca[:, pc], cv=loo)
        X_p = np.clip(pca3.inverse_transform(Y_p), 0, None)
        cos_s = np.array([cosine_similarity(X_spec[i:i+1], X_p[i:i+1])[0,0]
                          for i in range(len(X_spec))])
        prs   = np.array([np.corrcoef(X_spec[i], X_p[i])[0,1] for i in range(len(X_spec))])
        rlog(f"\n  [{label}]")
        rlog(f"    Mean cosine sim : {cos_s.mean():.4f} ± {cos_s.std():.4f}")
        rlog(f"    Mean Pearson r  : {prs.mean():.4f} ± {prs.std():.4f}")
        rlog(f"    Best : {names[prs.argmax()].split('(')[0].strip()}  r={prs.max():.3f}")
        rlog(f"    Worst: {names[prs.argmin()].split('(')[0].strip()}  r={prs.min():.3f}")
        return cos_s, prs, X_p

    cos_ir, r_ir, Xpred_ir = eval_prediction(X_desc, "RDKit descriptors")
    comparison["Task 3 — Spectrum prediction"] = {
        "Mean cosine sim": {"IR only": f"{cos_ir.mean():.4f}"},
        "Mean Pearson r":  {"IR only": f"{r_ir.mean():.4f}"},
    }

    # PCA variance + scatter
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(range(1, N_PC+1), pca3.explained_variance_ratio_*100, color="steelblue", edgecolor="white")
    axes[0].plot(range(1, N_PC+1), var_exp*100, "ro-", ms=4, label="Cumulative")
    axes[0].axhline(95, color="gray", linestyle="--", label="95%")
    axes[0].set_xlabel("PC"); axes[0].set_ylabel("Variance explained (%)"); axes[0].legend()
    axes[0].set_title("Spectral PCA")
    axes[1].scatter(cos_ir, r_ir, c="steelblue", s=40, alpha=0.7)
    axes[1].set_xlabel("Cosine sim"); axes[1].set_ylabel("Pearson r")
    axes[1].set_title("Task 3 Prediction Quality (LOO-CV)")
    for i in [r_ir.argmax(), r_ir.argmin()]:
        axes[1].annotate(names[i].split("(")[0].strip()[:15],
                         (cos_ir[i], r_ir[i]), fontsize=6, ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "task3_pca_variance.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Best/worst spectra panel
    order    = np.argsort(r_ir)
    show_idx = list(order[:3]) + list(order[-3:][::-1])
    labels_s = ["Worst 1", "Worst 2", "Worst 3", "Best 3", "Best 2", "Best 1"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, si, lbl in zip(axes.flatten(), show_idx, labels_s):
        pred_n = Xpred_ir[si] / max(Xpred_ir[si].max(), 1e-9)
        ax.plot(wn, X_spec[si], lw=1.0, color="steelblue", label="DFT", alpha=0.9)
        ax.plot(wn, pred_n,     lw=1.0, color="red",       label="Predicted",
                alpha=0.7, linestyle="--")
        ax.set_title(f"{lbl}: {names[si].split('(')[0].strip()[:20]}\nr={r_ir[si]:.3f}", fontsize=7)
        ax.set_xlim(500, 4000); ax.set_yticks([])
    axes.flatten()[0].legend(fontsize=7)
    fig.suptitle("Task 3: Spectrum Prediction (Ridge+PCA, LOO-CV)", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "task3_spectrum_prediction.png"), dpi=150, bbox_inches="tight")
    plt.close()
    rlog("  Saved: task3_pca_variance.png, task3_spectrum_prediction.png")

    # Save report
    report_path = os.path.join(OUTPUT_DIR, "pfas_ml_results.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(results_lines))
    rlog(f"\nResults saved to {report_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="PFAS IR ML analysis + prediction mode")
    parser.add_argument("--predict",  action="store_true",
                        help="Run in prediction mode (train on 68, predict new molecules)")
    parser.add_argument("--smiles",   type=str, default=None,
                        help="SMILES of new molecule to predict (non-interactive)")
    parser.add_argument("--mol_name", type=str, default=None,
                        help="Name of the new molecule (used in plot title)")
    parser.add_argument("--out_file", type=str, default=None,
                        help="Path to ORCA .out file for DFT comparison (optional)")
    parser.add_argument("--npz",      type=str, default=NPZ_PATH,
                        help="Path to pfas_ir_dataset.npz")
    parser.add_argument("--outdir",   type=str, default=None,
                        help="Directory for output plots/reports")
    args = parser.parse_args()

    global OUTPUT_DIR
    OUTPUT_DIR = args.outdir if args.outdir else OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("Loading dataset...")
    print("=" * 60)
    data   = np.load(args.npz, allow_pickle=True)
    X      = data['spectra'].astype(np.float64)
    wn     = data['wavenumbers'].astype(np.float64)
    names  = data['csv_names'].tolist()
    smiles = data['smiles'].tolist()
    print(f"  {len(names)} molecules, {X.shape[1]} wavenumber points")
    print(f"  Grid: {wn[0]:.0f} – {wn[-1]:.0f} cm⁻¹")

    if not RDKIT_OK:
        print("RDKit not available — cannot continue.")
        sys.exit(1)

    # ── Build descriptor matrix ───────────────────────────────────────────────
    desc_data, desc_idx = [], []
    for i, smi in enumerate(smiles):
        d = get_descriptors(smi)
        if d is not None:
            desc_data.append(d); desc_idx.append(i)

    desc_df     = pd.DataFrame(desc_data, index=desc_idx)
    X_desc      = desc_df.values.astype(np.float64)
    X_spec      = X[desc_idx]
    names_sub   = [names[i] for i in desc_idx]
    smiles_sub  = [smiles[i] for i in desc_idx]
    desc_columns = list(desc_df.columns)

    print(f"  Descriptor matrix: {X_desc.shape[0]} × {X_desc.shape[1]}")

    # ── Dispatch ──────────────────────────────────────────────────────────────
    if args.predict:
        run_prediction_mode(args, data, wn, names_sub, smiles_sub,
                            X_spec, X_desc, desc_columns)
    else:
        run_tasks(data, wn, names_sub, smiles_sub, X_spec, X_desc, desc_columns)


if __name__ == "__main__":
    main()
