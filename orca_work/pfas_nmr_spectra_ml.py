"""
pfas_nmr_spectra_ml.py
──────────────────────
ML pipeline for ¹⁹F NMR spectra using LOO-CV (Leave-One-Out Cross-Validation).

Three prediction tasks (analogous to existing IR tasks):

  Task A │ RDKit descriptors  →  ¹⁹F NMR spectrum      (Ridge + PCA)
  Task B │ IR spectrum        →  ¹⁹F NMR spectrum      (Ridge + PCA)  [cross-modal]
  Task C │ ¹⁹F NMR spectrum  →  IR spectrum            (Ridge + PCA)  [reverse cross-modal]

Inputs
------
  pfas_nmr_spectra.npz      – output of build_pfas_nmr_spectra.py
  pfas_combined_dataset.npz – X_ir, mol_names
  PFAS_smiles.csv           – for RDKit descriptor generation (Task A)

Usage
-----
    python pfas_nmr_spectra_ml.py \
        --nmr_spectra pfas_nmr_spectra.npz \
        --combined    pfas_combined_dataset.npz \
        --smiles_csv  PFAS_smiles.csv \
        --output_dir  results/nmr_spectra_ml/
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# ── try RDKit ──────────────────────────────────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.ML.Descriptors import MoleculeDescriptors
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    print("[WARN] RDKit not found – Task A will be skipped.")


# ── metrics ───────────────────────────────────────────────────────────────────

def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean per-sample Pearson r between true and predicted spectra."""
    rs = []
    for t, p in zip(y_true, y_pred):
        if t.std() == 0 or p.std() == 0:
            rs.append(0.0)
        else:
            rs.append(pearsonr(t, p)[0])
    return float(np.mean(rs))


def cosine_sim(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean per-sample cosine similarity."""
    sims = []
    for t, p in zip(y_true, y_pred):
        denom = np.linalg.norm(t) * np.linalg.norm(p)
        sims.append(float(np.dot(t, p) / denom) if denom > 0 else 0.0)
    return float(np.mean(sims))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ── LOO-CV with Ridge + PCA pipeline ──────────────────────────────────────────

def loo_predict(X: np.ndarray, Y: np.ndarray,
                n_components: int = 30,
                alpha: float = 1.0) -> np.ndarray:
    """
    Leave-One-Out cross-validated predictions.

    Pipeline: StandardScaler → PCA(n_components) → Ridge(alpha)

    Returns Y_pred with shape == Y.shape.
    """
    loo = LeaveOneOut()
    Y_pred = np.zeros_like(Y, dtype=np.float32)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=min(n_components, X.shape[0] - 1, X.shape[1]))),
        ("ridge",  Ridge(alpha=alpha)),
    ])

    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        Y_tr        = Y[train_idx]
        pipe.fit(X_tr, Y_tr)
        Y_pred[test_idx] = pipe.predict(X_te).astype(np.float32)

    return Y_pred


# ── RDKit descriptor matrix ───────────────────────────────────────────────────

def build_descriptor_matrix(smiles_list: list[str]) -> np.ndarray:
    """Compute all 200 RDKit 2D descriptors for each SMILES."""
    desc_names = [d[0] for d in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rows.append([np.nan] * len(desc_names))
        else:
            rows.append(list(calc.CalcDescriptors(mol)))
    X = np.array(rows, dtype=np.float32)
    # Replace inf / nan with column median
    for col in range(X.shape[1]):
        bad = ~np.isfinite(X[:, col])
        if bad.any():
            med = np.nanmedian(X[~bad, col]) if (~bad).any() else 0.0
            X[bad, col] = med
    return X


# ── plotting helpers ──────────────────────────────────────────────────────────

def plot_example_spectra(grid, Y_true, Y_pred, mol_names, n=6, title="", outpath=""):
    """Plot n overlay examples of true vs predicted spectra."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()
    indices = np.random.choice(len(Y_true), size=min(n, len(Y_true)), replace=False)
    for ax, idx in zip(axes, indices):
        ax.plot(grid, Y_true[idx], label="True",  color="steelblue",  linewidth=1.5)
        ax.plot(grid, Y_pred[idx], label="Predicted", color="tomato", linewidth=1.5, linestyle="--")
        ax.set_title(mol_names[idx][:35], fontsize=8)
        ax.set_xlabel("δ (ppm)" if "NMR" in title else "Wavenumber (cm⁻¹)", fontsize=7)
        ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")


def plot_scatter(r_values: dict, title: str, outpath: str):
    """Bar chart of per-task Pearson r scores."""
    labels = list(r_values.keys())
    values = list(r_values.values())
    colours = ["#4C72B0", "#DD8452", "#55A868"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colours[:len(labels)])
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Mean Pearson r")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {outpath}")


# ── main ──────────────────────────────────────────────────────────────────────

def run(nmr_spectra_path: str,
        combined_path:    str,
        smiles_csv_path:  str,
        output_dir:       str) -> None:

    os.makedirs(output_dir, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading data …")
    nmr_data  = np.load(nmr_spectra_path, allow_pickle=True)
    comb_data = np.load(combined_path,    allow_pickle=True)

    Y_nmr      = nmr_data["spectra"].astype(np.float32)    # (68, 251)
    grid_nmr   = nmr_data["grid_ppm"]                      # (251,)
    mol_names  = comb_data["mol_names"]

    X_ir       = comb_data["X_ir"].astype(np.float32)      # (68, 3501)
    n_mols     = X_ir.shape[0]

    print(f"  Molecules : {n_mols}")
    print(f"  IR shape  : {X_ir.shape}")
    print(f"  NMR shape : {Y_nmr.shape}\n")

    summary_rows = []

    # ════════════════════════════════════════════════════════════════════════════
    # Task A: RDKit descriptors → ¹⁹F NMR spectrum
    # ════════════════════════════════════════════════════════════════════════════
    if RDKIT_OK and os.path.isfile(smiles_csv_path):
        print("─" * 60)
        print("Task A: RDKit descriptors → ¹⁹F NMR spectrum (LOO-CV)")
        print("─" * 60)

        smiles_df = pd.read_csv(smiles_csv_path)
        # Align SMILES to mol_names order
        name_col  = smiles_df.columns[0]
        smi_col   = smiles_df.columns[1]
        name2smi  = dict(zip(smiles_df[name_col], smiles_df[smi_col]))
        smiles_list = [name2smi.get(n, "") for n in mol_names]

        X_desc = build_descriptor_matrix(smiles_list)
        print(f"  Descriptor matrix: {X_desc.shape}")

        Y_pred_A = loo_predict(X_desc, Y_nmr, n_components=30, alpha=1.0)

        r_A    = pearson_r(Y_nmr, Y_pred_A)
        cos_A  = cosine_sim(Y_nmr, Y_pred_A)
        rmse_A = rmse(Y_nmr, Y_pred_A)
        print(f"\n  Task A results (LOO-CV):")
        print(f"    Mean Pearson r   : {r_A:.4f}")
        print(f"    Mean cosine sim  : {cos_A:.4f}")
        print(f"    RMSE             : {rmse_A:.6f}")
        summary_rows.append({"Task": "A: Descriptors→NMR", "Pearson r": r_A,
                              "Cosine sim": cos_A, "RMSE": rmse_A})

        plot_example_spectra(
            grid_nmr, Y_nmr, Y_pred_A, mol_names,
            title="Task A: RDKit Descriptors → ¹⁹F NMR Spectrum (LOO-CV)",
            outpath=os.path.join(output_dir, "taskA_examples.png"),
        )
    else:
        print("Task A skipped (RDKit unavailable or SMILES CSV missing).")

    # ════════════════════════════════════════════════════════════════════════════
    # Task B: IR spectrum → ¹⁹F NMR spectrum
    # ════════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("Task B: IR spectrum → ¹⁹F NMR spectrum (LOO-CV)")
    print("─" * 60)
    print("  [Expected to perform poorly — ¹⁹F NMR is orthogonal to IR]\n")

    Y_pred_B = loo_predict(X_ir, Y_nmr, n_components=30, alpha=1.0)

    r_B    = pearson_r(Y_nmr, Y_pred_B)
    cos_B  = cosine_sim(Y_nmr, Y_pred_B)
    rmse_B = rmse(Y_nmr, Y_pred_B)
    print(f"  Task B results (LOO-CV):")
    print(f"    Mean Pearson r   : {r_B:.4f}")
    print(f"    Mean cosine sim  : {cos_B:.4f}")
    print(f"    RMSE             : {rmse_B:.6f}")
    summary_rows.append({"Task": "B: IR→NMR", "Pearson r": r_B,
                          "Cosine sim": cos_B, "RMSE": rmse_B})

    plot_example_spectra(
        grid_nmr, Y_nmr, Y_pred_B, mol_names,
        title="Task B: IR → ¹⁹F NMR Spectrum (LOO-CV) [cross-modal]",
        outpath=os.path.join(output_dir, "taskB_examples.png"),
    )

    # ════════════════════════════════════════════════════════════════════════════
    # Task C: ¹⁹F NMR spectrum → IR spectrum
    # ════════════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 60)
    print("Task C: ¹⁹F NMR spectrum → IR spectrum (LOO-CV)")
    print("─" * 60)
    print("  [Reverse cross-modal — also expected to perform poorly]\n")

    # IR wavenumber grid (500–4000, 3501 pts)
    grid_ir = np.linspace(500, 4000, X_ir.shape[1])

    Y_pred_C = loo_predict(Y_nmr, X_ir, n_components=30, alpha=1.0)

    r_C    = pearson_r(X_ir, Y_pred_C)
    cos_C  = cosine_sim(X_ir, Y_pred_C)
    rmse_C = rmse(X_ir, Y_pred_C)
    print(f"  Task C results (LOO-CV):")
    print(f"    Mean Pearson r   : {r_C:.4f}")
    print(f"    Mean cosine sim  : {cos_C:.4f}")
    print(f"    RMSE             : {rmse_C:.6f}")
    summary_rows.append({"Task": "C: NMR→IR", "Pearson r": r_C,
                          "Cosine sim": cos_C, "RMSE": rmse_C})

    plot_example_spectra(
        grid_ir, X_ir, Y_pred_C, mol_names,
        title="Task C: ¹⁹F NMR → IR Spectrum (LOO-CV) [reverse cross-modal]",
        outpath=os.path.join(output_dir, "taskC_examples.png"),
    )

    # ════════════════════════════════════════════════════════════════════════════
    # Summary
    # ════════════════════════════════════════════════════════════════════════════
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "nmr_spectra_ml_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'═'*60}")
    print("SUMMARY (LOO-CV)")
    print("═"*60)
    print(summary_df.to_string(index=False))
    print(f"\nSaved CSV → {summary_path}")

    # Bar chart
    r_dict = {row["Task"]: row["Pearson r"] for row in summary_rows}
    plot_scatter(
        r_dict,
        title="¹⁹F NMR Spectra ML – LOO-CV Mean Pearson r",
        outpath=os.path.join(output_dir, "summary_pearson_r.png"),
    )

    # ── Scientific interpretation ─────────────────────────────────────────────
    print("""
SCIENTIFIC INTERPRETATION
─────────────────────────
Task A (Descriptors → ¹⁹F NMR):
  Analogous to IR Task 3 (r=0.913). If r is similar, molecular topology
  encoded in RDKit descriptors is a sufficient proxy for ¹⁹F shielding
  patterns. If r is lower, ¹⁹F NMR encodes finer electronic structure
  detail that 2D descriptors miss.

Task B (IR → ¹⁹F NMR):
  Expected poor performance (r ~ 0.15 from Task 4 summary feature analysis).
  Directly confirms ¹⁹F NMR orthogonality to IR — the two spectroscopic
  probes sample non-redundant molecular information.

Task C (¹⁹F NMR → IR):
  Reverse probe. If also poor, it confirms the orthogonality is symmetric:
  neither modality is a surrogate for the other. This is the key scientific
  result justifying combined IR + ¹⁹F NMR for real-world PFAS identification.
""")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="¹⁹F NMR spectra ML pipeline (Tasks A/B/C)"
    )
    parser.add_argument("--nmr_spectra", default="pfas_nmr_spectra.npz",
                        help="Output of build_pfas_nmr_spectra.py")
    parser.add_argument("--combined",    default="pfas_combined_dataset.npz",
                        help="Combined IR+NMR dataset (for X_ir)")
    parser.add_argument("--smiles_csv",  default="PFAS_smiles.csv",
                        help="CSV with columns: name, SMILES  (for Task A)")
    parser.add_argument("--output_dir",  default="results/nmr_spectra_ml",
                        help="Directory to write plots and summary CSV")
    args = parser.parse_args()

    run(args.nmr_spectra, args.combined, args.smiles_csv, args.output_dir)
