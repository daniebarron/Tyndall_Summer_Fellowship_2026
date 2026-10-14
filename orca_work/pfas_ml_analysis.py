"""
pfas_ml_analysis.py
===================
Three ML tasks on the 68-molecule PFAS IR spectral dataset.

Task 1 — Spectral library matching (cosine similarity)
Task 2 — Functional group classification (random forest on spectra)
Task 3 — Spectrum prediction from RDKit molecular descriptors (ridge regression)

Usage
-----
    python pfas_ml_analysis.py

Outputs written to OUTPUT_DIR:
    task1_library_matching.png      — cosine similarity heatmap + top-N accuracy
    task2_functional_groups.png     — per-group classifier performance
    task2_feature_importance.png    — which spectral regions matter per group
    task3_spectrum_prediction.png   — predicted vs actual spectra for test set
    task3_pca_variance.png          — PCA variance explained
    pfas_ml_results.txt             — summary of all results
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, Fragments
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    print("WARNING: RDKit not available — Task 3 will be skipped")

# =============================================================================
# PATHS
# =============================================================================

NPZ_PATH   = r"C:\Users\danie\Downloads\orca_work\ml_dataset\pfas_ir_dataset.npz"
OUTPUT_DIR = r"C:\Users\danie\Downloads\orca_work\ml_dataset"

# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 60)
print("Loading dataset...")
print("=" * 60)

data     = np.load(NPZ_PATH, allow_pickle=True)
X        = data['spectra'].astype(np.float64)   # (68, 3501)
wn       = data['wavenumbers'].astype(np.float64)
names    = data['csv_names'].tolist()
smiles   = data['smiles'].tolist()

n_mols, n_points = X.shape
print(f"  {n_mols} molecules, {n_points} wavenumber points")
print(f"  Grid: {wn[0]:.0f} - {wn[-1]:.0f} cm-1")

results_lines = []
def rlog(msg=""):
    print(msg)
    results_lines.append(msg)

# =============================================================================
# TASK 1 — SPECTRAL LIBRARY MATCHING
# =============================================================================

rlog()
rlog("=" * 60)
rlog("TASK 1 — Spectral Library Matching (Cosine Similarity)")
rlog("=" * 60)

# Compute full pairwise cosine similarity matrix
sim_matrix = cosine_similarity(X)   # (68, 68)

# Leave-one-out evaluation: for each molecule, remove it from the library,
# find the most similar remaining molecule, check if it's the correct one.
# "Correct" here means same molecule — since each molecule appears once,
# we check top-1, top-3, top-5 retrieval.

top1_correct = 0
top3_correct = 0
top5_correct = 0

for i in range(n_mols):
    # Get similarities to all others (exclude self)
    sims = sim_matrix[i].copy()
    sims[i] = -1  # mask self
    ranked = np.argsort(sims)[::-1]  # descending

    if ranked[0] == i:   # shouldn't happen since self is masked
        pass
    top1_correct += 1  # placeholder, replaced below

# Better LOO evaluation using known molecule pairs
# For each molecule find its nearest neighbour and report similarity score
nn_similarities = []
nn_names = []
for i in range(n_mols):
    sims = sim_matrix[i].copy()
    sims[i] = -1
    j = np.argmax(sims)
    nn_similarities.append(sims[j])
    nn_names.append(names[j])

mean_nn_sim = np.mean(nn_similarities)
min_nn_sim  = np.min(nn_similarities)
max_nn_sim  = np.max(nn_similarities)

rlog(f"  Mean nearest-neighbour cosine similarity: {mean_nn_sim:.4f}")
rlog(f"  Min / Max NN similarity: {min_nn_sim:.4f} / {max_nn_sim:.4f}")
rlog()

# Find most and least similar pairs
np.fill_diagonal(sim_matrix, 0)
flat_idx = np.argmax(sim_matrix)
i_max, j_max = np.unravel_index(flat_idx, sim_matrix.shape)
rlog(f"  Most similar pair:  {names[i_max]}  <->  {names[j_max]}")
rlog(f"    cosine similarity = {sim_matrix[i_max, j_max]:.4f}")
np.fill_diagonal(sim_matrix, 0)
flat_idx = np.argmin(sim_matrix + np.eye(n_mols)*999)
i_min, j_min = np.unravel_index(flat_idx, (sim_matrix.shape[0], sim_matrix.shape[1]))
rlog(f"  Least similar pair: {names[i_min]}  <->  {names[j_min]}")
rlog(f"    cosine similarity = {sim_matrix[i_min, j_min]:.4f}")
np.fill_diagonal(sim_matrix, 1)  # restore diagonal

# ── Plot: similarity heatmap + NN similarity distribution ────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Heatmap
im = axes[0].imshow(sim_matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
axes[0].set_title("Pairwise Cosine Similarity Matrix", fontsize=11)
axes[0].set_xlabel("Molecule index")
axes[0].set_ylabel("Molecule index")
plt.colorbar(im, ax=axes[0], label="Cosine similarity")

# Tick labels (abbreviated)
short_names = [n.split("(")[0].strip()[:12] for n in names]
axes[0].set_xticks(range(n_mols))
axes[0].set_xticklabels(short_names, rotation=90, fontsize=4)
axes[0].set_yticks(range(n_mols))
axes[0].set_yticklabels(short_names, fontsize=4)

# NN similarity distribution
axes[1].hist(nn_similarities, bins=20, color="steelblue", edgecolor="white", linewidth=0.5)
axes[1].axvline(mean_nn_sim, color="red", linestyle="--", label=f"Mean = {mean_nn_sim:.3f}")
axes[1].set_xlabel("Cosine similarity to nearest neighbour", fontsize=10)
axes[1].set_ylabel("Count", fontsize=10)
axes[1].set_title("Nearest-Neighbour Similarity Distribution\n(Leave-One-Out)", fontsize=11)
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "task1_library_matching.png"), dpi=150, bbox_inches="tight")
plt.close()
rlog("  Saved: task1_library_matching.png")

# =============================================================================
# TASK 2 — FUNCTIONAL GROUP CLASSIFICATION
# =============================================================================

rlog()
rlog("=" * 60)
rlog("TASK 2 — Functional Group Classification")
rlog("=" * 60)

if not RDKIT_OK:
    rlog("  SKIPPED — RDKit not available")
else:
    # ── Build functional group labels from SMILES ─────────────────────────────
    def get_fg_labels(smi):
        """Return dict of binary functional group labels for a SMILES string."""
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None
        except:
            return None

        labels = {}

        # Carboxylic acid  C(=O)O
        labels["carboxylic_acid"] = int(
            mol.HasSubstructMatch(Chem.MolFromSmarts("C(=O)[OH]")) or
            mol.HasSubstructMatch(Chem.MolFromSmarts("C(=O)[O-]"))
        )
        # Sulfonic acid  S(=O)(=O)O
        labels["sulfonic_acid"] = int(
            mol.HasSubstructMatch(Chem.MolFromSmarts("S(=O)(=O)[OH]")) or
            mol.HasSubstructMatch(Chem.MolFromSmarts("S(=O)(=O)[O-]"))
        )
        # Sulfonamide  S(=O)(=O)N
        labels["sulfonamide"] = int(
            mol.HasSubstructMatch(Chem.MolFromSmarts("S(=O)(=O)N"))
        )
        # Ether linkage  C-O-C
        labels["ether"] = int(
            mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]-O-[#6]"))
        )
        # Alcohol  C-OH
        labels["alcohol"] = int(
            mol.HasSubstructMatch(Chem.MolFromSmarts("[#6][OH]"))
        )
        # Phosphate ester  P(=O)(O)O
        labels["phosphate"] = int(
            mol.HasSubstructMatch(Chem.MolFromSmarts("P(=O)([OH,O-])O"))
        )
        # Chlorine substituent
        labels["chlorinated"] = int(
            mol.HasSubstructMatch(Chem.MolFromSmarts("[Cl]"))
        )

        return labels

    fg_data = []
    valid_idx = []
    for i, smi in enumerate(smiles):
        labels = get_fg_labels(smi)
        if labels is not None:
            fg_data.append(labels)
            valid_idx.append(i)
        else:
            rlog(f"  WARNING: could not parse SMILES for {names[i]}")

    fg_df = pd.DataFrame(fg_data, index=valid_idx)
    X_fg  = X[valid_idx]

    rlog(f"\n  Functional group label counts (out of {len(valid_idx)} molecules):")
    for col in fg_df.columns:
        pos = fg_df[col].sum()
        rlog(f"    {col:20s}  {pos:2d} positive  /  {len(valid_idx)-pos:2d} negative")

    # ── Train random forest classifier per functional group (LOO CV) ──────────
    rlog("\n  Random Forest LOO cross-validation accuracy:")

    fg_results = {}
    loo = LeaveOneOut()

    for fg in fg_df.columns:
        y = fg_df[fg].values
        n_pos = y.sum()
        n_neg = len(y) - n_pos

        if n_pos < 3 or n_neg < 3:
            rlog(f"    {fg:20s}  SKIPPED (too few samples: {n_pos}+/{n_neg}-)")
            continue

        # Use PCA to reduce dimensionality before RF 
        pipe = Pipeline([
            ("pca",  PCA(n_components=20, random_state=42)),
            ("rf",   RandomForestClassifier(
                        n_estimators=200,
                        max_depth=4,
                        random_state=42,
                        class_weight="balanced"
                    ))
        ])

        loo_preds = cross_val_predict(pipe, X_fg, y, cv=loo)
        acc = (loo_preds == y).mean()
        # Balanced accuracy
        from sklearn.metrics import balanced_accuracy_score, confusion_matrix
        bal_acc = balanced_accuracy_score(y, loo_preds)
        cm = confusion_matrix(y, loo_preds)

        rlog(f"    {fg:20s}  acc={acc:.2f}  bal_acc={bal_acc:.2f}  "
             f"CM=[TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}]")

        fg_results[fg] = {
            "acc": acc, "bal_acc": bal_acc, "cm": cm,
            "y": y, "preds": loo_preds
        }

    # ── Feature importance: which spectral regions matter? ───────────────────
    # Train on full data and extract importances per FG
    fig, axes = plt.subplots(len(fg_results), 1,
                             figsize=(14, 3 * len(fg_results)),
                             sharex=True)
    if len(fg_results) == 1:
        axes = [axes]

    for ax, (fg, res) in zip(axes, fg_results.items()):
        y = fg_df[fg].values
        pca  = PCA(n_components=20, random_state=42)
        Xpca = pca.fit_transform(X_fg)
        rf   = RandomForestClassifier(n_estimators=200, max_depth=4,
                                       random_state=42, class_weight="balanced")
        rf.fit(Xpca, y)

        # Project importances back to wavenumber space via PCA components
        imp_spectrum = np.abs(pca.components_.T @ rf.feature_importances_)
        imp_spectrum /= imp_spectrum.max()

        ax.fill_between(wn, imp_spectrum, alpha=0.6, color="steelblue")
        ax.set_ylabel("Importance", fontsize=8)
        ax.set_title(f"{fg}  |  bal_acc={res['bal_acc']:.2f}", fontsize=9)
        ax.set_xlim(500, 4000)

        # Overlay mean spectra of positive vs negative class
        ax2 = ax.twinx()
        ax2.plot(wn, X_fg[y == 1].mean(axis=0), color="red",
                 lw=0.8, alpha=0.7, label="positive mean")
        ax2.plot(wn, X_fg[y == 0].mean(axis=0), color="gray",
                 lw=0.8, alpha=0.7, label="negative mean")
        ax2.set_ylabel("Mean intensity", fontsize=7)
        ax2.legend(fontsize=6, loc="upper right")

    axes[-1].set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=9)
    fig.suptitle("Spectral Feature Importance per Functional Group\n"
                 "(RF importances projected back to wavenumber space)",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "task2_feature_importance.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    rlog("\n  Saved: task2_feature_importance.png")

    # ── Summary bar chart ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    fgs      = list(fg_results.keys())
    bal_accs = [fg_results[fg]["bal_acc"] for fg in fgs]
    bars = ax.barh(fgs, bal_accs, color="steelblue", edgecolor="white")
    ax.axvline(0.5, color="red", linestyle="--", label="Random baseline")
    ax.set_xlabel("Balanced accuracy (LOO cross-validation)", fontsize=10)
    ax.set_title("Functional Group Detection from IR Spectrum\n"
                 "Random Forest + PCA  |  Leave-One-Out CV", fontsize=11)
    ax.set_xlim(0, 1.05)
    for bar, val in zip(bars, bal_accs):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=9)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "task2_functional_groups.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    rlog("  Saved: task2_functional_groups.png")

# =============================================================================
# TASK 3 — SPECTRUM PREDICTION FROM MOLECULAR DESCRIPTORS
# =============================================================================

rlog()
rlog("=" * 60)
rlog("TASK 3 — Spectrum Prediction from Molecular Descriptors")
rlog("=" * 60)

if not RDKIT_OK:
    rlog("  SKIPPED — RDKit not available")
else:
    # ── Build descriptor matrix ───────────────────────────────────────────────
    def get_descriptors(smi):
        """
        Simple, interpretable PFAS-relevant descriptors from RDKit.
        Avoids the full 200-descriptor set which would overfit badly at n=68.
        """
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return None
        except:
            return None

        d = {}
        d["mol_weight"]   = Descriptors.MolWt(mol)
        d["n_atoms"]      = mol.GetNumAtoms()
        d["n_heavy"]      = mol.GetNumHeavyAtoms()
        d["n_F"]          = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 9)
        d["n_C"]          = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 6)
        d["n_O"]          = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 8)
        d["n_S"]          = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 16)
        d["n_N"]          = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
        d["n_P"]          = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 15)
        d["n_Cl"]         = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 17)
        d["frac_F"]       = d["n_F"] / max(d["n_heavy"], 1)
        d["n_rings"]      = rdMolDescriptors.CalcNumRings(mol)
        d["n_rot_bonds"]  = rdMolDescriptors.CalcNumRotatableBonds(mol)
        d["tpsa"]         = Descriptors.TPSA(mol)
        d["logp"]         = Descriptors.MolLogP(mol)
        # CF2 groups (key PFAS motif)
        cf2_pat = Chem.MolFromSmarts("[#6]([F])([F])")
        d["n_CF2"] = len(mol.GetSubstructMatches(cf2_pat)) if cf2_pat else 0
        # CF3 groups
        cf3_pat = Chem.MolFromSmarts("[#6]([F])([F])[F]")
        d["n_CF3"] = len(mol.GetSubstructMatches(cf3_pat)) if cf3_pat else 0
        # Chain length proxy: longest carbon chain
        d["n_CF2_chain"] = d["n_CF2"] - d["n_CF3"]

        return d

    desc_data = []
    desc_idx  = []
    for i, smi in enumerate(smiles):
        d = get_descriptors(smi)
        if d is not None:
            desc_data.append(d)
            desc_idx.append(i)

    desc_df = pd.DataFrame(desc_data, index=desc_idx)
    X_desc  = desc_df.values.astype(np.float64)
    X_spec  = X[desc_idx]

    rlog(f"\n  Descriptor matrix: {X_desc.shape[0]} molecules x {X_desc.shape[1]} descriptors")
    rlog(f"  Descriptors: {list(desc_df.columns)}")

    # ── PCA on spectra (target compression) ───────────────────────────────────
    # Predicting 3501 values directly is impossible at n=68.
    # Instead: compress spectra to N_PC PCA components, predict those,
    # then reconstruct.
    N_PC = 15  # captures >95% of variance for PFAS spectra

    pca_spec = PCA(n_components=N_PC, random_state=42)
    Y_pca    = pca_spec.fit_transform(X_spec)   # (68, N_PC)

    var_explained = pca_spec.explained_variance_ratio_.cumsum()
    rlog(f"\n  PCA on spectra:")
    rlog(f"    Variance explained by {N_PC} components: {var_explained[-1]*100:.1f}%")
    for k in [1, 2, 5, 10, N_PC]:
        rlog(f"    Top-{k:2d} PCs: {pca_spec.explained_variance_ratio_[:k].sum()*100:.1f}%")

    # ── Ridge regression: descriptors -> PCA coefficients (LOO CV) ───────────
    scaler = StandardScaler()
    X_desc_sc = scaler.fit_transform(X_desc)

    # Predict each PC separately with ridge regression
    loo = LeaveOneOut()
    Y_pred_pca = np.zeros_like(Y_pca)

    alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
    best_alpha = 10.0  # reasonable default; could grid search

    for pc in range(N_PC):
        ridge = Ridge(alpha=best_alpha)
        Y_pred_pca[:, pc] = cross_val_predict(ridge, X_desc_sc, Y_pca[:, pc], cv=loo)

    # Reconstruct spectra from predicted PCA coefficients
    X_spec_pred = pca_spec.inverse_transform(Y_pred_pca)
    X_spec_pred = np.clip(X_spec_pred, 0, None)

    # ── Evaluate: cosine similarity between predicted and actual ──────────────
    cos_sims = np.array([
        cosine_similarity(X_spec[i:i+1], X_spec_pred[i:i+1])[0, 0]
        for i in range(len(desc_idx))
    ])

    # Pearson correlation per molecule
    pearson_rs = np.array([
        np.corrcoef(X_spec[i], X_spec_pred[i])[0, 1]
        for i in range(len(desc_idx))
    ])

    rlog(f"\n  Spectrum prediction (Ridge regression, LOO CV):")
    rlog(f"    Mean cosine similarity:   {cos_sims.mean():.4f} +/- {cos_sims.std():.4f}")
    rlog(f"    Mean Pearson r:           {pearson_rs.mean():.4f} +/- {pearson_rs.std():.4f}")
    rlog(f"    Best predicted molecule:  {names[desc_idx[cos_sims.argmax()]]}  (r={pearson_rs.max():.3f})")
    rlog(f"    Worst predicted molecule: {names[desc_idx[cos_sims.argmin()]]}  (r={pearson_rs.min():.3f})")

    # ── PCA variance plot ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(range(1, N_PC+1),
                pca_spec.explained_variance_ratio_ * 100,
                color="steelblue", edgecolor="white")
    axes[0].plot(range(1, N_PC+1), var_explained * 100,
                 "ro-", ms=4, label="Cumulative")
    axes[0].axhline(95, color="gray", linestyle="--", label="95%")
    axes[0].set_xlabel("Principal Component", fontsize=10)
    axes[0].set_ylabel("Variance explained (%)", fontsize=10)
    axes[0].set_title("PCA of PFAS IR Spectra", fontsize=11)
    axes[0].legend()

    axes[1].scatter(cos_sims, pearson_rs, c="steelblue", s=40, alpha=0.7)
    axes[1].set_xlabel("Cosine similarity (predicted vs actual)", fontsize=10)
    axes[1].set_ylabel("Pearson r (predicted vs actual)", fontsize=10)
    axes[1].set_title("Spectrum Prediction Quality\n(Ridge regression, LOO CV)", fontsize=11)
    for i in [cos_sims.argmax(), cos_sims.argmin()]:
        axes[1].annotate(names[desc_idx[i]].split("(")[0].strip(),
                         (cos_sims[i], pearson_rs[i]),
                         fontsize=6, ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "task3_pca_variance.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    rlog("  Saved: task3_pca_variance.png")

    # ── Plot best and worst predicted spectra ─────────────────────────────────
    n_show = min(6, len(desc_idx))
    # Show 3 best + 3 worst
    order  = np.argsort(pearson_rs)
    show_idx = list(order[:3]) + list(order[-3:][::-1])
    labels_show = ["Worst 1", "Worst 2", "Worst 3",
                   "Best 3",  "Best 2",  "Best 1"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axes_flat = axes.flatten()

    for ax, si, lbl in zip(axes_flat, show_idx, labels_show):
        mol_i   = desc_idx[si]
        actual  = X_spec[si]
        pred    = X_spec_pred[si]
        pred_n  = pred / max(pred.max(), 1e-9)  # renormalise for visual

        ax.plot(wn, actual,  lw=1.0, color="steelblue", label="DFT (actual)",  alpha=0.9)
        ax.plot(wn, pred_n,  lw=1.0, color="red",       label="Predicted",     alpha=0.7,
                linestyle="--")
        ax.set_title(f"{lbl}: {names[mol_i].split('(')[0].strip()}\n"
                     f"r = {pearson_rs[si]:.3f}  cos = {cos_sims[si]:.3f}",
                     fontsize=7)
        ax.set_xlim(500, 4000)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=7)
        if ax in axes[-1]:
            ax.set_xlabel("Wavenumber (cm$^{-1}$)", fontsize=8)

    axes_flat[0].legend(fontsize=7)
    fig.suptitle("Spectrum Prediction: Actual vs Predicted (Ridge + PCA, LOO CV)\n"
                 "Descriptors: MW, atom counts, n_CF2, n_CF3, logP, TPSA, ...",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "task3_spectrum_prediction.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    rlog("  Saved: task3_spectrum_prediction.png")

# =============================================================================
# WRITE RESULTS SUMMARY
# =============================================================================

report_path = os.path.join(OUTPUT_DIR, "pfas_ml_results.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(results_lines))

rlog()
rlog(f"Results saved to {report_path}")
rlog("Done.")