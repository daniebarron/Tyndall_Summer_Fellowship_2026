"""
pfas_ml_analysis.py
===================
Three ML tasks on the 68-molecule PFAS dataset, run in two modes:
    Mode A - IR only       (3501-point spectra)
    Mode B - IR + NMR      (3501 IR + 9 NMR features)

A comparison table is printed at the end so you can see directly whether
adding NMR improves each task.

Task 1 - Spectral library matching   (cosine similarity, LOO)
Task 2 - Functional group classification (RF + PCA, LOO)
Task 3 - Spectrum prediction from RDKit descriptors [+ NMR features] (Ridge + PCA, LOO)

Usage
-----
    python pfas_ml_analysis.py

Outputs written to OUTPUT_DIR:
    task1_library_matching.png
    task2_functional_groups.png         (IR-only)
    task2_functional_groups_nmr.png     (IR+NMR)
    task2_feature_importance.png
    task3_pca_variance.png
    task3_spectrum_prediction.png
    pfas_ml_results.txt                 - full log + comparison table
"""

import os
import re
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
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    RDKIT_OK = True
except ImportError:
    RDKIT_OK = False
    print("WARNING: RDKit not available - Task 3 will be skipped")

# PATHS

IR_NPZ       = r"C:\Users\danie\Downloads\orca_work\ml_dataset\pfas_ir_dataset.npz"
COMBINED_NPZ = r"C:\Users\danie\Downloads\orca_work\ml_dataset\pfas_combined_dataset.npz"
OUTPUT_DIR   = r"C:\Users\danie\Downloads\orca_work\ml_dataset"

# LOAD DATA

print("=" * 60)
print("Loading datasets...")
print("=" * 60)

# IR-only dataset
ir_data  = np.load(IR_NPZ, allow_pickle=True)
X_ir     = ir_data['spectra'].astype(np.float64)       # (68, 3501)
wn       = ir_data['wavenumbers'].astype(np.float64)
names    = ir_data['csv_names'].tolist()
smiles   = ir_data['smiles'].tolist()

# Combined IR+NMR dataset - aligned to IR names
comb_data    = np.load(COMBINED_NPZ, allow_pickle=True)
X_combined   = comb_data['X_combined'].astype(np.float64)   # (68, 3510)
X_nmr_scaled = comb_data['X_nmr_scaled'].astype(np.float64) # (68, 9)
X_nmr_raw    = comb_data['X_nmr'].astype(np.float64)        # (68, 9) unscaled
nmr_feat_names = comb_data['nmr_feature_names'].tolist()
comb_names   = comb_data['mol_names'].tolist()

# Align IR arrays to combined ordering (may differ by 1 for diPAP)
comb_name_idx = {n: i for i, n in enumerate(comb_names)}
ir_name_idx   = {n: i for i, n in enumerate(names)}

# Build a shared index: rows present in both, in combined order
def normalise(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())

norm_comb = {normalise(n): i for i, n in enumerate(comb_names)}
norm_ir   = {normalise(n): i for i, n in enumerate(names)}

# Prefix-tolerant match (same logic as build_pfas_combined_dataset.py)
shared_keys = []
for ck in norm_comb:
    if ck in norm_ir:
        shared_keys.append(ck)
    else:
        for ik in norm_ir:
            if ck.startswith(ik) or ik.startswith(ck):
                shared_keys.append(ck)
                break

shared_keys = sorted(set(shared_keys))
N = len(shared_keys)

# Final aligned arrays
X_ir_aln   = np.array([X_ir[norm_ir.get(k) or next(norm_ir[ik] for ik in norm_ir if k.startswith(ik) or ik.startswith(k))]
                        for k in shared_keys])
X_comb_aln = np.array([X_combined[norm_comb[k]] for k in shared_keys])
X_nmr_aln  = np.array([X_nmr_raw[norm_comb[k]]  for k in shared_keys])
smiles_aln  = []
names_aln   = []
for k in shared_keys:
    ir_idx = norm_ir.get(k) or next((norm_ir[ik] for ik in norm_ir if k.startswith(ik) or ik.startswith(k)), None)
    smiles_aln.append(smiles[ir_idx])
    names_aln.append(names[ir_idx])

print(f"  IR-only:  {X_ir_aln.shape[0]} molecules, {X_ir_aln.shape[1]} features")
print(f"  Combined: {X_comb_aln.shape[0]} molecules, {X_comb_aln.shape[1]} features")
print(f"  NMR features: {nmr_feat_names}")

results_lines = []
comparison = {}   # {task_label: {metric: {ir: val, nmr: val}}}

def rlog(msg=""):
    print(msg)
    results_lines.append(msg)

# TASK 1 - SPECTRAL LIBRARY MATCHING

rlog()
rlog("=" * 60)
rlog("TASK 1 - Spectral Library Matching (Cosine Similarity)")
rlog("=" * 60)

def run_library_matching(X, label):
    sim = cosine_similarity(X)
    nn_sims = []
    for i in range(len(X)):
        s = sim[i].copy()
        s[i] = -1
        nn_sims.append(s.max())
    mean_sim = np.mean(nn_sims)
    rlog(f"\n  [{label}]")
    rlog(f"    Mean NN cosine similarity: {mean_sim:.4f}")
    rlog(f"    Min / Max: {np.min(nn_sims):.4f} / {np.max(nn_sims):.4f}")
    return mean_sim, nn_sims, sim

mean_ir,   nn_ir,   sim_ir   = run_library_matching(X_ir_aln,   "IR only")
mean_comb, nn_comb, sim_comb = run_library_matching(X_comb_aln, "IR + NMR")

comparison["Task 1 - Library matching"] = {
    "Mean NN cosine sim": {"IR only": f"{mean_ir:.4f}", "IR + NMR": f"{mean_comb:.4f}"}
}

# Plot: side-by-side heatmaps
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

short_names = [n.split("(")[0].strip()[:12] for n in names_aln]

for ax, sim, title in zip(axes[0], [sim_ir, sim_comb],
                           ["IR only", "IR + NMR"]):
    np.fill_diagonal(sim, 1)
    im = ax.imshow(sim, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_title(f"Cosine Similarity - {title}", fontsize=10)
    ax.set_xticks(range(N)); ax.set_xticklabels(short_names, rotation=90, fontsize=3.5)
    ax.set_yticks(range(N)); ax.set_yticklabels(short_names, fontsize=3.5)
    plt.colorbar(im, ax=ax)

for ax, nn_sims, mean_sim, title in zip(axes[1],
    [nn_ir, nn_comb], [mean_ir, mean_comb], ["IR only", "IR + NMR"]):
    ax.hist(nn_sims, bins=20, color="steelblue", edgecolor="white")
    ax.axvline(mean_sim, color="red", linestyle="--", label=f"Mean={mean_sim:.3f}")
    ax.set_xlabel("Cosine similarity to nearest neighbour")
    ax.set_ylabel("Count")
    ax.set_title(f"NN Similarity Distribution - {title}")
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "task1_library_matching.png"), dpi=150, bbox_inches="tight")
plt.close()
rlog("\n  Saved: task1_library_matching.png")

# TASK 2 - FUNCTIONAL GROUP CLASSIFICATION

rlog()
rlog("=" * 60)
rlog("TASK 2 - Functional Group Classification")
rlog("=" * 60)

if not RDKIT_OK:
    rlog("  SKIPPED - RDKit not available")
else:
    def get_fg_labels(smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None: return None
        except: return None
        labels = {}
        labels["carboxylic_acid"] = int(
            mol.HasSubstructMatch(Chem.MolFromSmarts("C(=O)[OH]")) or
            mol.HasSubstructMatch(Chem.MolFromSmarts("C(=O)[O-]")))
        labels["sulfonic_acid"] = int(
            mol.HasSubstructMatch(Chem.MolFromSmarts("S(=O)(=O)[OH]")) or
            mol.HasSubstructMatch(Chem.MolFromSmarts("S(=O)(=O)[O-]")))
        labels["sulfonamide"]   = int(mol.HasSubstructMatch(Chem.MolFromSmarts("S(=O)(=O)N")))
        labels["ether"]         = int(mol.HasSubstructMatch(Chem.MolFromSmarts("[#6]-O-[#6]")))
        labels["alcohol"]       = int(mol.HasSubstructMatch(Chem.MolFromSmarts("[#6][OH]")))
        labels["phosphate"]     = int(mol.HasSubstructMatch(Chem.MolFromSmarts("P(=O)([OH,O-])O")))
        labels["chlorinated"]   = int(mol.HasSubstructMatch(Chem.MolFromSmarts("[Cl]")))
        return labels

    fg_data, valid_idx = [], []
    for i, smi in enumerate(smiles_aln):
        labels = get_fg_labels(smi)
        if labels is not None:
            fg_data.append(labels); valid_idx.append(i)
    fg_df = pd.DataFrame(fg_data, index=valid_idx)

    rlog(f"\n  Label counts ({len(valid_idx)} molecules):")
    for col in fg_df.columns:
        pos = fg_df[col].sum()
        rlog(f"    {col:20s}  {pos:2d}+ / {len(valid_idx)-pos:2d}-")

    def run_fg_classification(X_feat, feat_label, n_pca=20):
        rlog(f"\n  [{feat_label}]  RF + PCA({n_pca}) LOO:")
        res = {}
        loo = LeaveOneOut()
        X_sub = X_feat[valid_idx]
        for fg in fg_df.columns:
            y = fg_df[fg].values
            if y.sum() < 3 or (len(y) - y.sum()) < 3:
                rlog(f"    {fg:20s}  SKIPPED"); continue
            pipe = Pipeline([
                ("pca", PCA(n_components=n_pca, random_state=42)),
                ("rf",  RandomForestClassifier(n_estimators=200, max_depth=4,
                                               random_state=42, class_weight="balanced"))
            ])
            preds   = cross_val_predict(pipe, X_sub, y, cv=loo)
            bal_acc = balanced_accuracy_score(y, preds)
            acc     = (preds == y).mean()
            cm      = confusion_matrix(y, preds)
            rlog(f"    {fg:20s}  acc={acc:.2f}  bal_acc={bal_acc:.2f}  "
                 f"CM=[TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}]")
            res[fg] = {"acc": acc, "bal_acc": bal_acc, "cm": cm, "y": y, "preds": preds}
        return res

    # IR-only: use spectra features
    res_ir  = run_fg_classification(X_ir_aln,   "IR only",   n_pca=20)
    # IR+NMR: spectra + scaled NMR - use more PCA components since feature space larger
    res_nmr = run_fg_classification(X_comb_aln, "IR + NMR",  n_pca=20)

    # Comparison for Task 2
    comparison["Task 2 - FG classification"] = {}
    for fg in res_ir:
        if fg in res_nmr:
            comparison["Task 2 - FG classification"][fg] = {
                "IR only":  f"{res_ir[fg]['bal_acc']:.3f}",
                "IR + NMR": f"{res_nmr[fg]['bal_acc']:.3f}",
            }

    # Bar chart comparison
    fgs_common = [fg for fg in res_ir if fg in res_nmr]
    x = np.arange(len(fgs_common))
    w = 0.35

    for suffix, res, fname in [("IR only", res_ir,  "task2_functional_groups.png"),
                                ("IR + NMR", res_nmr, "task2_functional_groups_nmr.png")]:
        fig, ax = plt.subplots(figsize=(10, 5))
        vals = [res[fg]["bal_acc"] for fg in fgs_common]
        bars = ax.barh(fgs_common, vals, color="steelblue", edgecolor="white")
        ax.axvline(0.5, color="red", linestyle="--", label="Random baseline")
        ax.set_xlabel("Balanced accuracy (LOO CV)")
        ax.set_title(f"Functional Group Classification - {suffix}\nRF + PCA  |  LOO CV")
        ax.set_xlim(0, 1.05)
        for bar, val in zip(bars, vals):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{val:.2f}", va="center", fontsize=9)
        ax.legend(); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, fname), dpi=150, bbox_inches="tight")
        plt.close()

    # Side-by-side comparison bar chart
    fig, ax = plt.subplots(figsize=(11, 5))
    vals_ir  = [res_ir[fg]["bal_acc"]  for fg in fgs_common]
    vals_nmr = [res_nmr[fg]["bal_acc"] for fg in fgs_common]
    bars1 = ax.barh(x - w/2, vals_ir,  w, label="IR only",   color="steelblue", edgecolor="white")
    bars2 = ax.barh(x + w/2, vals_nmr, w, label="IR + NMR",  color="darkorange", edgecolor="white")
    ax.set_yticks(x); ax.set_yticklabels(fgs_common)
    ax.axvline(0.5, color="red", linestyle="--", label="Random baseline")
    ax.set_xlabel("Balanced accuracy (LOO CV)")
    ax.set_title("Functional Group Classification: IR only vs IR + NMR\nRF + PCA  |  LOO CV")
    ax.set_xlim(0, 1.1)
    for bars, vals in [(bars1, vals_ir), (bars2, vals_nmr)]:
        for bar, val in zip(bars, vals):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                    f"{val:.2f}", va="center", fontsize=8)
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "task2_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    rlog("\n  Saved: task2_functional_groups.png, task2_functional_groups_nmr.png, task2_comparison.png")

    # Feature importance (IR only, for interpretability)
    fig, axes = plt.subplots(len(res_ir), 1,
                             figsize=(14, 3 * len(res_ir)), sharex=True)
    if len(res_ir) == 1: axes = [axes]
    for ax, (fg, res) in zip(axes, res_ir.items()):
        y = fg_df[fg].values
        X_sub = X_ir_aln[valid_idx]
        pca = PCA(n_components=20, random_state=42)
        Xp  = pca.fit_transform(X_sub)
        rf  = RandomForestClassifier(n_estimators=200, max_depth=4,
                                     random_state=42, class_weight="balanced")
        rf.fit(Xp, y)
        imp = np.abs(pca.components_.T @ rf.feature_importances_)
        imp /= imp.max()
        ax.fill_between(wn, imp, alpha=0.6, color="steelblue")
        ax.set_ylabel("Importance", fontsize=8)
        ax.set_title(f"{fg}  |  bal_acc(IR)={res['bal_acc']:.2f}", fontsize=9)
        ax.set_xlim(500, 4000)
        ax2 = ax.twinx()
        ax2.plot(wn, X_sub[y==1].mean(axis=0), color="red",  lw=0.8, alpha=0.7, label="+")
        ax2.plot(wn, X_sub[y==0].mean(axis=0), color="gray", lw=0.8, alpha=0.7, label="-")
        ax2.legend(fontsize=6, loc="upper right")
    axes[-1].set_xlabel("Wavenumber (cm⁻¹)", fontsize=9)
    fig.suptitle("Spectral Feature Importance per Functional Group (IR only)", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "task2_feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    rlog("  Saved: task2_feature_importance.png")

# TASK 3 - SPECTRUM PREDICTION FROM MOLECULAR DESCRIPTORS

rlog()
rlog("=" * 60)
rlog("TASK 3 - Spectrum Prediction from Molecular Descriptors")
rlog("=" * 60)

if not RDKIT_OK:
    rlog("  SKIPPED - RDKit not available")
else:
    def get_descriptors(smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None: return None
        except: return None
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
        cf2_pat = Chem.MolFromSmarts("[#6]([F])([F])")
        cf3_pat = Chem.MolFromSmarts("[#6]([F])([F])[F]")
        d["n_CF2"]       = len(mol.GetSubstructMatches(cf2_pat)) if cf2_pat else 0
        d["n_CF3"]       = len(mol.GetSubstructMatches(cf3_pat)) if cf3_pat else 0
        d["n_CF2_chain"] = d["n_CF2"] - d["n_CF3"]
        return d

    desc_data, desc_idx = [], []
    for i, smi in enumerate(smiles_aln):
        d = get_descriptors(smi)
        if d is not None:
            desc_data.append(d); desc_idx.append(i)

    desc_df   = pd.DataFrame(desc_data, index=desc_idx)
    X_desc    = desc_df.values.astype(np.float64)
    X_spec    = X_ir_aln[desc_idx]
    X_nmr_sub = X_nmr_aln[desc_idx]   # raw (unscaled) NMR features

    rlog(f"\n  RDKit descriptor matrix: {X_desc.shape}")
    rlog(f"  Descriptors: {list(desc_df.columns)}")

    N_PC = 15
    pca_spec = PCA(n_components=N_PC, random_state=42)
    Y_pca    = pca_spec.fit_transform(X_spec)

    var_exp = pca_spec.explained_variance_ratio_.cumsum()
    rlog(f"\n  Spectral PCA: {N_PC} components explain {var_exp[-1]*100:.1f}% variance")

    def run_spectrum_prediction(X_input, label):
        sc  = StandardScaler()
        Xsc = sc.fit_transform(X_input)
        loo = LeaveOneOut()
        Y_pred = np.zeros_like(Y_pca)
        for pc in range(N_PC):
            ridge = Ridge(alpha=10.0)
            Y_pred[:, pc] = cross_val_predict(ridge, Xsc, Y_pca[:, pc], cv=loo)
        X_pred = pca_spec.inverse_transform(Y_pred)
        X_pred = np.clip(X_pred, 0, None)

        cos_sims = np.array([
            cosine_similarity(X_spec[i:i+1], X_pred[i:i+1])[0, 0]
            for i in range(len(desc_idx))
        ])
        pearson_rs = np.array([
            np.corrcoef(X_spec[i], X_pred[i])[0, 1]
            for i in range(len(desc_idx))
        ])
        rlog(f"\n  [{label}]")
        rlog(f"    Mean cosine sim:  {cos_sims.mean():.4f} +/- {cos_sims.std():.4f}")
        rlog(f"    Mean Pearson r:   {pearson_rs.mean():.4f} +/- {pearson_rs.std():.4f}")
        rlog(f"    Best:  {names_aln[desc_idx[pearson_rs.argmax()]].split('(')[0].strip()}  r={pearson_rs.max():.3f}")
        rlog(f"    Worst: {names_aln[desc_idx[pearson_rs.argmin()]].split('(')[0].strip()}  r={pearson_rs.min():.3f}")
        return cos_sims, pearson_rs, X_pred

    # IR-only descriptors
    cos_ir, r_ir, Xpred_ir = run_spectrum_prediction(X_desc, "RDKit descriptors only")

    # RDKit + NMR features concatenated
    X_desc_nmr = np.hstack([X_desc, X_nmr_sub])
    rlog(f"\n  Combined descriptor matrix (RDKit + NMR): {X_desc_nmr.shape}")
    cos_nmr, r_nmr, Xpred_nmr = run_spectrum_prediction(X_desc_nmr, "RDKit + NMR descriptors")

    comparison["Task 3 - Spectrum prediction"] = {
        "Mean cosine sim":  {"IR only": f"{cos_ir.mean():.4f}", "IR + NMR": f"{cos_nmr.mean():.4f}"},
        "Mean Pearson r":   {"IR only": f"{r_ir.mean():.4f}",   "IR + NMR": f"{r_nmr.mean():.4f}"},
    }

    # PCA variance plot + scatter
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].bar(range(1, N_PC+1), pca_spec.explained_variance_ratio_*100,
                color="steelblue", edgecolor="white")
    axes[0].plot(range(1, N_PC+1), var_exp*100, "ro-", ms=4, label="Cumulative")
    axes[0].axhline(95, color="gray", linestyle="--", label="95%")
    axes[0].set_xlabel("Principal Component"); axes[0].set_ylabel("Variance explained (%)")
    axes[0].set_title("Spectral PCA"); axes[0].legend()

    for ax, cos_s, pears, lbl, col in [
        (axes[1], cos_ir,  r_ir,  "RDKit only",     "steelblue"),
        (axes[2], cos_nmr, r_nmr, "RDKit + NMR",    "darkorange"),
    ]:
        ax.scatter(cos_s, pears, c=col, s=40, alpha=0.7)
        ax.set_xlabel("Cosine similarity"); ax.set_ylabel("Pearson r")
        ax.set_title(f"Prediction quality - {lbl}\n(Ridge + PCA, LOO CV)")
        for i in [pears.argmax(), pears.argmin()]:
            ax.annotate(names_aln[desc_idx[i]].split("(")[0].strip()[:15],
                        (cos_s[i], pears[i]), fontsize=6, ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "task3_pca_variance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    rlog("\n  Saved: task3_pca_variance.png")

    # Best/worst spectra for both modes
    for pears, Xpred, suffix in [(r_ir, Xpred_ir, "rdkit"), (r_nmr, Xpred_nmr, "rdkit_nmr")]:
        order    = np.argsort(pears)
        show_idx = list(order[:3]) + list(order[-3:][::-1])
        labels_show = ["Worst 1", "Worst 2", "Worst 3", "Best 3", "Best 2", "Best 1"]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
        for ax, si, lbl in zip(axes.flatten(), show_idx, labels_show):
            actual = X_spec[si]
            pred   = Xpred[si]
            pred_n = pred / max(pred.max(), 1e-9)
            ax.plot(wn, actual, lw=1.0, color="steelblue", label="DFT", alpha=0.9)
            ax.plot(wn, pred_n, lw=1.0, color="red", label="Predicted", alpha=0.7, linestyle="--")
            ax.set_title(f"{lbl}: {names_aln[desc_idx[si]].split('(')[0].strip()[:20]}\n"
                         f"r={pears[si]:.3f}", fontsize=7)
            ax.set_xlim(500, 4000); ax.set_yticks([])
        axes.flatten()[0].legend(fontsize=7)
        fig.suptitle(f"Spectrum Prediction ({suffix.replace('_', ' + ').upper()}, Ridge+PCA, LOO CV)", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"task3_spectrum_prediction_{suffix}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
    rlog("  Saved: task3_spectrum_prediction_rdkit.png, task3_spectrum_prediction_rdkit_nmr.png")

# COMPARISON TABLE

rlog()
rlog("=" * 60)
rlog("COMPARISON: IR only  vs  IR + NMR")
rlog("=" * 60)

col_w = 28
rlog(f"\n  {'Metric':{col_w}}  {'IR only':>12}  {'IR + NMR':>12}  {'Δ':>10}")
rlog(f"  {'-'*col_w}  {'-'*12}  {'-'*12}  {'-'*10}")

for task, metrics in comparison.items():
    rlog(f"\n  {task}")
    for metric, vals in metrics.items():
        ir_val  = vals["IR only"]
        nmr_val = vals["IR + NMR"]
        try:
            delta = float(nmr_val) - float(ir_val)
            delta_str = f"{delta:+.4f}"
            arrow = "▲" if delta > 0.001 else ("▼" if delta < -0.001 else "~")
            delta_str = f"{arrow} {delta_str}"
        except:
            delta_str = "-"
        rlog(f"    {metric:{col_w}}  {ir_val:>12}  {nmr_val:>12}  {delta_str:>10}")

rlog()

# SAVE REPORT

report_path = os.path.join(OUTPUT_DIR, "pfas_ml_results.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(results_lines))

rlog(f"Full results saved to {report_path}")
rlog("Done.")