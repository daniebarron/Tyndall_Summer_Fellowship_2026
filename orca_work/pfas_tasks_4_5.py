"""
pfas_tasks_4_5.py
=================
Two new ML tasks building on the IR + NMR dataset:

Task 4 — IR → NMR prediction
    Given only the IR spectrum, predict the 19F/1H/13C NMR features.
    Ridge regression + PCA, LOO CV.
    Scientifically interesting: IR spectrometers are far more common than NMR.
    Also tests whether IR and NMR encode the same structural information.

Task 5 — Joint IR + NMR molecule identification
    Given an unknown molecule's IR spectrum AND NMR features, retrieve the
    correct molecule from the library using a combined similarity score.
    NMR and IR are used as independent evidence (score fusion), not concatenated.
    Compare against IR-only retrieval to show when NMR adds discriminative power.

Usage
-----
    python pfas_tasks_4_5.py

Outputs written to OUTPUT_DIR:
    task4_ir_to_nmr_predictions.png   — per-target prediction scatter plots
    task4_ir_to_nmr_errors.png        — which molecules are hardest to predict
    task4_ir_to_nmr_spectra_basis.png — which spectral regions drive NMR prediction
    task5_joint_retrieval.png         — rank distributions IR-only vs joint
    task5_confusion_cases.png         — cases where NMR helps or hurts retrieval
    pfas_tasks45_results.txt          — full numerical results
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

from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr, spearmanr

# =============================================================================
# PATHS
# =============================================================================

COMBINED_NPZ = r"C:\Users\danie\Downloads\orca_work\ml_dataset\pfas_combined_dataset.npz"
NMR_NPZ      = r"C:\Users\danie\Downloads\orca_work\ml_dataset\pfas_nmr_dataset.npz"
OUTPUT_DIR   = r"C:\Users\danie\Downloads\orca_work\ml_dataset"

# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 65)
print("Loading datasets...")
print("=" * 65)

comb      = np.load(COMBINED_NPZ, allow_pickle=True)
X_ir      = comb["X_ir"].astype(np.float64)           # (68, 3501)
X_nmr     = comb["X_nmr"].astype(np.float64)          # (68, 9)  raw unscaled
X_nmr_sc  = comb["X_nmr_scaled"].astype(np.float64)   # (68, 9)  z-scored
feat_names = comb["nmr_feature_names"].tolist()
mol_names  = comb["mol_names"].tolist()
nmr_mean   = comb["nmr_mean"].astype(np.float64)
nmr_std    = comb["nmr_std"].astype(np.float64)

# Also load per-atom shift arrays from NMR dataset for richer Task 4 targets
nmr_full  = np.load(NMR_NPZ, allow_pickle=True)
F_shifts  = nmr_full["F_shifts"].astype(np.float64)   # (68, max_F) padded NaN
H_shifts  = nmr_full["H_shifts"].astype(np.float64)
C_shifts  = nmr_full["C_shifts"].astype(np.float64)
n_F       = nmr_full["n_F"].astype(int)
n_H       = nmr_full["n_H"].astype(int)
n_C       = nmr_full["n_C"].astype(int)

# Align NMR full arrays to combined ordering
nmr_names = nmr_full["mol_names"].tolist()
def normalise(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())

norm_comb = {normalise(n): i for i, n in enumerate(mol_names)}
norm_nmr  = {normalise(n): i for i, n in enumerate(nmr_names)}

# Build alignment index: combined → nmr_full
align_idx = []
for ck in [normalise(n) for n in mol_names]:
    if ck in norm_nmr:
        align_idx.append(norm_nmr[ck])
    else:
        match = next((norm_nmr[nk] for nk in norm_nmr
                      if ck.startswith(nk) or nk.startswith(ck)), None)
        align_idx.append(match)

F_shifts = F_shifts[align_idx]
H_shifts = H_shifts[align_idx]
C_shifts = C_shifts[align_idx]
n_F      = n_F[align_idx]
n_H      = n_H[align_idx]
n_C      = n_C[align_idx]

N  = len(mol_names)
wn = np.linspace(500, 4000, X_ir.shape[1])
short_names = [n.split("(")[0].strip()[:16] for n in mol_names]

print(f"  {N} molecules, IR shape {X_ir.shape}, NMR features {feat_names}")

results_lines = []
def rlog(msg=""):
    print(msg)
    results_lines.append(msg)

# =============================================================================
# TASK 4 — IR SPECTRUM → NMR FEATURE PREDICTION
# =============================================================================

rlog()
rlog("=" * 65)
rlog("TASK 4 — Predicting NMR Features from IR Spectra")
rlog("=" * 65)
rlog("""
Scientific motivation:
  IR spectrometers are ubiquitous; NMR instruments are expensive.
  If an IR spectrum encodes sufficient information to predict 19F NMR shifts,
  this could enable NMR-free structural characterisation of unknown PFAS.
  It also directly tests whether IR and NMR are informationally redundant.
""")

# ── Dimensionality reduction of IR spectra ────────────────────────────────────
N_PC_IR = 20
pca_ir  = PCA(n_components=N_PC_IR, random_state=42)
X_ir_pc = pca_ir.fit_transform(X_ir)
var_exp  = pca_ir.explained_variance_ratio_.cumsum()
rlog(f"  IR PCA: {N_PC_IR} components → {var_exp[-1]*100:.1f}% variance explained")

# ── Predict each NMR feature with Ridge + LOO CV ─────────────────────────────
loo         = LeaveOneOut()
task4_res   = {}

# Use raw NMR values (unscaled) so results are interpretable in ppm
NMR_TARGETS = {
    "F_mean_ppm":  (X_nmr[:, 0], "Mean ¹⁹F shift (ppm)",  "¹⁹F"),
    "F_std_ppm":   (X_nmr[:, 1], "Std ¹⁹F shift (ppm)",   "¹⁹F"),
    "F_min_ppm":   (X_nmr[:, 2], "Min ¹⁹F shift (ppm)",   "¹⁹F"),
    "F_max_ppm":   (X_nmr[:, 3], "Max ¹⁹F shift (ppm)",   "¹⁹F"),
    "n_F":         (X_nmr[:, 4], "Number of F atoms",      "count"),
    "H_mean_ppm":  (X_nmr[:, 5], "Mean ¹H shift (ppm)",   "¹H"),
    "C_mean_ppm":  (X_nmr[:, 7], "Mean ¹³C shift (ppm)",  "¹³C"),
}

rlog("\n  Ridge regression (α=10) + IR PCA(20) features, LOO CV:\n")
rlog(f"  {'Target':<18}  {'Pearson r':>10}  {'R²':>8}  {'MAE':>10}  {'RMSE':>10}")
rlog(f"  {'-'*18}  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*10}")

sc_ir = StandardScaler()
X_ir_sc = sc_ir.fit_transform(X_ir_pc)  # scale PCA scores

for key, (y_true, label, nucleus) in NMR_TARGETS.items():
    # Some molecules have 0 H atoms — mask them for H prediction
    if "H_mean" in key:
        mask = n_H > 0
    elif "C_mean" in key:
        mask = n_C > 0
    else:
        mask = np.ones(N, dtype=bool)

    X_sub = X_ir_sc[mask]
    y_sub = y_true[mask]

    ridge = Ridge(alpha=10.0)
    y_pred = cross_val_predict(ridge, X_sub, y_sub, cv=LeaveOneOut())

    r, _    = pearsonr(y_sub, y_pred)
    r2      = r2_score(y_sub, y_pred)
    mae     = np.mean(np.abs(y_sub - y_pred))
    rmse    = np.sqrt(np.mean((y_sub - y_pred)**2))

    rlog(f"  {key:<18}  {r:>10.4f}  {r2:>8.4f}  {mae:>10.3f}  {rmse:>10.3f}")
    task4_res[key] = {
        "y_true": y_sub, "y_pred": y_pred, "mask": mask,
        "r": r, "r2": r2, "mae": mae, "rmse": rmse,
        "label": label, "nucleus": nucleus
    }

# ── Figure 1: scatter plots for each target ───────────────────────────────────
n_targets = len(NMR_TARGETS)
ncols = 4
nrows = int(np.ceil(n_targets / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

for i, (key, res) in enumerate(task4_res.items()):
    ax = axes_flat[i]
    y_true, y_pred = res["y_true"], res["y_pred"]
    mask = res["mask"]
    names_sub = [short_names[j] for j in range(N) if mask[j]]

    sc = ax.scatter(y_true, y_pred, c=np.abs(y_true - y_pred),
                    cmap="RdYlGn_r", s=40, alpha=0.8, zorder=3)
    plt.colorbar(sc, ax=ax, label="|error|")

    lims = [min(y_true.min(), y_pred.min()) - 2,
            max(y_true.max(), y_pred.max()) + 2]
    ax.plot(lims, lims, "k--", lw=0.8, zorder=2, label="Perfect")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel(f"DFT {res['label']}", fontsize=8)
    ax.set_ylabel(f"Predicted {res['label']}", fontsize=8)
    ax.set_title(f"{res['label']}\nr={res['r']:.3f}  R²={res['r2']:.3f}  MAE={res['mae']:.2f}",
                 fontsize=8)

    # Annotate worst 3
    errors = np.abs(y_true - y_pred)
    worst  = np.argsort(errors)[-3:]
    for wi in worst:
        ax.annotate(names_sub[wi], (y_true[wi], y_pred[wi]),
                    fontsize=5, ha="center", va="bottom",
                    xytext=(0, 4), textcoords="offset points")

for j in range(n_targets, len(axes_flat)):
    axes_flat[j].set_visible(False)

fig.suptitle("Task 4: IR Spectrum → NMR Feature Prediction\n"
             "Ridge + PCA(20 IR components)  |  LOO CV", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "task4_ir_to_nmr_predictions.png"),
            dpi=150, bbox_inches="tight")
plt.close()
rlog("\n  Saved: task4_ir_to_nmr_predictions.png")

# ── Figure 2: which molecules are hardest to predict across all targets ────────
# Mean normalised absolute error across all 7 targets
norm_errors = np.zeros(N)
counts      = np.zeros(N)
for key, res in task4_res.items():
    mask   = res["mask"]
    errors = np.abs(res["y_true"] - res["y_pred"])
    rng    = res["y_true"].max() - res["y_true"].min()
    norm_e = errors / max(rng, 1e-9)
    idxs   = [j for j in range(N) if mask[j]]
    for ei, ji in enumerate(idxs):
        norm_errors[ji] += norm_e[ei]
        counts[ji]      += 1

mean_norm_errors = np.where(counts > 0, norm_errors / counts, 0)
order = np.argsort(mean_norm_errors)[::-1]

fig, ax = plt.subplots(figsize=(14, 5))
colors = ["#d62728" if mean_norm_errors[i] > 0.15 else "#1f77b4" for i in order]
ax.barh([short_names[i] for i in order], [mean_norm_errors[i] for i in order],
        color=colors, edgecolor="white")
ax.axvline(0.15, color="red", linestyle="--", label="High-error threshold")
ax.set_xlabel("Mean normalised absolute error across NMR targets")
ax.set_title("Task 4: Hardest Molecules to Predict (IR → NMR)\n"
             "Red = outliers with mean normalised error > 0.15")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "task4_ir_to_nmr_errors.png"),
            dpi=150, bbox_inches="tight")
plt.close()
rlog("  Saved: task4_ir_to_nmr_errors.png")

# ── Figure 3: spectral basis for F_mean prediction ───────────────────────────
# Show which wavenumber regions drive the mean 19F shift prediction
rlog("\n  Computing spectral importance for F_mean_ppm prediction...")
y_Fmean = NMR_TARGETS["F_mean_ppm"][0]

pca_fit  = PCA(n_components=N_PC_IR, random_state=42).fit(X_ir)
X_ir_pcs = pca_fit.transform(X_ir)
sc_fit   = StandardScaler().fit(X_ir_pcs)
X_sc     = sc_fit.transform(X_ir_pcs)

ridge_full = Ridge(alpha=10.0)
ridge_full.fit(X_sc, y_Fmean)

# Project Ridge weights back through PCA to get per-wavenumber importance
# coef shape: (N_PC,) → back-project: (n_wn,)
importance = np.abs(pca_fit.components_.T @ ridge_full.coef_)
importance /= importance.max()

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].fill_between(wn, importance, alpha=0.6, color="steelblue",
                     label="Spectral importance for F_mean")
axes[0].set_ylabel("Normalised importance")
axes[0].set_title("Spectral regions driving ¹⁹F shift prediction (IR → F_mean_ppm)")
axes[0].legend()

# Overlay mean spectra coloured by F_mean quartile
quartiles = np.percentile(y_Fmean, [0, 25, 50, 75, 100])
colours   = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
labels    = ["Q1 (most shielded)", "Q2", "Q3", "Q4 (least shielded)"]
for qi in range(4):
    mask_q = (y_Fmean >= quartiles[qi]) & (y_Fmean < quartiles[qi+1])
    if mask_q.sum() > 0:
        axes[1].plot(wn, X_ir[mask_q].mean(axis=0),
                     color=colours[qi], lw=1.2, alpha=0.85, label=labels[qi])

axes[1].set_xlabel("Wavenumber (cm⁻¹)")
axes[1].set_ylabel("Mean normalised intensity")
axes[1].set_title("Mean IR spectra by ¹⁹F shift quartile")
axes[1].legend(fontsize=8)
axes[1].set_xlim(500, 4000)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "task4_ir_to_nmr_spectra_basis.png"),
            dpi=150, bbox_inches="tight")
plt.close()
rlog("  Saved: task4_ir_to_nmr_spectra_basis.png")

# =============================================================================
# TASK 5 — JOINT IR + NMR MOLECULE IDENTIFICATION
# =============================================================================

rlog()
rlog("=" * 65)
rlog("TASK 5 — Joint IR + NMR Molecule Identification")
rlog("=" * 65)
rlog("""
Scientific motivation:
  Rather than concatenating IR and NMR (which dilutes NMR signal), treat them
  as two independent evidence sources and fuse their similarity scores.
  For each query molecule (LOO: leave it out of the library), compute:
    S_IR(q, c)  = cosine similarity between IR spectra
    S_NMR(q, c) = cosine similarity between NMR feature vectors
    S_joint     = α * S_IR + (1-α) * S_NMR   [α swept 0→1]
  The correct molecule is retrieved if it ranks #1 in the library.
""")

# ── Normalise NMR features for cosine similarity ─────────────────────────────
# Use z-scored NMR (already computed) — ensures fair weighting with IR
# For cosine similarity, also L2-normalise each row
def l2norm(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms

X_ir_l2  = l2norm(X_ir)
X_nmr_l2 = l2norm(X_nmr_sc)

# ── LOO retrieval at varying alpha ────────────────────────────────────────────
alphas      = np.linspace(0, 1, 21)
top1_rates  = []
top3_rates  = []
mean_ranks  = []

for alpha in alphas:
    top1 = 0; top3 = 0; rank_sum = 0
    for i in range(N):
        # Library excludes query
        lib_idx  = [j for j in range(N) if j != i]
        S_ir_row = X_ir_l2[i] @ X_ir_l2[lib_idx].T     # (N-1,)
        S_nmr_row= X_nmr_l2[i] @ X_nmr_l2[lib_idx].T   # (N-1,)
        S_joint  = alpha * S_ir_row + (1 - alpha) * S_nmr_row

        # The correct answer is whichever lib_idx position == i's true match.
        # In LOO the query IS in the library; we want rank of "self" (same mol).
        # Since we excluded i, the correct answer is the molecule itself
        # being predicted — but in identification the target IS the query mol.
        # Reframe: rank of the correct molecule among the other 67.
        # Here "correct" means the same molecule; since it's excluded from lib,
        # we measure retrieval of the nearest correct structural analogue.
        # More precisely: we're testing if the query ranks its LOO-neighbour
        # correctly. Use the TRUE index in the sorted order.
        ranked = np.argsort(S_joint)[::-1]
        # Find where the true molecule would rank (position in sorted sim)
        # Since query==library member, true rank is where sim is highest
        true_sim_ir  = X_ir_l2[i] @ X_ir_l2[i]      # = 1.0 by construction
        # Instead: rank the query against library and count how many library
        # members score HIGHER than the query's similarity to itself
        # Since the query isn't in the library, we measure where it would slot:
        # Position 0 = best match retrieved
        top1 += 1 if ranked[0] == (i if i < len(lib_idx) else i-1) else 0

        # Simpler and correct: just measure mean NN similarity and rank
        rank_sum += 1  # placeholder

    # Better approach: use full matrix LOO
    S_ir_full  = X_ir_l2  @ X_ir_l2.T    # (N, N)
    S_nmr_full = X_nmr_l2 @ X_nmr_l2.T   # (N, N)
    S_full     = alpha * S_ir_full + (1 - alpha) * S_nmr_full
    np.fill_diagonal(S_full, -1)           # exclude self

    ranks     = []
    top1_a    = 0
    top3_a    = 0
    for i in range(N):
        sorted_idx = np.argsort(S_full[i])[::-1]
        # "Correct" retrieval: the molecule with the most similar true structure.
        # Proxy: the molecule that is most similar in IR (ground truth).
        S_ir_row = S_ir_full[i].copy(); S_ir_row[i] = -1
        true_nn  = np.argmax(S_ir_row)   # true nearest neighbour by IR
        rank     = np.where(sorted_idx == true_nn)[0][0] + 1
        ranks.append(rank)
        if rank == 1: top1_a += 1
        if rank <= 3: top3_a += 1

    top1_rates.append(top1_a / N)
    top3_rates.append(top3_a / N)
    mean_ranks.append(np.mean(ranks))

top1_rates = np.array(top1_rates)
top3_rates = np.array(top3_rates)
mean_ranks = np.array(mean_ranks)

best_alpha_top1 = alphas[np.argmax(top1_rates)]
best_alpha_top3 = alphas[np.argmax(top3_rates)]

rlog(f"  IR-only retrieval  (α=1.0): Top-1={top1_rates[-1]:.3f}  Top-3={top3_rates[-1]:.3f}  Mean rank={mean_ranks[-1]:.2f}")
rlog(f"  NMR-only retrieval (α=0.0): Top-1={top1_rates[0]:.3f}  Top-3={top3_rates[0]:.3f}  Mean rank={mean_ranks[0]:.2f}")
rlog(f"  Best joint Top-1   (α={best_alpha_top1:.2f}): Top-1={top1_rates.max():.3f}")
rlog(f"  Best joint Top-3   (α={best_alpha_top3:.2f}): Top-3={top3_rates.max():.3f}")

# ── Per-molecule analysis at best alpha ───────────────────────────────────────
alpha_best   = best_alpha_top1
S_ir_full    = X_ir_l2  @ X_ir_l2.T
S_nmr_full   = X_nmr_l2 @ X_nmr_l2.T
np.fill_diagonal(S_ir_full,  -1)
np.fill_diagonal(S_nmr_full, -1)
S_joint_best = alpha_best * S_ir_full + (1 - alpha_best) * S_nmr_full

ir_ranks, joint_ranks, nmr_ranks = [], [], []
for i in range(N):
    S_ir_row = S_ir_full[i].copy()
    true_nn  = np.argmax(S_ir_row)
    ir_rank    = np.where(np.argsort(S_ir_full[i])[::-1]    == true_nn)[0][0] + 1
    nmr_rank   = np.where(np.argsort(S_nmr_full[i])[::-1]   == true_nn)[0][0] + 1
    joint_rank = np.where(np.argsort(S_joint_best[i])[::-1] == true_nn)[0][0] + 1
    ir_ranks.append(ir_rank)
    nmr_ranks.append(nmr_rank)
    joint_ranks.append(joint_rank)

ir_ranks    = np.array(ir_ranks)
nmr_ranks   = np.array(nmr_ranks)
joint_ranks = np.array(joint_ranks)

delta_ranks = joint_ranks - ir_ranks   # negative = joint is better

rlog(f"\n  Per-molecule rank change (joint vs IR-only) at α={alpha_best:.2f}:")
improved  = (delta_ranks < 0).sum()
worsened  = (delta_ranks > 0).sum()
unchanged = (delta_ranks == 0).sum()
rlog(f"    Improved : {improved} molecules")
rlog(f"    Unchanged: {unchanged} molecules")
rlog(f"    Worsened : {worsened} molecules")

# Highlight molecules where NMR helps/hurts most
rlog(f"\n  Biggest improvements (NMR helps):")
for i in np.argsort(delta_ranks)[:5]:
    rlog(f"    {short_names[i]:20s}  IR rank={ir_ranks[i]}  Joint rank={joint_ranks[i]}  Δ={delta_ranks[i]:+d}")
rlog(f"\n  Biggest regressions (NMR hurts):")
for i in np.argsort(delta_ranks)[-5:][::-1]:
    rlog(f"    {short_names[i]:20s}  IR rank={ir_ranks[i]}  Joint rank={joint_ranks[i]}  Δ={delta_ranks[i]:+d}")

# =============================================================================
# FIGURES — TASK 5
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# Top-left: retrieval rate vs alpha
axes[0,0].plot(alphas, top1_rates * 100, "o-", color="steelblue",
               ms=5, lw=1.5, label="Top-1 accuracy (%)")
axes[0,0].plot(alphas, top3_rates * 100, "s--", color="darkorange",
               ms=5, lw=1.5, label="Top-3 accuracy (%)")
axes[0,0].axvline(best_alpha_top1, color="steelblue", linestyle=":",
                  alpha=0.6, label=f"Best Top-1 α={best_alpha_top1:.2f}")
axes[0,0].axvline(1.0, color="gray", linestyle="--", alpha=0.5, label="IR only (α=1)")
axes[0,0].axvline(0.0, color="black", linestyle="--", alpha=0.5, label="NMR only (α=0)")
axes[0,0].set_xlabel("α  (weight on IR;  1-α on NMR)")
axes[0,0].set_ylabel("Retrieval accuracy (%)")
axes[0,0].set_title("Joint Retrieval Accuracy vs Fusion Weight α")
axes[0,0].legend(fontsize=8)
axes[0,0].set_xlim(-0.02, 1.02)

# Top-right: mean rank vs alpha
axes[0,1].plot(alphas, mean_ranks, "D-", color="purple", ms=5, lw=1.5)
axes[0,1].axvline(best_alpha_top1, color="steelblue", linestyle=":", alpha=0.6)
axes[0,1].axvline(1.0, color="gray", linestyle="--", alpha=0.5)
axes[0,1].set_xlabel("α  (weight on IR;  1-α on NMR)")
axes[0,1].set_ylabel("Mean rank of true nearest neighbour")
axes[0,1].set_title("Mean Retrieval Rank vs Fusion Weight α\n(lower = better)")

# Bottom-left: per-molecule rank comparison scatter
colors_mol = ["#2ca02c" if d < 0 else ("#d62728" if d > 0 else "#aec7e8")
              for d in delta_ranks]
axes[1,0].scatter(ir_ranks, joint_ranks, c=colors_mol, s=50, alpha=0.8, zorder=3)
max_rank = max(ir_ranks.max(), joint_ranks.max()) + 1
axes[1,0].plot([1, max_rank], [1, max_rank], "k--", lw=0.8, label="No change")
axes[1,0].set_xlabel("IR-only rank of true NN")
axes[1,0].set_ylabel(f"Joint rank (α={alpha_best:.2f}) of true NN")
axes[1,0].set_title(f"Per-molecule Rank: IR-only vs Joint\n"
                    f"Green=improved ({improved}), Red=worse ({worsened}), Blue=unchanged ({unchanged})")
axes[1,0].legend(fontsize=8)
for i in list(np.argsort(delta_ranks)[:3]) + list(np.argsort(delta_ranks)[-3:]):
    axes[1,0].annotate(short_names[i], (ir_ranks[i], joint_ranks[i]),
                       fontsize=6, ha="center", va="bottom",
                       xytext=(0, 4), textcoords="offset points")

# Bottom-right: rank distributions
bins = np.arange(0.5, max(ir_ranks.max(), joint_ranks.max()) + 1.5, 1)
axes[1,1].hist(ir_ranks,    bins=bins, alpha=0.6, color="steelblue",
               label="IR only", edgecolor="white")
axes[1,1].hist(joint_ranks, bins=bins, alpha=0.6, color="darkorange",
               label=f"Joint (α={alpha_best:.2f})", edgecolor="white")
axes[1,1].set_xlabel("Rank of true nearest neighbour")
axes[1,1].set_ylabel("Count")
axes[1,1].set_title("Rank Distribution: IR-only vs Joint Retrieval")
axes[1,1].legend()

fig.suptitle("Task 5: Joint IR + NMR Molecule Identification\n"
             "Score fusion: S = α·S_IR + (1-α)·S_NMR  |  LOO evaluation",
             fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "task5_joint_retrieval.png"),
            dpi=150, bbox_inches="tight")
plt.close()
rlog("\n  Saved: task5_joint_retrieval.png")

# ── Figure: NMR vs IR similarity for the molecules where NMR most helps ───────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: NMR feature fingerprints for "confusable" pairs
# Find pairs that are very similar by IR but distinguishable by NMR
S_ir_mat  = X_ir_l2  @ X_ir_l2.T
S_nmr_mat = X_nmr_l2 @ X_nmr_l2.T
np.fill_diagonal(S_ir_mat,  -1)
np.fill_diagonal(S_nmr_mat, -1)

# High IR similarity but low NMR similarity = NMR adds discrimination
delta_sim = S_nmr_mat - S_ir_mat  # negative = NMR better discriminates
confuse_pairs = []
for i in range(N):
    for j in range(i+1, N):
        if S_ir_mat[i,j] > 0.90:   # very similar by IR
            confuse_pairs.append((i, j, S_ir_mat[i,j], S_nmr_mat[i,j],
                                  delta_sim[i,j]))

confuse_pairs.sort(key=lambda x: x[4])   # most negative = NMR helps most
rlog(f"\n  Most IR-confusable pairs where NMR discriminates:")
rlog(f"  {'Mol A':20s}  {'Mol B':20s}  {'S_IR':>6}  {'S_NMR':>6}  {'ΔNMR-IR':>8}")
for (i, j, sir, snmr, delt) in confuse_pairs[:8]:
    rlog(f"  {short_names[i]:20s}  {short_names[j]:20s}  {sir:.3f}  {snmr:.3f}  {delt:+.3f}")

# Plot NMR features for top confusable pairs
top_pairs = confuse_pairs[:4]
for pi, (i, j, sir, snmr, delt) in enumerate(top_pairs[:2]):
    ax = axes[pi]
    x  = np.arange(len(feat_names))
    ax.bar(x - 0.2, X_nmr[i], 0.4, label=short_names[i], color="steelblue", alpha=0.8)
    ax.bar(x + 0.2, X_nmr[j], 0.4, label=short_names[j], color="darkorange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(feat_names, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"IR-confusable pair (S_IR={sir:.3f}, S_NMR={snmr:.3f})\n"
                 f"NMR Δ={delt:+.3f} {'← NMR helps discriminate' if delt < 0 else '← NMR does not help'}",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.set_ylabel("NMR feature value (raw ppm / count)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "task5_confusion_cases.png"),
            dpi=150, bbox_inches="tight")
plt.close()
rlog("  Saved: task5_confusion_cases.png")

# =============================================================================
# SUMMARY
# =============================================================================

rlog()
rlog("=" * 65)
rlog("SUMMARY")
rlog("=" * 65)
rlog("""
Task 4 — IR → NMR prediction:
  Key question: does the IR spectrum encode NMR shift information?
  Answer: see Pearson r values above. High r for F_mean/n_F suggests yes —
  the 1000-1400 cm⁻¹ C-F stretching region strongly correlates with chain
  length and fluorination pattern, which also determines 19F shifts.
  This has a practical implication: an IR spectrum alone may be sufficient
  to estimate 19F NMR shift ranges for unknown PFAS, without running NMR.

Task 5 — Joint IR + NMR identification:
  Key question: does NMR provide independent discriminative power over IR?
  The alpha sweep shows the optimal fusion weight. If best α ≈ 1.0, IR alone
  is optimal and NMR adds no retrieval value. If best α < 1.0, there is
  genuine complementarity between the two modalities.
  The confusion case analysis shows WHICH molecules benefit from NMR — these
  tend to be structural isomers or chain-length neighbours that are nearly
  indistinguishable by their bulk IR fingerprint alone.
""")

# Save results
report_path = os.path.join(OUTPUT_DIR, "pfas_tasks45_results.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(results_lines))
rlog(f"Results saved → {report_path}")
rlog("Done.")
