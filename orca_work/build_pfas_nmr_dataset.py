"""
build_pfas_nmr_dataset.py
=========================
Parses all 68 PFAS ORCA NMR output files and builds a combined dataset of
isotropic chemical shieldings, then converts them to chemical shifts (δ)
using reference shieldings computed at the same level of theory (PBE0/def2-TZVP).

Reference compounds (run ref_nmr_inputs/ first):
    19F : CFCl3  (σ_ref from cfcl3_nmr.out)
    1H  : TMS    (σ_ref from tms_nmr.out)
    13C : TMS    (σ_ref from tms_nmr.out)

Convention: δ = σ_ref - σ_molecule  (positive δ = downfield)

Outputs
-------
    ml_dataset/pfas_nmr_dataset.npz
        mol_names   : (68,)       molecule name strings
        F_shifts    : (68, max_F) 19F chemical shifts, NaN-padded
        H_shifts    : (68, max_H) 1H  chemical shifts, NaN-padded
        C_shifts    : (68, max_C) 13C chemical shifts, NaN-padded
        F_shield    : (68, max_F) raw 19F shieldings
        H_shield    : (68, max_H) raw 1H  shieldings
        C_shield    : (68, max_C) raw 13C shieldings
        n_F         : (68,)       number of F atoms per molecule
        n_H         : (68,)       number of H atoms per molecule
        n_C         : (68,)       number of C atoms per molecule

    ml_dataset/pfas_nmr_summary.csv
        Human-readable CSV with mean/std shifts per nucleus per molecule.

Usage
-----
    # First parse references (or set REF_* manually if you have literature values)
    python build_pfas_nmr_dataset.py --refs path/to/ref_outputs/
    # Or skip reference conversion and just save raw shieldings:
    python build_pfas_nmr_dataset.py --no-refs
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd

# =============================================================================
# PATHS
# =============================================================================

NMR_OUT_DIR  = r"C:\Users\danie\Downloads\orca_work\nmr_outputs"   # folder of mol subdirs
REF_OUT_DIR  = r"C:\Users\danie\Downloads\orca_work\ref_nmr_outputs"  # cfcl3 + tms .out files
ML_DIR       = r"C:\Users\danie\Downloads\orca_work\ml_dataset"

# =============================================================================
# REFERENCE SHIELDINGS (PBE0/def2-TZVP GIAO, computed on cluster)
#
# CFCl3 19F : 179.716 ppm  (from cfcl3_nmr.out)
# TMS   1H  :  31.00  ppm  (mean of all 12 H in tms_nmr.out: 30.869×8, 32.142×4)
# TMS   13C : 197.342 ppm  (mean of all 4 C in tms_nmr.out)
#
# Convention for all nuclei: δ = σ_ref - σ_molecule
#   19F: CFCl3 is 0 ppm; PFAS CF2 groups (σ~310) → δ = 179.7-310 = -130 ppm ✓
#   1H:  TMS   is 0 ppm; PFAS H (σ~6)    → δ = 31.0-6   =  +25 ppm ✓
#   13C: TMS   is 0 ppm; PFAS C (σ~70)   → δ = 197.3-70 = +127 ppm ✓
#
# These are used as fallbacks if computed ref .out files are not found.
# =============================================================================

LIT_REF_F  = 179.716  # 19F in CFCl3 (ppm) — computed PBE0/def2-TZVP GIAO
LIT_REF_H  =  31.00   # 1H  in TMS   (ppm) — computed PBE0/def2-TZVP GIAO
LIT_REF_C  = 197.342  # 13C in TMS   (ppm) — computed PBE0/def2-TZVP GIAO

# =============================================================================
# PARSER: extract shieldings from ORCA NMR .out file
# =============================================================================

def parse_nmr_out(filepath):
    """
    Parse ORCA NMR output file.
    Returns dict: {'C': [σ,...], 'H': [σ,...], 'F': [σ,...], 'N': [...], ...}
    Uses the CHEMICAL SHIELDING SUMMARY block (most reliable).
    Handles both UTF-8 and UTF-16 encoded files (cluster may produce either).
    """
    text = None
    for enc in ("utf-8", "utf-16"):
        try:
            with open(filepath, encoding=enc, errors="replace") as f:
                text = f.read()
            if "ORCA" in text:
                break
        except Exception:
            continue

    if text is None:
        print(f"  [READ ERROR] {filepath}")
        return None

    if "ORCA TERMINATED NORMALLY" not in text:
        print(f"  [DID NOT CONVERGE] {filepath}")
        return None

    # Find CHEMICAL SHIELDING SUMMARY block
    # Format:
    #   Nucleus  Element    Isotropic     Anisotropy
    #       0       C           23.616        -70.681
    summary_match = re.search(
        r"CHEMICAL SHIELDING SUMMARY \(ppm\).*?Nucleus\s+Element\s+Isotropic\s+Anisotropy\s*\n"
        r"\s*-+\s*-+\s*-+\s*-+\s*\n(.*?)\n\s*\n",
        text, re.DOTALL
    )
    if not summary_match:
        print(f"  [NO SUMMARY BLOCK] {filepath}")
        return None

    shieldings = {}
    for line in summary_match.group(1).strip().split("\n"):
        parts = line.split()
        if len(parts) >= 3:
            try:
                element = parts[1]
                iso = float(parts[2])
                shieldings.setdefault(element, []).append(iso)
            except (ValueError, IndexError):
                continue

    return shieldings if shieldings else None


def parse_reference(ref_dir, filename):
    """Parse a single reference .out file and return its shieldings dict."""
    path = os.path.join(ref_dir, filename)
    if not os.path.exists(path):
        # Also try in subdirectory
        name = filename.replace("_nmr.out", "")
        path = os.path.join(ref_dir, name, filename)
    if not os.path.exists(path):
        return None
    return parse_nmr_out(path)


def get_reference_shieldings(ref_dir):
    """
    Extract reference shieldings:
        19F ref: F shielding in CFCl3
        1H  ref: H shielding in TMS (average of 12 equivalent H)
        13C ref: C shielding in TMS (average of 4 equivalent C, or just use one)
    Returns (ref_F, ref_H, ref_C)
    """
    ref_F, ref_H, ref_C = LIT_REF_F, LIT_REF_H, LIT_REF_C

    # CFCl3
    cfcl3 = parse_reference(ref_dir, "cfcl3_nmr.out")
    if cfcl3 and "F" in cfcl3:
        ref_F = float(np.mean(cfcl3["F"]))
        print(f"  CFCl3 19F reference shielding: {ref_F:.3f} ppm (computed)")
    else:
        print(f"  CFCl3 not found or failed — using literature fallback: {ref_F} ppm")

    # TMS
    tms = parse_reference(ref_dir, "tms_nmr.out")
    if tms:
        if "H" in tms:
            ref_H = float(np.mean(tms["H"]))
            print(f"  TMS 1H  reference shielding: {ref_H:.3f} ppm (computed)")
        else:
            print(f"  TMS 1H  not found — using literature fallback: {ref_H} ppm")
        if "C" in tms:
            ref_C = float(np.mean(tms["C"]))
            print(f"  TMS 13C reference shielding: {ref_C:.3f} ppm (computed)")
        else:
            print(f"  TMS 13C not found — using literature fallback: {ref_C} ppm")
    else:
        print(f"  TMS not found or failed — using literature fallbacks: H={ref_H}, C={ref_C} ppm")

    return ref_F, ref_H, ref_C


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refs", default=None,
                        help="Path to folder containing cfcl3_nmr.out and tms_nmr.out. "
                             "Defaults to REF_OUT_DIR in script.")
    parser.add_argument("--no-refs", action="store_true",
                        help="Skip reference conversion; save raw shieldings only.")
    args = parser.parse_args()

    ref_dir = args.refs or REF_OUT_DIR
    use_refs = not args.no_refs

    os.makedirs(ML_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. Get reference shieldings
    # -------------------------------------------------------------------------
    if use_refs:
        print("\n--- Reference shieldings ---")
        ref_F, ref_H, ref_C = get_reference_shieldings(ref_dir)
    else:
        ref_F = ref_H = ref_C = None
        print("Skipping reference conversion — saving raw shieldings only.")

    # -------------------------------------------------------------------------
    # 2. Parse all molecule NMR outputs
    # -------------------------------------------------------------------------
    # Support two layouts:
    #   nmr_outputs/<mol_name>/<mol_name>_nmr.out   (subdirectory per molecule)
    #   nmr_outputs/<mol_name>_nmr.out               (flat)
    out_files = sorted(
        glob.glob(os.path.join(NMR_OUT_DIR, "*", "*_nmr.out")) +
        glob.glob(os.path.join(NMR_OUT_DIR, "*_nmr.out"))
    )
    # Deduplicate
    out_files = sorted(set(out_files))
    print(f"\nFound {len(out_files)} NMR output files")

    results = []  # list of dicts
    failed  = []

    for fpath in out_files:
        fname = os.path.basename(fpath)
        mol_name = re.sub(r"_nmr\.out$", "", fname)

        shields = parse_nmr_out(fpath)
        if shields is None:
            failed.append(mol_name)
            continue

        row = {"mol_name": mol_name}
        for element, vals in shields.items():
            row[f"{element}_shields"] = vals
        results.append(row)
        print(f"  [OK] {mol_name:30s}  "
              f"F={len(shields.get('F',[]))}  "
              f"H={len(shields.get('H',[]))}  "
              f"C={len(shields.get('C',[]))}")

    if failed:
        print(f"\nFailed molecules: {failed}")

    # Sanity check: TFA CF3 should be ~-76.5 ppm vs CFCl3 (experimental)
    tfa = next((r for r in results if r["mol_name"] == "TFA"), None)
    if tfa and use_refs:
        tfa_f = tfa.get("F_shields", [])
        if tfa_f:
            tfa_delta = [ref_F - s for s in tfa_f]
            print(f"\n  [SANITY] TFA 19F shifts: {[f'{d:.1f}' for d in tfa_delta]} ppm")
            print(f"           Expected experimentally: ~-76.5 ppm (all three F equivalent)")
            if all(-95 < d < -65 for d in tfa_delta):
                print(f"           ✓ Reference looks correct")
            else:
                print(f"           ✗ WARNING: shifts outside expected range (-65 to -95 ppm) — check ref_F value")

    if not results:
        print("No results to save. Exiting.")
        return

    # -------------------------------------------------------------------------
    # 3. Build padded arrays
    # -------------------------------------------------------------------------
    mol_names = [r["mol_name"] for r in results]
    N = len(mol_names)

    def build_padded(key):
        """key e.g. 'F_shields' → padded (N, max_count) array"""
        lists = [r.get(key, []) for r in results]
        max_n = max(len(x) for x in lists) if lists else 0
        arr = np.full((N, max_n), np.nan)
        for i, vals in enumerate(lists):
            arr[i, :len(vals)] = vals
        counts = np.array([len(x) for x in lists], dtype=int)
        return arr, counts

    F_shield, n_F = build_padded("F_shields")
    H_shield, n_H = build_padded("H_shields")
    C_shield, n_C = build_padded("C_shields")

    # -------------------------------------------------------------------------
    # 4. Convert shieldings → chemical shifts  δ = σ_ref - σ
    # -------------------------------------------------------------------------
    def to_shifts(shield_arr, sigma_ref):
        if sigma_ref is None:
            return np.full_like(shield_arr, np.nan)
        return sigma_ref - shield_arr   # NaN cells stay NaN

    F_shifts = to_shifts(F_shield, ref_F)
    H_shifts = to_shifts(H_shield, ref_H)
    C_shifts = to_shifts(C_shield, ref_C)

    # -------------------------------------------------------------------------
    # 5. Save .npz
    # -------------------------------------------------------------------------
    npz_path = os.path.join(ML_DIR, "pfas_nmr_dataset.npz")
    np.savez(
        npz_path,
        mol_names = np.array(mol_names),
        F_shifts  = F_shifts,
        H_shifts  = H_shifts,
        C_shifts  = C_shifts,
        F_shield  = F_shield,
        H_shield  = H_shield,
        C_shield  = C_shield,
        n_F       = n_F,
        n_H       = n_H,
        n_C       = n_C,
        ref_F     = np.array([ref_F if ref_F else np.nan]),
        ref_H     = np.array([ref_H if ref_H else np.nan]),
        ref_C     = np.array([ref_C if ref_C else np.nan]),
    )
    print(f"\nSaved NMR dataset → {npz_path}")
    print(f"  Shape: {N} molecules")
    print(f"  Max F atoms: {F_shield.shape[1]},  H: {H_shield.shape[1]},  C: {C_shield.shape[1]}")

    # -------------------------------------------------------------------------
    # 6. Save human-readable CSV summary
    # -------------------------------------------------------------------------
    rows = []
    for i, name in enumerate(mol_names):
        f_vals = F_shifts[i, :n_F[i]] if use_refs else F_shield[i, :n_F[i]]
        h_vals = H_shifts[i, :n_H[i]] if use_refs else H_shield[i, :n_H[i]]
        c_vals = C_shifts[i, :n_C[i]] if use_refs else C_shield[i, :n_C[i]]

        rows.append({
            "molecule":     name,
            "n_F":          n_F[i],
            "F_mean_ppm":   np.nanmean(f_vals) if len(f_vals) else np.nan,
            "F_std_ppm":    np.nanstd(f_vals)  if len(f_vals) else np.nan,
            "F_min_ppm":    np.nanmin(f_vals)  if len(f_vals) else np.nan,
            "F_max_ppm":    np.nanmax(f_vals)  if len(f_vals) else np.nan,
            "n_H":          n_H[i],
            "H_mean_ppm":   np.nanmean(h_vals) if len(h_vals) else np.nan,
            "n_C":          n_C[i],
            "C_mean_ppm":   np.nanmean(c_vals) if len(c_vals) else np.nan,
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(ML_DIR, "pfas_nmr_summary.csv")
    df.to_csv(csv_path, index=False, float_format="%.3f")
    print(f"Saved summary CSV  → {csv_path}")

    # Quick sanity print
    label = "δ (ppm)" if use_refs else "σ (ppm, raw shielding)"
    print(f"\n--- Sample ({label}) ---")
    print(df[["molecule", "n_F", "F_mean_ppm", "n_H", "H_mean_ppm"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
