"""
build_pfas_ir_dataset.py
========================
Extracts IR frequencies and intensities from 68 ORCA .out files in a single
flat folder, broadens onto a fixed wavenumber grid, matches to SMILES, and
saves a structured ML-ready dataset.

Usage
-----
    python build_pfas_ir_dataset.py

Outputs (written to OUTPUT_DIR):
    pfas_ir_dataset.npz      - main ML dataset (spectra + metadata)
    pfas_ir_metadata.csv     - per-molecule summary table
    pfas_ir_spectra.png      - overview plot of all 68 spectra
    extraction_report.txt    - log of what was found / any warnings
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# PATHS - edit for the local machine

OUT_DIR    = r"C:\Users\danie\Downloads\orca_work\pfas_out_files"
SMILES_CSV = r"C:\Users\danie\Downloads\orca_work\PFAS_smiles.csv"
OUTPUT_DIR = r"C:\Users\danie\Downloads\orca_work\ml_dataset"

# SPECTRUM PARAMETERS

SCALE_FACTOR = 0.9614    # PBE0/def2-SVP literature scale factor
FREQ_MIN     = 500.0     # cm-1
FREQ_MAX     = 4000.0    # cm-1
N_POINTS     = 3501      # 1 cm-1 spacing
LORENTZ_FWHM = 20.0      # cm-1 Lorentzian broadening FWHM

# HELPERS

def lorentzian(x, x0, fwhm):
    gamma = fwhm / 2.0
    return (gamma / np.pi) / ((x - x0)**2 + gamma**2)


def broaden_spectrum(freqs, intensities, grid, fwhm=LORENTZ_FWHM):
    spectrum = np.zeros_like(grid)
    for f, i in zip(freqs, intensities):
        if FREQ_MIN <= f <= FREQ_MAX:
            spectrum += i * lorentzian(grid, f, fwhm)
    if spectrum.max() > 0:
        spectrum /= spectrum.max()
    return spectrum


def parse_orca_ir(filepath):
    """
    Parse ORCA 5 .out file for IR spectrum.

    IR SPECTRUM block format:
     Mode   freq       eps      Int      T**2         TX        TY        TZ
    -------------------------------------------------------------------
      6:     31.37   0.000330    1.67  0.003279  ( 0.004197 -0.008785  0.056430)

    Int (km/mol) is the 3rd numeric column after the mode index and freq.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except Exception as e:
        return None, None, str(e)

    freqs, intens = [], []
    in_ir_block = False
    found_data = False

    for line in lines:
        if "IR SPECTRUM" in line and "YES" not in line:
            in_ir_block = True
            found_data = False
            continue

        if not in_ir_block:
            continue

        stripped = line.strip()

        if stripped.startswith("-") or stripped.startswith("Mode"):
            continue

        if stripped == "":
            if found_data:
                break
            continue

        # " 6: 31.37 0.000330 1.67 ..."
        m = re.match(
            r"\s*(\d+):\s+([\d.]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)",
            line
        )
        if m:
            freq = float(m.group(2))
            intensity = float(m.group(4))
            if freq > 0:
                freqs.append(freq * SCALE_FACTOR)
                intens.append(intensity)
                found_data = True

    if not freqs:
        return None, None, "IR block not found or empty"

    return np.array(freqs), np.array(intens), None


def normalise_name(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def match_smiles(stem, smiles_df):
    """
    Find the best matching SMILES entry for a filename stem.
    Uses longest common substring (length >= 3) between normalised names.
    Falls back to exact short names like TFA, TFMS.
    """
    norm_stem = normalise_name(stem)
    best_score = 0
    best_smiles = ""
    best_name = ""

    for _, row in smiles_df.iterrows():
        norm_csv = row["norm"]
        score = 0
        min_len = min(len(norm_csv), len(norm_stem))
        for length in range(min_len, 2, -1):
            for start in range(len(norm_csv) - length + 1):
                token = norm_csv[start:start + length]
                if token in norm_stem:
                    score = length
                    break
            if score:
                break
        if score > best_score:
            best_score = score
            best_smiles = row["smiles"]
            best_name = row["name"]

    return best_name, best_smiles


# MAIN

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_lines = []

    def log(msg):
        print(msg)
        report_lines.append(msg)

    # load SMILES
    smiles_df = pd.read_csv(SMILES_CSV, header=None, names=["name", "smiles"])
    smiles_df["norm"] = smiles_df["name"].apply(normalise_name)
    log(f"Loaded {len(smiles_df)} SMILES entries")

    # find all .out files
    out_files = sorted(glob.glob(os.path.join(OUT_DIR, "*.out")))
    log(f"Found {len(out_files)} .out files in {OUT_DIR}\n")

    # wavenumber grid
    grid = np.linspace(FREQ_MIN, FREQ_MAX, N_POINTS)

    # extract spectra
    records  = []
    spectra  = []

    for filepath in out_files:
        filename = os.path.basename(filepath)
        # strip .out and _freq suffix to get molecule stem
        stem = re.sub(r"_freq$", "", filename[:-4])

        freqs, intens, err = parse_orca_ir(filepath)

        if freqs is None:
            log(f"  [FAIL]  {filename}  --  {err}")
            continue

        csv_name, smiles = match_smiles(stem, smiles_df)
        spectrum = broaden_spectrum(freqs, intens, grid)

        log(f"  [OK]  {filename}  ({len(freqs)} modes)  ->  {csv_name}")

        records.append({
            "filename":     filename,
            "stem":         stem,
            "csv_name":     csv_name,
            "smiles":       smiles,
            "n_modes":      len(freqs),
            "freq_min_cm1": float(freqs.min()),
            "freq_max_cm1": float(freqs.max()),
        })
        spectra.append(spectrum)

    n_ok = len(records)
    log(f"\nSuccessfully extracted {n_ok} / {len(out_files)} spectra")

    if n_ok == 0:
        log("ERROR: No spectra extracted. Check OUT_DIR path.")
        return

    # save dataset
    spectra_array = np.array(spectra, dtype=np.float32)
    meta_df = pd.DataFrame(records)

    npz_path = os.path.join(OUTPUT_DIR, "pfas_ir_dataset.npz")
    np.savez(
        npz_path,
        wavenumbers  = grid.astype(np.float32),
        spectra      = spectra_array,
        smiles       = np.array(meta_df["smiles"].tolist()),
        filenames    = np.array(meta_df["filename"].tolist()),
        csv_names    = np.array(meta_df["csv_name"].tolist()),
    )
    log(f"\nSaved ML dataset  ->  {npz_path}")
    log(f"  spectra shape: {spectra_array.shape}")

    csv_path = os.path.join(OUTPUT_DIR, "pfas_ir_metadata.csv")
    meta_df.to_csv(csv_path, index=False)
    log(f"Saved metadata    ->  {csv_path}")

    # overview plot
    ncols = 4
    nrows = int(np.ceil(n_ok / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(20, nrows * 2.5), sharex=True)
    axes_flat = axes.flatten()

    for i, (spectrum, rec) in enumerate(zip(spectra, records)):
        ax = axes_flat[i]
        ax.plot(grid, spectrum, lw=0.8, color="steelblue")
        ax.set_title(rec["csv_name"].split("(")[0].strip(), fontsize=6, pad=2)
        ax.set_xlim(FREQ_MIN, FREQ_MAX)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=6)

    for j in range(n_ok, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.supxlabel("Wavenumber (cm$^{-1}$)", fontsize=9)
    fig.supylabel("Normalised Intensity", fontsize=9)
    fig.suptitle(
        f"PFAS IR Spectra  |  PBE0/def2-SVP x {SCALE_FACTOR} scale factor  |"
        f"  Lorentzian FWHM {LORENTZ_FWHM} cm$^{{-1}}$",
        fontsize=9
    )
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "pfas_ir_spectra.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved plot        ->  {plot_path}")

    # report
    report_path = os.path.join(OUTPUT_DIR, "extraction_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    log(f"Saved report      ->  {report_path}")

    # summary
    log("\n" + "=" * 55)
    log("DATASET SUMMARY")
    log("=" * 55)
    log(f"  Molecules  :  {n_ok}")
    log(f"  Grid       :  {FREQ_MIN}-{FREQ_MAX} cm-1, {N_POINTS} points")
    log(f"  Scale      :  {SCALE_FACTOR}")
    log(f"  Broadening :  Lorentzian FWHM {LORENTZ_FWHM} cm-1")
    log(f"  X shape    :  ({n_ok}, {N_POINTS})")
    log(f"\nLoad with:")
    log(f"  data = np.load(r'{npz_path}', allow_pickle=True)")
    log(f"  X    = data['spectra']    # ({n_ok}, {N_POINTS})")
    log(f"  Y    = data['csv_names']  # ({n_ok},)")


if __name__ == "__main__":
    main()
