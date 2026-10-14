"""
compare_ir_nist.py
==================
Compares ORCA-computed IR spectra (PBE0/def2-SVP) against experimental
NIST spectra for TFA, PFBA, and PFOA.

Workflow
--------
1. Parse frequencies and intensities from ORCA .out files
2. Apply PBE0/def2-SVP scaling factor (0.9614, Merrick et al. 2007)
3. Broaden stick spectrum with a Lorentzian lineshape
4. Load experimental NIST JCAMP-DX files
5. Plot computed vs experimental side by side, with peak labels

How to get NIST JCAMP files (do this once manually)
----------------------------------------------------
For each molecule, go to the NIST WebBook and download the IR spectrum
as a JCAMP-DX file:

  TFA  (trifluoroacetic acid):
    https://webbook.nist.gov/cgi/cbook.cgi?ID=C76051&Type=IRSpec&Index=0
    → click "Download" → save as: TFA_nist.jdx

  PFBA (perfluorobutanoic acid):
    https://webbook.nist.gov/cgi/cbook.cgi?ID=C375229&Type=IRSpec&Index=0
    → click "Download" → save as: PFBA_nist.jdx

  PFOA (perfluorooctanoic acid):
    https://webbook.nist.gov/cgi/cbook.cgi?ID=C335671&Type=IRSpec&Index=0
    → click "Download" → save as: PFOA_nist.jdx

Place the .jdx files in the same folder as this script, or pass their
paths via the NIST_FILES dict below.

Usage
-----
    python compare_ir_nist.py                        # uses defaults below
    python compare_ir_nist.py --out_dir my_plots/   # saves to custom folder

Output
------
    ir_comparison_TFA.png
    ir_comparison_PFBA.png
    ir_comparison_PFOA.png
    ir_comparison_all.png   (3-panel summary figure)
"""

import os
import re
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.signal import find_peaks

# ── USER CONFIGURATION ────────────────────────────────────────────────────────

# ORCA output files — adjust paths as needed
ORCA_FILES = {
    "TFA" : "TFA.out",
    "PFBA": "PFBA_Perfluorobutanoic_acid.out",
    "PFOA": "PFOA_Perfluorooctanoic_acid.out",
}

# NIST JCAMP-DX files — download manually as described above
NIST_FILES = {
    "TFA" : "TFA_nist.jdx",
    "PFBA": "PFBA_nist.jdx",
    "PFOA": "PFOA_nist.jdx",
}

# Molecule display names for plot titles
DISPLAY_NAMES = {
    "TFA" : "TFA (trifluoroacetic acid)",
    "PFBA": "PFBA (perfluorobutanoic acid)",
    "PFOA": "PFOA (perfluorooctanoic acid)",
}

# Frequency scaling factor for PBE0/def2-SVP
# Source: Merrick, Moran & Radom, J. Phys. Chem. A, 2007, 111, 11683
SCALE_FACTOR = 0.9614

# Lorentzian broadening FWHM (cm⁻¹) — 8-15 is typical for gas-phase comparison
FWHM = 10.0

# Frequency range for plots (cm⁻¹)
FREQ_MIN = 400
FREQ_MAX = 4000

# Grid resolution for broadened spectrum (cm⁻¹)
GRID_STEP = 1.0

# Number of strongest peaks to label on computed spectrum
N_LABEL_PEAKS = 6

# ─────────────────────────────────────────────────────────────────────────────


def parse_orca_ir(filepath):
    """
    Parse IR frequencies (cm⁻¹) and intensities (km/mol) from an ORCA .out file.
    Uses the LAST IR SPECTRUM block (correct for Opt+Freq jobs).
    Skips modes 0-5 (translations/rotations).

    Returns
    -------
    freqs : np.ndarray   — unscaled frequencies (cm⁻¹)
    ints  : np.ndarray   — IR intensities (km/mol)
    """
    with open(filepath, "rb") as f:
        raw = f.read()

    # Handle UTF-16 (ORCA on Windows) and UTF-8
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff") or (len(raw) > 2 and raw[1] == 0):
        text = raw.decode("utf-16-le", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")

    # Pattern: "  6:    238.61   0.000273    1.38  ..."
    ir_pattern = re.compile(
        r"^\s*(\d+):\s+(-?\d+\.\d+)\s+\S+\s+(\d+\.\d+)", re.MULTILINE
    )

    freqs, ints = [], []
    block_start = 0
    while True:
        idx = text.find("IR SPECTRUM", block_start)
        if idx == -1:
            break
        # Parse this block — 10000 chars handles up to ~200 modes (large PFAS)
        block = text[idx: idx + 10000]
        freqs_tmp, ints_tmp = [], []
        for m in ir_pattern.finditer(block):
            mode = int(m.group(1))
            freq = float(m.group(2))
            intensity = float(m.group(3))
            if mode >= 6 and freq > 0:   # skip translations/rotations and imaginary
                freqs_tmp.append(freq)
                ints_tmp.append(intensity)
        if freqs_tmp:
            freqs, ints = freqs_tmp, ints_tmp
        block_start = idx + 10

    return np.array(freqs), np.array(ints)


def parse_jcamp(filepath):
    """
    Parse a NIST JCAMP-DX file and return wavenumbers and absorbances/transmittances.
    Handles both (X++(Y..Y)) compressed and uncompressed (X, Y) formats.
    Converts transmittance to absorbance if needed.

    Returns
    -------
    wavenumbers : np.ndarray
    absorbance  : np.ndarray  (normalised 0-1)
    data_type   : str         ('transmittance' or 'absorbance')
    """
    with open(filepath, "r", errors="replace") as f:
        lines = f.readlines()

    xfactor, yfactor = 1.0, 1.0
    firstx, lastx = None, None
    npoints = None
    data_type = "absorbance"
    in_data = False
    xy_pairs = []
    xydata_format = None

    for line in lines:
        line = line.rstrip()
        up = line.upper()

        if up.startswith("##XFACTOR="):
            xfactor = float(line.split("=", 1)[1].strip())
        elif up.startswith("##YFACTOR="):
            yfactor = float(line.split("=", 1)[1].strip())
        elif up.startswith("##FIRSTX="):
            firstx = float(line.split("=", 1)[1].strip())
        elif up.startswith("##LASTX="):
            lastx = float(line.split("=", 1)[1].strip())
        elif up.startswith("##NPOINTS="):
            npoints = int(line.split("=", 1)[1].strip())
        elif "TRANSMITTANCE" in up and "##" in line:
            data_type = "transmittance"
        elif up.startswith("##XYDATA="):
            xydata_format = line.split("=", 1)[1].strip()
            in_data = True
        elif up.startswith("##XYPOINTS="):
            xydata_format = "XY"
            in_data = True
        elif up.startswith("##END"):
            in_data = False
        elif in_data and line and not line.startswith("##"):
            xy_pairs.append(line)

    # Parse XY data
    x_vals, y_vals = [], []

    if xydata_format and "(X++(Y..Y))" in xydata_format.upper():
        # JCAMP compressed format: first value is X, rest are Y
        for row in xy_pairs:
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", row)
            if not nums:
                continue
            x_start = float(nums[0]) * xfactor
            ys = [float(v) * yfactor for v in nums[1:]]
            if len(x_vals) > 0 and len(ys) > 1:
                step = (x_start - x_vals[-1]) / len(ys) if x_vals else 1.0
            else:
                step = None
            for i, y in enumerate(ys):
                if step is not None and x_vals:
                    x_vals.append(x_vals[-1] + step)
                else:
                    x_vals.append(x_start)
                y_vals.append(y)
    else:
        # Simple X, Y pairs
        for row in xy_pairs:
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", row)
            if len(nums) >= 2:
                x_vals.append(float(nums[0]) * xfactor)
                y_vals.append(float(nums[1]) * yfactor)

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)

    # Sort by wavenumber ascending
    order = np.argsort(x_vals)
    x_vals = x_vals[order]
    y_vals = y_vals[order]

    # Convert transmittance → absorbance
    if data_type == "transmittance":
        # Clip to avoid log(0)
        t = np.clip(y_vals / 100.0, 1e-6, 1.0)
        y_vals = -np.log10(t)

    # Normalise to 0-1
    y_min, y_max = y_vals.min(), y_vals.max()
    if y_max > y_min:
        y_vals = (y_vals - y_min) / (y_max - y_min)

    return x_vals, y_vals, data_type


def lorentzian(x, x0, fwhm):
    """Lorentzian lineshape centred at x0 with given FWHM."""
    gamma = fwhm / 2.0
    return (gamma**2) / ((x - x0)**2 + gamma**2)


def broaden_spectrum(freqs, intensities, grid, fwhm):
    """
    Broaden a stick spectrum into a continuous spectrum using Lorentzian lineshapes.

    Parameters
    ----------
    freqs       : array of peak positions (cm⁻¹)
    intensities : array of peak intensities (km/mol)
    grid        : array of wavenumber grid points
    fwhm        : full width at half maximum (cm⁻¹)

    Returns
    -------
    spectrum : array of broadened intensities (normalised 0-1)
    """
    spectrum = np.zeros_like(grid)
    for freq, intensity in zip(freqs, intensities):
        spectrum += intensity * lorentzian(grid, freq, fwhm)

    # Normalise
    if spectrum.max() > 0:
        spectrum /= spectrum.max()

    return spectrum


def label_peaks(ax, grid, spectrum, n=6, color="steelblue", min_freq=500):
    """Label the N strongest peaks on a broadened spectrum plot."""
    peaks, props = find_peaks(spectrum, height=0.05, distance=20)
    if len(peaks) == 0:
        return
    # Filter to reasonable frequency range
    peaks = [p for p in peaks if grid[p] >= min_freq]
    # Sort by intensity, take top N
    peaks_sorted = sorted(peaks, key=lambda p: spectrum[p], reverse=True)[:n]
    for p in peaks_sorted:
        ax.annotate(
            f"{grid[p]:.0f}",
            xy=(grid[p], spectrum[p]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color=color,
            rotation=90,
        )


def plot_comparison(molecule, orca_file, nist_file, out_dir, show=True):
    """
    Generate a comparison plot for one molecule.

    Returns fig, axes for use in multi-panel figure.
    """
    # ── Parse ORCA ────────────────────────────────────────────────────────────
    freqs_raw, intensities = parse_orca_ir(orca_file)
    freqs_scaled = freqs_raw * SCALE_FACTOR

    # Build grid and broaden
    grid = np.arange(FREQ_MIN, FREQ_MAX + GRID_STEP, GRID_STEP)
    computed = broaden_spectrum(freqs_scaled, intensities, grid, FWHM)

    # ── Parse NIST ────────────────────────────────────────────────────────────
    has_nist = os.path.isfile(nist_file)
    if has_nist:
        nist_x, nist_y, nist_type = parse_jcamp(nist_file)
        # Interpolate NIST onto same grid
        nist_interp = np.interp(grid, nist_x, nist_y,
                                left=0.0, right=0.0)
        # Normalise again after interpolation
        if nist_interp.max() > 0:
            nist_interp /= nist_interp.max()
    else:
        print(f"  ⚠  NIST file not found: {nist_file} — plotting computed only")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        2 if has_nist else 1, 1,
        figsize=(10, 6 if has_nist else 3.5),
        sharex=True,
        gridspec_kw={"hspace": 0.08}
    )
    if not has_nist:
        axes = [axes]

    # Computed spectrum (top panel)
    ax_comp = axes[0]
    ax_comp.plot(grid, computed, color="steelblue", lw=1.2,
                 label=f"Computed PBE0/def2-SVP (×{SCALE_FACTOR})")
    # Stick spectrum
    for f, i in zip(freqs_scaled, intensities):
        if FREQ_MIN <= f <= FREQ_MAX:
            ax_comp.vlines(f, 0, (i / intensities.max()) * 0.3,
                           color="steelblue", alpha=0.35, lw=0.8)
    label_peaks(ax_comp, grid, computed, n=N_LABEL_PEAKS, color="steelblue")
    ax_comp.set_ylabel("Norm. Absorbance", fontsize=9)
    ax_comp.set_ylim(-0.05, 1.25)
    ax_comp.legend(fontsize=8, loc="upper right")
    ax_comp.set_title(DISPLAY_NAMES.get(molecule, molecule), fontsize=11, pad=6)
    ax_comp.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax_comp.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))

    # Experimental spectrum (bottom panel)
    if has_nist:
        ax_nist = axes[1]
        ax_nist.plot(grid, nist_interp, color="firebrick", lw=1.2,
                     label=f"Experimental NIST ({nist_type})")
        label_peaks(ax_nist, grid, nist_interp, n=N_LABEL_PEAKS, color="firebrick")
        ax_nist.set_ylabel("Norm. Absorbance", fontsize=9)
        ax_nist.set_ylim(-0.05, 1.25)
        ax_nist.legend(fontsize=8, loc="upper right")
        ax_nist.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax_nist.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))

    # Shared x-axis formatting
    axes[-1].set_xlabel("Wavenumber (cm⁻¹)", fontsize=10)
    axes[-1].set_xlim(FREQ_MIN, FREQ_MAX)
    axes[-1].xaxis.set_major_locator(ticker.MultipleLocator(500))
    axes[-1].xaxis.set_minor_locator(ticker.MultipleLocator(100))

    for ax in axes:
        ax.tick_params(which="both", direction="in")
        ax.grid(axis="x", which="major", ls="--", alpha=0.3)

    plt.tight_layout()

    # Save
    out_path = os.path.join(out_dir, f"ir_comparison_{molecule}.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Saved → {out_path}")

    if show:
        plt.show()

    return fig, axes, has_nist


def plot_all_summary(molecules, orca_files, nist_files, out_dir):
    """3-panel summary figure with all molecules stacked."""
    fig, axes = plt.subplots(len(molecules), 2,
                              figsize=(14, 4 * len(molecules)),
                              gridspec_kw={"hspace": 0.4, "wspace": 0.15})

    grid = np.arange(FREQ_MIN, FREQ_MAX + GRID_STEP, GRID_STEP)

    for row, mol in enumerate(molecules):
        freqs_raw, intensities = parse_orca_ir(orca_files[mol])
        freqs_scaled = freqs_raw * SCALE_FACTOR
        computed = broaden_spectrum(freqs_scaled, intensities, grid, FWHM)

        ax_c = axes[row, 0]
        ax_c.plot(grid, computed, color="steelblue", lw=1.2)
        ax_c.set_title(f"{DISPLAY_NAMES.get(mol, mol)}\nComputed", fontsize=9)
        ax_c.set_ylim(-0.05, 1.25)
        label_peaks(ax_c, grid, computed, n=5, color="steelblue")

        ax_n = axes[row, 1]
        nist_file = nist_files[mol]
        if os.path.isfile(nist_file):
            nist_x, nist_y, _ = parse_jcamp(nist_file)
            nist_interp = np.interp(grid, nist_x, nist_y, left=0, right=0)
            if nist_interp.max() > 0:
                nist_interp /= nist_interp.max()
            ax_n.plot(grid, nist_interp, color="firebrick", lw=1.2)
            label_peaks(ax_n, grid, nist_interp, n=5, color="firebrick")
        else:
            ax_n.text(0.5, 0.5, "NIST file\nnot found",
                      transform=ax_n.transAxes, ha="center", va="center",
                      color="grey", fontsize=10)
        ax_n.set_title(f"{DISPLAY_NAMES.get(mol, mol)}\nNIST Experimental", fontsize=9)
        ax_n.set_ylim(-0.05, 1.25)

        for ax in [ax_c, ax_n]:
            ax.set_xlim(FREQ_MIN, FREQ_MAX)
            ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=8)
            ax.set_ylabel("Norm. Absorbance", fontsize=8)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
            ax.tick_params(which="both", direction="in", labelsize=7)
            ax.grid(axis="x", which="major", ls="--", alpha=0.3)

    fig.suptitle("ORCA PBE0/def2-SVP vs NIST Experimental IR Spectra\n"
                 f"(scaling factor = {SCALE_FACTOR}, Lorentzian FWHM = {FWHM} cm⁻¹)",
                 fontsize=11, y=1.01)

    out_path = os.path.join(out_dir, "ir_comparison_all.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Saved → {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare ORCA IR vs NIST experimental spectra")
    parser.add_argument("--out_dir", default=".", help="Output directory for plots")
    parser.add_argument("--no_show", action="store_true", help="Don't display plots interactively")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    show = not args.no_show

    molecules = ["TFA", "PFBA", "PFOA"]

    print(f"\nIR Comparison: ORCA PBE0/def2-SVP vs NIST")
    print(f"  Scaling factor : {SCALE_FACTOR}  (Merrick et al. 2007)")
    print(f"  Lorentzian FWHM: {FWHM} cm⁻¹")
    print(f"  Frequency range: {FREQ_MIN}–{FREQ_MAX} cm⁻¹")
    print()

    for mol in molecules:
        print(f"Processing {mol}...")
        orca_f = ORCA_FILES[mol]
        nist_f = NIST_FILES[mol]

        if not os.path.isfile(orca_f):
            print(f"  ⚠  ORCA file not found: {orca_f} — skipping")
            continue

        freqs, ints = parse_orca_ir(orca_f)
        print(f"  Parsed {len(freqs)} IR modes from ORCA")
        print(f"  Frequency range: {freqs.min():.1f}–{freqs.max():.1f} cm⁻¹ (unscaled)")
        print(f"  Strongest mode : {freqs[np.argmax(ints)]:.1f} cm⁻¹ "
              f"({freqs[np.argmax(ints)] * SCALE_FACTOR:.1f} scaled), "
              f"{np.max(ints):.1f} km/mol")

        plot_comparison(mol, orca_f, nist_f, args.out_dir, show=show)
        print()

    print("Generating summary figure...")
    plot_all_summary(molecules, ORCA_FILES, NIST_FILES, args.out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
