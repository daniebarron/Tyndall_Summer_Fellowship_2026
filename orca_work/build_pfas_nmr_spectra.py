"""
build_pfas_nmr_spectra.py
─────────────────────────
Parse per-atom ¹⁹F shieldings from ORCA NMR .out files (UTF-16 encoded),
convert to chemical shifts (δ = σ_ref − σ, σ_ref = 179.716 ppm),
and broaden onto a ¹⁹F chemical shift grid to produce a (68, 251) spectral
matrix saved as pfas_nmr_spectra.npz.

Grid: −250 to 0 ppm inclusive, 1 pt/ppm  → 251 points
Broadening: Lorentzian FWHM = 1 ppm
Each spectrum is normalised so its peak height = 1.

Usage
-----
    python build_pfas_nmr_spectra.py \
        --nmr_dir   nmr_outputs/          \
        --combined  pfas_combined_dataset.npz \
        --output    pfas_nmr_spectra.npz

The mol_names order in pfas_combined_dataset.npz is used to enforce
consistent row ordering in the output matrix.
"""

import os
import re
import argparse
import numpy as np

# ── constants ──────────────────────────────────────────────────────────────────
SIGMA_REF_F  = 179.716          # CFCl₃ ¹⁹F shielding at PBE0/def2-TZVP
GRID_START   = -250.0
GRID_END     =    0.0
GRID_NPTS    =  251             # −250, −249, …, 0  (1 pt/ppm)
FWHM         =    1.0           # ppm

SHIELDING_BLOCK = "CHEMICAL SHIELDING SUMMARY"


# ── helpers ────────────────────────────────────────────────────────────────────

def _try_read(path: str) -> str:
    """Try UTF-16 first (ORCA ¹⁹F outputs), then fall back to UTF-8."""
    for enc in ("utf-16", "utf-8", "latin-1"):
        try:
            with open(path, encoding=enc) as fh:
                return fh.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise RuntimeError(f"Cannot decode {path}")


def parse_f_shieldings(path: str) -> np.ndarray:
    """
    Return an array of ¹⁹F isotropic shielding constants (ppm) from an
    ORCA output file containing a CHEMICAL SHIELDING SUMMARY block.

    Raises ValueError if no F atoms are found.
    """
    content = _try_read(path)
    idx = content.rfind(SHIELDING_BLOCK)          # last occurrence
    if idx == -1:
        raise ValueError(f"No '{SHIELDING_BLOCK}' block in {path}")

    block = content[idx:]
    # Match lines like:   9       F          269.346        106.314
    pattern = re.compile(
        r"^\s*\d+\s+F\s+([-+]?\d+\.\d+)\s+[-+]?\d+\.\d+",
        re.MULTILINE,
    )
    matches = pattern.findall(block)
    if not matches:
        raise ValueError(f"No F atoms found in shielding block of {path}")
    return np.array([float(m) for m in matches])


def shieldings_to_shifts(sigma: np.ndarray) -> np.ndarray:
    """δ = σ_ref − σ  (convention for ¹⁹F NMR relative to CFCl₃)."""
    return SIGMA_REF_F - sigma


def lorentzian_broadening(
    shifts: np.ndarray,
    grid: np.ndarray,
    fwhm: float = FWHM,
) -> np.ndarray:
    """
    Sum normalised Lorentzian peaks for each shift onto `grid`.
    Each peak has unit area (∫ L dν = 1); the returned spectrum is
    subsequently peak-normalised to max = 1.
    """
    gamma = fwhm / 2.0                 # half-width at half-maximum
    spectrum = np.zeros(len(grid), dtype=np.float64)
    for delta in shifts:
        spectrum += gamma / (np.pi * ((grid - delta) ** 2 + gamma ** 2))
    # peak-normalise
    peak = spectrum.max()
    if peak > 0:
        spectrum /= peak
    return spectrum.astype(np.float32)


# ── main ───────────────────────────────────────────────────────────────────────

def build(nmr_dir: str, combined_path: str, output_path: str) -> None:
    # Load reference mol order from combined dataset
    ref = np.load(combined_path, allow_pickle=True)
    mol_names = ref["mol_names"]                   # (68,) strings
    n_mols    = len(mol_names)

    # Build a filename → mol_name lookup from directory
    out_files = {}
    for fname in os.listdir(nmr_dir):
        if fname.endswith(".out"):
            out_files[fname] = os.path.join(nmr_dir, fname)

    # Grid
    grid = np.linspace(GRID_START, GRID_END, GRID_NPTS)   # (251,)

    spectra   = np.zeros((n_mols, GRID_NPTS), dtype=np.float32)
    f_shifts_list = []           # list of arrays for storage
    found_names   = []

    print(f"Processing {n_mols} molecules …\n")
    for i, mol_name in enumerate(mol_names):
        # Find the matching .out file – search by mol_name substring (case-insensitive)
        mol_slug = mol_name.lower().replace(" ", "_").replace("/", "_")
        matched = None
        for fname, fpath in out_files.items():
            if mol_slug[:10] in fname.lower() or fname.lower().replace(".out","") in mol_slug:
                matched = fpath
                break
        # Fallback: iterate by sorted index (assume files are in the same order)
        if matched is None:
            sorted_files = sorted(out_files.values())
            if i < len(sorted_files):
                matched = sorted_files[i]

        if matched is None:
            print(f"  [WARN] No file for mol {i}: {mol_name!r} – filling with zeros")
            f_shifts_list.append(np.array([], dtype=np.float32))
            continue

        try:
            sigma   = parse_f_shieldings(matched)
            delta   = shieldings_to_shifts(sigma)
            spec    = lorentzian_broadening(delta, grid)
            spectra[i] = spec
            f_shifts_list.append(delta.astype(np.float32))
            found_names.append(mol_name)
            print(f"  [{i:02d}] {mol_name[:50]:<50s}  nF={len(sigma):2d}  "
                  f"δ=[{delta.min():.1f}, {delta.max():.1f}] ppm")
        except Exception as exc:
            print(f"  [ERR] {mol_name!r}: {exc}")
            f_shifts_list.append(np.array([], dtype=np.float32))

    # Pad f_shifts to ragged object array for storage
    f_shifts_obj = np.empty(n_mols, dtype=object)
    for i, arr in enumerate(f_shifts_list):
        f_shifts_obj[i] = arr

    np.savez(
        output_path,
        mol_names   = mol_names,
        grid_ppm    = grid.astype(np.float32),
        spectra     = spectra,                  # (68, 251)  peak-normalised
        f_shifts    = f_shifts_obj,             # per-mol delta arrays
        sigma_ref   = np.float32(SIGMA_REF_F),
        fwhm_ppm    = np.float32(FWHM),
    )
    print(f"\nSaved → {output_path}")
    print(f"  spectra  : {spectra.shape}")
    print(f"  grid_ppm : {grid[0]:.0f} … {grid[-1]:.0f} ppm  ({GRID_NPTS} pts)")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ¹⁹F NMR spectral matrix from ORCA outputs")
    parser.add_argument("--nmr_dir",  default="nmr_outputs",
                        help="Directory containing the 68 NMR .out files")
    parser.add_argument("--combined", default="pfas_combined_dataset.npz",
                        help="pfas_combined_dataset.npz (for mol name order)")
    parser.add_argument("--output",   default="pfas_nmr_spectra.npz",
                        help="Output .npz path")
    args = parser.parse_args()

    build(args.nmr_dir, args.combined, args.output)
