"""
collect_nmr_outputs.py
======================
Collects all *_nmr.out files from the downloaded nmr_inputs/ folder
(which has one subdirectory per molecule) into a flat nmr_outputs/ folder,
ready for build_pfas_nmr_dataset.py to parse.

Expected input layout (after scp from cluster):
    nmr_inputs/
        TFA/
            TFA_nmr.out
            TFA_nmr.inp
            TFA_nmr.log
        PFOA/
            PFOA_nmr.out
            ...

Output layout:
    nmr_outputs/
        TFA_nmr.out
        PFOA_nmr.out
        ...

Usage
-----
    python collect_nmr_outputs.py
"""

import os
import glob
import shutil

# =============================================================================
# PATHS — adjust if needed
# =============================================================================

NMR_INPUTS_DIR  = r"C:\Users\danie\Downloads\orca_work\nmr_inputs"
NMR_OUTPUTS_DIR = r"C:\Users\danie\Downloads\orca_work\nmr_outputs"

# =============================================================================
# COLLECT
# =============================================================================

def main():
    os.makedirs(NMR_OUTPUTS_DIR, exist_ok=True)

    # Find all *_nmr.out files anywhere under nmr_inputs/
    pattern = os.path.join(NMR_INPUTS_DIR, "**", "*_nmr.out")
    out_files = sorted(glob.glob(pattern, recursive=True))

    if not out_files:
        print(f"No *_nmr.out files found under {NMR_INPUTS_DIR}")
        return

    print(f"Found {len(out_files)} .out files\n")

    copied   = 0
    skipped  = 0
    missing  = []

    for src in out_files:
        fname = os.path.basename(src)
        dst   = os.path.join(NMR_OUTPUTS_DIR, fname)

        # Check the job actually finished (file non-empty)
        if os.path.getsize(src) < 1000:
            print(f"  [SKIP — too small, likely empty] {fname}")
            missing.append(fname)
            skipped += 1
            continue

        shutil.copy2(src, dst)
        print(f"  [OK] {fname}")
        copied += 1

    print(f"\nCopied {copied} files to {NMR_OUTPUTS_DIR}")
    if skipped:
        print(f"Skipped {skipped} files (empty/failed): {missing}")
    print(f"\nNext step: python build_pfas_nmr_dataset.py")

if __name__ == "__main__":
    main()
