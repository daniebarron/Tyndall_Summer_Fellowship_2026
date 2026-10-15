"""
PFAS Frequency Restart Script
==============================
For molecules where the geometry optimisation completed successfully but the
frequency calculation crashed (typically due to disk space), this script:

1. Scans a list of failed molecule output files
2. Extracts the final optimised geometry from each .out file
3. Writes a Freq-only .inp file in the same folder as the original calculation
4. Writes a run_failed.bat to rerun just these molecules

Usage:
    1. Place this script in the orca_work folder (same level as pfas_calculations/)
    2. Edit FAILED_MOLECULES below to list the folder names that failed
    3. Run: python generate_freq_restarts.py
    4. Then run pfas_calculations/run_failed.bat
"""

import os
import re

# CONFIGURATION

CALCULATIONS_DIR = "pfas_calculations"       # Root folder of all calculations
ORCA_PATH        = r"C:\Orca_6.1.1\orca.exe" # Path to the ORCA executable
NPROCS           = 4
MEMORY_MB        = 3000

# ORCA settings - must match original run
THEORY      = "PBE0"
BASIS       = "def2-SVP"
DISPERSION  = "D3BJ"
SCF_CONV    = "TightSCF"
RI_APPROX   = "RIJCOSX"
AUX_BASIS   = "def2/J"

# Folder names of failed molecules (the ones that need restarting)
# These are the 8 that had opt converged but freq crashed
FAILED_MOLECULES = [
    "PFPrS_Perfluoropropanesulfonic_acid",
    "PFEESA_Perfluoro-2-ethoxyethane-sulfonic_acid",
    "MeFOSE_N-methylperfluorooctanesulfonamido-ethanol",
    "MeFOSAA_N-methylperfluorooctanesulfonamid-HAc",
    "8-2_FTUCA_2H-Perfluoro-2-decenoic_acid",
    "6-2_FTAB_Capstone_B",
    "6-2_FTS_Fluorotelomer_sulfonate",
    "6-2_diPAP",
    "3-3_FTCA_Fluorotelomer_carboxylic_acid",
]

# FUNCTIONS

def extract_optimised_geometry(out_text):
    """
    Extract the final optimised Cartesian coordinates from an ORCA output file.
    Returns (xyz_block, charge, multiplicity) or (None, None, None) on failure.
    """
    lines = out_text.splitlines()

    # Find the last CARTESIAN COORDINATES (ANGSTROEM) block
    last_coords_start = None
    for i, line in enumerate(lines):
        if 'CARTESIAN COORDINATES (ANGSTROEM)' in line:
            last_coords_start = i

    if last_coords_start is None:
        return None, None, None

    # Extract coordinates - they follow a dashed line then atom entries
    # until a blank line
    coord_lines = []
    in_coords = False
    for line in lines[last_coords_start + 1:]:
        stripped = line.strip()
        if stripped.startswith('---'):
            in_coords = True
            continue
        if in_coords:
            if stripped == '':
                break
            # Should be " SYMBOL X Y Z"
            parts = stripped.split()
            if len(parts) == 4:
                coord_lines.append(f"  {parts[0]:<3} {parts[1]:>14} {parts[2]:>14} {parts[3]:>14}")
            else:
                break

    if not coord_lines:
        return None, None, None

    # Find charge and multiplicity from the input section
    charge = 0
    multiplicity = 1
    for line in lines:
        m = re.search(r'\|\s*Charge\s*=\s*(-?\d+)\s*Multiplicity\s*=\s*(\d+)', line)
        if m:
            charge = int(m.group(1))
            multiplicity = int(m.group(2))
            break

    return '\n'.join(coord_lines), charge, multiplicity


def write_freq_input(filepath, mol_name, xyz_block, charge, multiplicity):
    """Write a Freq-only ORCA input file."""
    keywords = f"! {THEORY} {BASIS} {AUX_BASIS} {RI_APPROX} Freq {DISPERSION} {SCF_CONV}"
    content = f"""{keywords}

%pal
  nprocs {NPROCS}
End

%maxcore {MEMORY_MB}

# Molecule: {mol_name}
# Freq-only restart using optimised geometry from previous run
* xyz {charge} {multiplicity}
{xyz_block}
*
"""
    with open(filepath, 'w') as f:
        f.write(content)


def write_batch_script(output_dir, jobs):
    """Write a Windows .bat file to run the failed jobs."""
    bat_path = os.path.join(output_dir, "run_failed.bat")
    lines = [
        "@echo off",
        f"echo Rerunning {len(jobs)} failed frequency calculations...",
        "echo.",
        ""
    ]
    for i, (folder, inp_file, mol_name) in enumerate(jobs, 1):
        out_file = inp_file.replace('.inp', '.out')
        lines += [
            f"echo [{i}/{len(jobs)}] Running Freq for: {mol_name}",
            f"cd /d \"{folder}\"",
            f"\"{ORCA_PATH}\" \"{inp_file}\" > \"{out_file}\"",
            f"echo Finished: {mol_name}",
            "echo.",
            ""
        ]
    lines += ["echo All done!", "pause"]
    with open(bat_path, 'w') as f:
        f.write('\n'.join(lines))
    return bat_path


# MAIN

def main():
    if not os.path.isdir(CALCULATIONS_DIR):
        print(f"ERROR: Cannot find '{CALCULATIONS_DIR}' folder.")
        return

    jobs = []
    failed = []

    for folder_name in FAILED_MOLECULES:
        folder_path = os.path.join(CALCULATIONS_DIR, folder_name)

        if not os.path.isdir(folder_path):
            print(f"[SKIP] Folder not found: {folder_path}")
            failed.append((folder_name, "Folder not found"))
            continue

        # Find the .out file in this folder
        out_files = [f for f in os.listdir(folder_path) if f.endswith('.out')]
        if not out_files:
            print(f"[SKIP] No .out file found in {folder_path}")
            failed.append((folder_name, "No .out file found"))
            continue

        # Use the largest .out file (most likely the main one)
        out_file = max(out_files, key=lambda f: os.path.getsize(os.path.join(folder_path, f)))
        out_path = os.path.join(folder_path, out_file)

        print(f"Processing: {folder_name}")
        print(f"  Reading: {out_file}")

        with open(out_path, 'rb') as f:
            raw = f.read()
        text = raw.decode('latin-1')

        # Verify opt actually converged
        if 'HURRAY' not in text:
            print(f"  WARNING: Geometry optimisation did not converge in this file!")
            print(f"           This molecule needs a full Opt+Freq rerun, not just Freq.")
            failed.append((folder_name, "Opt did not converge - needs full rerun"))
            continue

        # Extract geometry
        xyz_block, charge, multiplicity = extract_optimised_geometry(text)
        if xyz_block is None:
            print(f"  ERROR: Could not extract coordinates")
            failed.append((folder_name, "Could not extract coordinates"))
            continue

        atom_count = xyz_block.count('\n') + 1
        print(f"  Extracted {atom_count} atoms, charge={charge}, multiplicity={multiplicity}")

        # Write Freq-only input
        inp_filename = folder_name + "_freq.inp"
        inp_filepath = os.path.join(folder_path, inp_filename)
        out_filename = folder_name + "_freq.out"

        write_freq_input(inp_filepath, folder_name, xyz_block, charge, multiplicity)
        print(f"  Written: {inp_filepath}")

        jobs.append((os.path.abspath(folder_path), inp_filename, folder_name))

    # Write batch script
    bat_path = write_batch_script(CALCULATIONS_DIR, jobs)

    print(f"\n{'='*60}")
    print(f"Done! {len(jobs)} restart inputs written, {len(failed)} skipped.")
    print(f"Batch script: {os.path.abspath(bat_path)}")

    if failed:
        print(f"\nSkipped (need manual attention):")
        for name, reason in failed:
            print(f"  - {name}: {reason}")

    print(f"\nTo rerun the failed frequency calculations:")
    print(f"  {os.path.abspath(bat_path)}")
    print(f"\nNOTE: 8-2_diPAP is NOT included - its opt did not converge and needs")
    print(f"      a full Opt+Freq rerun. Consider running it with tighter settings.")

if __name__ == "__main__":
    main()
