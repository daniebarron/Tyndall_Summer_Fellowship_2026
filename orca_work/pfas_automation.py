"""
PFAS ORCA Automation Script
============================
This script:
1. Reads a CSV file containing molecule names AND their SMILES (both are supplied in the CSV)
2. Generates 3D coordinates using RDKit (MMFF force field, same as Avogadro)
3. Writes ORCA input files, each in their own folder
4. Writes a batch script to run all calculations sequentially

No PubChem lookup is performed - verified SMILES are given directly,
since automated name-based lookups can silently return the wrong structure
(e.g. metabolites/conjugates instead of the parent compound).

Requirements:
    pip install rdkit

Usage:
    python pfas_automation.py

Input CSV format (no header row), comma-separated:
    Molecule Name,SMILES,Charge
    PFOA,C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O,0
    HFPO-DA,C(=O)(C(C(F)(F)F)(OC(C(C(F)(F)F)(F)F)(F)F)F)[O-],-1

    The Charge column is optional - if omitted, charge is inferred from
    formal charges in the SMILES itself (e.g. [O-] contributes -1).

Output:
    A folder called 'pfas_calculations' containing one subfolder per molecule,
    each with an ORCA .inp file. Also writes run_all.bat to run them all.
"""

import os
import re
import csv
from rdkit import Chem
from rdkit.Chem import AllChem

# CONFIGURATION

INPUT_CSV       = "PFAS_smiles.csv"          # CSV with name,smiles[,charge]
OUTPUT_DIR      = "pfas_calculations"        # Root folder for all calculations
ORCA_PATH       = r"C:\Orca_6.1.1\orca.exe" # Path to the ORCA executable
NPROCS          = 4                          # Number of CPU cores to use
MEMORY_MB       = 3000                       # Memory per core in MB

# ORCA calculation settings
THEORY          = "PBE0"
BASIS           = "def2-SVP"
DISPERSION      = "D3BJ"
SCF_CONV        = "TightSCF"
RI_APPROX       = "RIJCOSX"
AUX_BASIS       = "def2/J"

# Geometry convergence (loose, for speed across many molecules)
TOLE            = "1e-4"    # Energy tolerance (Ha)
TOLRMSG         = "1e-2"    # RMS gradient
TOLMAXG         = "3e-2"    # Max gradient
TOLRMSD         = "1e-3"    # RMS displacement
TOLMAXD         = "3e-3"    # Max displacement

# FUNCTIONS

def sanitize_folder_name(name):
    """Convert molecule name to a safe folder name, valid on Windows/Mac/Linux."""
    # Replace ":" with "-" instead of dropping it, so "6:2 FTS" -> "6-2_FTS" stays readable
    name = name.replace(':', '-')
    # Remove all other characters not allowed in Windows folder names: \ / : * ? " < > |
    name = re.sub(r'[\\/:*?"<>|]', '', name)
    # Remove any remaining non-word characters except space, hyphen, underscore, period
    name = re.sub(r'[^\w\s\-.]', '', name)
    name = re.sub(r'[\s]+', '_', name)
    # Windows folder names can't end with a period or space
    name = name.strip('_. ')
    return name[:80]

def read_molecule_csv(path):
    """
    Read the input CSV of (name, smiles[, charge]) rows.
    Returns a list of dicts: {"name": ..., "smiles": ..., "charge": int or None}
    Charge is None if not provided in the CSV - it will then be inferred
    from formal charges in the SMILES string itself.
    """
    rows = []
    with open(path, 'rb') as f:
        raw = f.read()
    text = raw.decode('latin-1')
    reader = csv.reader(text.splitlines())
    for line_num, row in enumerate(reader, 1):
        if not row or not row[0].strip():
            continue
        name = row[0].strip().strip('"')
        if len(row) < 2 or not row[1].strip():
            print(f"  [CSV] Line {line_num}: '{name}' has no SMILES - skipping")
            continue
        smiles = row[1].strip().strip('"')
        charge = None
        if len(row) >= 3 and row[2].strip():
            try:
                charge = int(row[2].strip())
            except ValueError:
                pass
        rows.append({"name": name, "smiles": smiles, "charge": charge})
    return rows

def get_formal_charge(mol):
    """Sum formal charges across all atoms in an RDKit mol."""
    return sum(atom.GetFormalCharge() for atom in mol.GetAtoms())

# Common counterions to strip when a SMILES contains multiple disconnected fragments
COMMON_COUNTERIONS = {
    "[K+]", "[Na+]", "[Li+]", "[NH4+]",
    "[Cl-]", "[Br-]", "[I-]",
}

def strip_counterions(smiles):
    """
    If a SMILES contains multiple disconnected fragments (separated by '.'),
    such as a salt like a potassium sulfonate, remove known small counterions
    and keep the largest organic fragment instead. This avoids RDKit placing
    a stray ion at an arbitrary 3D position, which is not meaningful for a
    gas-phase ORCA optimisation/frequency calculation.

    The charge of the kept organic fragment is used as-is (it already reflects
    its own formal charges, e.g. a built-in [O-]) - no further adjustment is
    needed since the counterion's own charge is simply discarded along with it.

    Returns (clean_smiles, was_modified, unrecognised_extra_fragments)
    """
    fragments = smiles.split('.')
    if len(fragments) == 1:
        return smiles, False, []

    unrecognised = []
    kept_fragments = [f for f in fragments if f not in COMMON_COUNTERIONS]

    if not kept_fragments:
        # Shouldn't happen, but guard anyway
        return smiles, False, []

    # If more than one non-counterion fragment remains, keep the largest by atom count
    # and flag the rest as unrecognised
    if len(kept_fragments) > 1:
        mols = [(f, Chem.MolFromSmiles(f)) for f in kept_fragments]
        mols = [(f, m) for f, m in mols if m is not None]
        if not mols:
            return smiles, False, kept_fragments
        mols.sort(key=lambda fm: fm[1].GetNumAtoms(), reverse=True)
        main_frag = mols[0][0]
        unrecognised = [f for f, _ in mols[1:]]
    else:
        main_frag = kept_fragments[0]

    return main_frag, True, unrecognised

def smiles_to_xyz_block(smiles):
    """
    Convert SMILES to a 3D XYZ coordinate block using RDKit.
    Uses ETKDGv3 embedding + MMFF94 optimisation (same as Avogadro default).
    Returns coordinate string or None on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    # Generate 3D coordinates
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol, params)
    if result != 0:
        # Fallback: try random coords
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if result != 0:
            return None

    # MMFF94 force field optimisation
    ff_result = AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    if ff_result == -1:
        # MMFF failed, try UFF
        AllChem.UFFOptimizeMolecule(mol, maxIters=2000)

    conf = mol.GetConformer()
    lines = []
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        lines.append(f"  {atom.GetSymbol():<3} {pos.x:>12.6f} {pos.y:>12.6f} {pos.z:>12.6f}")
    return '\n'.join(lines)

def write_orca_input(filepath, mol_name, xyz_block, charge, multiplicity=1):
    """Write a complete ORCA input file."""
    keywords = f"! {THEORY} {BASIS} {AUX_BASIS} {RI_APPROX} Opt Freq {DISPERSION} {SCF_CONV}"

    content = f"""{keywords}

%pal
  nprocs {NPROCS}
End

%maxcore {MEMORY_MB}

%geom
  TolE    {TOLE}
  TolRMSG {TOLRMSG}
  TolMaxG {TOLMAXG}
  TolRMSD {TOLRMSD}
  TolMaxD {TOLMAXD}
End

# Molecule: {mol_name}
* xyz {charge} {multiplicity}
{xyz_block}
*
"""
    with open(filepath, 'w') as f:
        f.write(content)

def write_batch_script(output_dir, jobs):
    """Write a Windows .bat file to run all ORCA jobs sequentially."""
    bat_path = os.path.join(output_dir, "run_all.bat")
    lines = [
        "@echo off",
        "echo Starting PFAS ORCA calculations...",
        f"echo Total jobs: {len(jobs)}",
        "echo.",
        ""
    ]
    for i, (folder, inp_file, mol_name) in enumerate(jobs, 1):
        out_file = inp_file.replace('.inp', '.out')
        lines += [
            f"echo [{i}/{len(jobs)}] Running: {mol_name}",
            f"cd /d \"{folder}\"",
            f"\"{ORCA_PATH}\" \"{inp_file}\" > \"{out_file}\"",
            f"echo Finished: {mol_name}",
            "echo.",
            ""
        ]
    lines += [
        "echo All calculations complete!",
        "pause"
    ]
    with open(bat_path, 'w') as f:
        f.write('\n'.join(lines))
    return bat_path

# MAIN

def main():
    # Read molecule + SMILES list
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: Cannot find '{INPUT_CSV}'. Make sure it's in the same folder as this script.")
        return

    entries = read_molecule_csv(INPUT_CSV)
    print(f"Found {len(entries)} molecules to process.\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    jobs = []
    failed = []
    salt_adjusted = []

    for i, entry in enumerate(entries, 1):
        mol_name = entry["name"]
        smiles = entry["smiles"]
        csv_charge = entry["charge"]

        print(f"[{i}/{len(entries)}] {mol_name}")

        # 1. Check for multi-fragment SMILES (salts) and strip known counterions
        clean_smiles, was_modified, unrecognised = strip_counterions(smiles)
        if was_modified:
            print(f"  NOTE: Multiple fragments detected in SMILES - stripped counterion(s).")
            print(f"        Original: {smiles}")
            print(f"        Using organic fragment: {clean_smiles}")
            if unrecognised:
                print(f"  WARNING: Additional non-counterion fragment(s) found and DISCARDED: {unrecognised}")
                print(f"           Verify this is correct - the structure may be incomplete!")
        smiles = clean_smiles
        if was_modified:
            salt_adjusted.append((mol_name, smiles, unrecognised))

        # 2. Parse SMILES to validate and get formal charge
        check_mol = Chem.MolFromSmiles(smiles)
        if check_mol is None:
            print(f"  SKIPPED: Invalid SMILES - could not parse")
            failed.append((mol_name, "Invalid SMILES"))
            continue

        inferred_charge = get_formal_charge(check_mol)
        charge = csv_charge if csv_charge is not None else inferred_charge

        if csv_charge is not None and csv_charge != inferred_charge:
            print(f"  WARNING: CSV charge ({csv_charge}) differs from inferred charge ({inferred_charge}). Using CSV value: {csv_charge}")

        print(f"  SMILES: {smiles[:60]}{'...' if len(smiles)>60 else ''}, Charge: {charge}")

        # 3. Generate 3D coordinates
        xyz_block = smiles_to_xyz_block(smiles)
        if xyz_block is None:
            print(f"  SKIPPED: RDKit could not generate 3D coordinates")
            failed.append((mol_name, "3D coordinate generation failed"))
            continue

        atom_count = xyz_block.count('\n') + 1
        print(f"  Generated {atom_count} atoms")

        # 4. Create output folder and write input file
        folder_name = sanitize_folder_name(mol_name)
        folder_path = os.path.join(OUTPUT_DIR, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        inp_filename = folder_name + ".inp"
        inp_filepath = os.path.join(folder_path, inp_filename)

        write_orca_input(inp_filepath, mol_name, xyz_block, charge)
        print(f"  Written: {inp_filepath}")

        jobs.append((os.path.abspath(folder_path), inp_filename, mol_name))

    # 4. Write batch run script
    bat_path = write_batch_script(OUTPUT_DIR, jobs)

    # 5. Summary
    print(f"\n{'='*60}")
    print(f"Done! {len(jobs)} input files written, {len(failed)} failed.")
    print(f"Output folder: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Batch script:  {os.path.abspath(bat_path)}")

    if failed:
        print(f"\nFailed molecules (will need manual handling):")
        for name, reason in failed:
            print(f"  - {name}: {reason}")

    if salt_adjusted:
        print(f"\nMolecules with counterions stripped (double-check these):")
        for name, used_smiles, unrecognised in salt_adjusted:
            flag = "  <- had unrecognised extra fragment, verify structure!" if unrecognised else ""
            print(f"  - {name}{flag}")

    print(f"\nTo run all calculations, open a Command Prompt and run:")
    print(f"  {os.path.abspath(bat_path)}")

if __name__ == "__main__":
    main()
