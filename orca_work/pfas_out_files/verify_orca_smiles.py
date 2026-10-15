import os
import re
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, rdDetermineBonds

# Suppress RDKit logging
RDLogger.DisableLog('rdApp.*')

COVALENT_RADII = {
    'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
    'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'K': 2.03, 'Na': 1.54
}

CSV_FILE = r"../PFAS_smiles.csv"

def clean_name(text):
    text = str(text)
    # Strip suffixes regardless of order before stripping punctuation
    text = re.sub(r'(_freq|_reopt|\.out)', '', text, flags=re.IGNORECASE)
    return re.sub(r'[^a-zA-Z0-9]', '', text).lower()

def normalize_formula(formula):
    """
    Strips formal charges (+/-) and metal counter-ions (K, Na) 
    for 1-to-1 formula comparison against DFT output geometries.
    """
    formula = re.sub(r'[+\-]', '', formula)
    formula = re.sub(r'K\d*|Na\d*', '', formula)
    return formula

def load_reference_smiles(csv_path):
    df = pd.read_csv(csv_path, header=None, names=["Name", "SMILES"])
    ref_dict = {}
    for _, row in df.iterrows():
        raw_name = str(row["Name"]).strip()
        smiles = str(row["SMILES"]).strip()
        
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            clean_k = clean_name(raw_name)
            ref_dict[clean_k] = {
                "raw_name": raw_name,
                "smiles": smiles,
                "inchikey": Chem.MolToInchiKey(mol),
                "canon_smiles": Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True)
            }
    return ref_dict

def parse_orca_final_xyz(out_filepath):
    with open(out_filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    blocks = re.findall(r"CARTESIAN COORDINATES \(ANGSTROMS\)\n-+\n(.*?)\n\n", content, re.DOTALL)
    if not blocks:
        blocks = re.findall(r"CARTESIAN COORDINATES \(ANGSTROMS\)\n-+\n(.*?)\n-", content, re.DOTALL)
    if not blocks:
        blocks = re.findall(r"CARTESIAN COORDINATES \(A.U.\)\n-+\n(.*?)\n-", content, re.DOTALL)

    if not blocks:
        return None

    last_block = blocks[-1].strip().split("\n")
    atoms_xyz = []
    
    for line in last_block:
        parts = line.split()
        if len(parts) < 4:
            continue

        if parts[0].isdigit():
            symbol = parts[1]
            coord_candidates = parts[2:]
        else:
            symbol = parts[0]
            coord_candidates = parts[1:]

        floats = []
        for item in coord_candidates:
            try:
                floats.append(float(item))
            except ValueError:
                continue

        if len(floats) >= 3:
            atoms_xyz.append((symbol, floats[0], floats[1], floats[2]))
            
    return atoms_xyz if len(atoms_xyz) > 0 else None

def build_mol_from_coords(atoms_xyz):
    mol = Chem.RWMol()
    positions = []
    symbols = []

    for sym, x, y, z in atoms_xyz:
        atom = Chem.Atom(sym)
        idx = mol.AddAtom(atom)
        positions.append([x, y, z])
        symbols.append(sym)

    positions = np.array(positions)
    num_atoms = len(symbols)

    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            dist = np.linalg.norm(positions[i] - positions[j])
            r1 = COVALENT_RADII.get(symbols[i], 0.7)
            r2 = COVALENT_RADII.get(symbols[j], 0.7)
            
            if dist <= 1.25 * (r1 + r2):
                mol.AddBond(i, j, Chem.BondType.SINGLE)

    raw_mol = mol.GetMol()
    try:
        rdDetermineBonds.DetermineConnectivity(raw_mol)
    except Exception:
        pass
        
    return raw_mol

def run_verification():
    print("=" * 65)
    print(" Verifying ORCA Optimized Geometries vs. CSV Reference SMILES")
    print("=" * 65)

    csv_path = CSV_FILE
    if not os.path.exists(csv_path):
        if os.path.exists("PFAS_smiles.csv"):
            csv_path = "PFAS_smiles.csv"
        else:
            print(f"FAIL Error: Could not locate CSV file at '{CSV_FILE}' or './PFAS_smiles.csv'")
            return

    ref_dict = load_reference_smiles(csv_path)
    print(f"Loaded {len(ref_dict)} reference entries from {csv_path}.\n")

    out_files = [f for f in os.listdir(".") if f.endswith(".out")]
    if not out_files:
        print("No .out files found in current working directory.")
        return

    passed, failed, skipped = 0, 0, 0

    for out_file in out_files:
        clean_file = clean_name(out_file)

        matched_key = None
        for ref_key in ref_dict:
            if ref_key in clean_file or clean_file in ref_key:
                matched_key = ref_key
                break

        if not matched_key:
            print(f"WARN  [SKIP] {out_file}: No matching CSV reference.")
            skipped += 1
            continue

        ref_info = ref_dict[matched_key]
        xyz_data = parse_orca_final_xyz(out_file)

        if not xyz_data:
            print(f"FAIL [FAIL] {out_file}: Unable to parse Cartesian coordinates.")
            failed += 1
            continue

        actual_mol = build_mol_from_coords(xyz_data)
        if actual_mol is None:
            print(f"FAIL [FAIL] {out_file}: RDKit could not reconstruct 3D bonds.")
            failed += 1
            continue

        actual_inchikey = Chem.MolToInchiKey(actual_mol)
        
        if actual_inchikey == ref_info["inchikey"]:
            print(f"OK [PASS] {out_file} (Matched: {ref_info['raw_name']})")
            passed += 1
        else:
            ref_mol = Chem.MolFromSmiles(ref_info["canon_smiles"])
            raw_ref_formula = rdMolDescriptors.CalcMolFormula(ref_mol) if ref_mol else ""
            raw_act_formula = rdMolDescriptors.CalcMolFormula(actual_mol)

            norm_ref_formula = normalize_formula(raw_ref_formula)
            norm_act_formula = normalize_formula(raw_act_formula)

            if norm_ref_formula == norm_act_formula:
                print(f"OK [PASS - Formula Match] {out_file} (Matched: {ref_info['raw_name']} | Formula: {raw_act_formula})")
                passed += 1
            else:
                print(f"\nFAIL [MISMATCH DETECTED] {out_file}")
                print(f"   Matched CSV Reference: {ref_info['raw_name']}")
                print(f"   Expected Formula: {raw_ref_formula}")
                print(f"   Actual Formula:   {raw_act_formula}\n")
                failed += 1

    print("=" * 65)
    print(f"Summary: {passed} Passed | {failed} Mismatches/Failed | {skipped} Skipped")
    print("=" * 65)

if __name__ == "__main__":
    run_verification()