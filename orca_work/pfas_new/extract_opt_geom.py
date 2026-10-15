"""
extract_opt_geom.py
====================
Extracts the final optimised geometry from a completed ORCA IR/freq .out file
and writes a ready-to-submit NMR .inp file using that geometry.

Usage
-----
    python extract_opt_geom.py <ir_out_file> <nmr_inp_file>

Examples
--------
    python extract_opt_geom.py 6_2_FTSA_ir.out 6_2_FTSA_nmr.inp
    python extract_opt_geom.py 6_2_FTAB_protonated_ir.out 6_2_FTAB_protonated_nmr.inp
    python extract_opt_geom.py DONA_3H_interior_ir.out DONA_3H_interior_nmr.inp

The script will:
    1. Check the IR job completed successfully (looks for 'ORCA TERMINATED NORMALLY')
    2. Extract charge and multiplicity from the input block
    3. Extract the FINAL optimised geometry (last CARTESIAN COORDINATES block)
    4. Write a NMR .inp file ready to submit
    5. Print a summary so you can sanity-check before submitting
"""

import re
import sys
import os

# NMR settings - matches your TFA_nmr.inp exactly
NMR_TEMPLATE = """\
# PFAS NMR calculation: {name}
# PBE0/def2-TZVP GIAO NMR
# Geometry from: {source}

! NMR PBE0 def2-TZVP def2/J RIJCOSX NormalPrint

%pal
  nprocs {nprocs}
end

%maxcore 4000

* xyz {charge} {mult}
{xyz}
*
"""

def read_out(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()

def check_normal_termination(content):
    return "ORCA TERMINATED NORMALLY" in content

def get_charge_mult(content):
    """Read charge and multiplicity from the input block near the top."""
    # Matches: * xyz <charge> <mult>
    m = re.search(r"\*\s+xyz\s+(-?\d+)\s+(\d+)", content)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise ValueError("Could not find charge/multiplicity in .out file.")

def get_job_name(out_path):
    """Derive a clean name from the filename."""
    base = os.path.basename(out_path)
    # Remove _ir.out or _freq.out suffix
    name = re.sub(r"[_\-](ir|freq|opt)\.out$", "", base, flags=re.IGNORECASE)
    name = re.sub(r"\.out$", "", name)
    return name

def extract_final_geometry(content):
    """
    Extract the LAST 'CARTESIAN COORDINATES (ANGSTROEM)' block -
    this is the final optimised geometry after the last step.
    Returns a list of (element, x, y, z) tuples.
    """
    # Find all occurrences of the coordinates block
    pattern = re.compile(
        r"CARTESIAN COORDINATES \(ANGSTROEM\)\s*-+\s*(.*?)\s*-{5,}",
        re.DOTALL
    )
    matches = list(pattern.finditer(content))

    if not matches:
        raise ValueError(
            "No 'CARTESIAN COORDINATES (ANGSTROEM)' block found.\n"
            "Make sure the IR job completed and the .out file is complete."
        )

    # Take the LAST match = final optimised geometry
    last_block = matches[-1].group(1)
    atoms = []
    for line in last_block.strip().split("\n"):
        parts = line.split()
        if len(parts) == 4:
            elem, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append((elem, x, y, z))

    if not atoms:
        raise ValueError("Geometry block found but no atom lines could be parsed.")

    return atoms

def format_xyz_block(atoms):
    lines = []
    for elem, x, y, z in atoms:
        lines.append(f"  {elem:2s}  {x:14.8f}  {y:14.8f}  {z:14.8f}")
    return "\n".join(lines)

def guess_nprocs(natoms):
    """Scale nprocs with molecule size, matching your cluster setup."""
    if natoms < 35:
        return 4
    elif natoms < 50:
        return 8
    else:
        return 16

def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_opt_geom.py <ir_out_file> <nmr_inp_file>")
        print()
        print("Example:")
        print("  python extract_opt_geom.py 6_2_FTSA_ir.out 6_2_FTSA_nmr.inp")
        sys.exit(1)

    out_path = sys.argv[1]
    nmr_path = sys.argv[2]

    if not os.path.isfile(out_path):
        print(f"ERROR: File not found: {out_path}")
        sys.exit(1)

    print(f"Reading: {out_path}")
    content = read_out(out_path)

    # Check job finished
    if not check_normal_termination(content):
        print("WARNING: 'ORCA TERMINATED NORMALLY' not found in .out file.")
        print("         The job may not have finished or may have crashed.")
        print("         Proceeding anyway - check the geometry carefully.\n")
    else:
        print("  OK ORCA terminated normally")

    # Extract info
    charge, mult = get_charge_mult(content)
    print(f"  OK Charge={charge}, Multiplicity={mult}")

    atoms = extract_final_geometry(content)
    natoms = len(atoms)
    print(f"  OK Extracted {natoms} atoms from final optimised geometry")

    # Count elements for sanity check
    from collections import Counter
    elem_counts = Counter(a[0] for a in atoms)
    formula = "".join(f"{e}{n}" for e, n in sorted(elem_counts.items()))
    print(f"  OK Formula: {formula}")

    # Write NMR inp
    name     = get_job_name(out_path)
    nprocs   = guess_nprocs(natoms)
    xyz_block = format_xyz_block(atoms)

    nmr_content = NMR_TEMPLATE.format(
        name=name,
        source=os.path.basename(out_path),
        charge=charge,
        mult=mult,
        nprocs=nprocs,
        xyz=xyz_block,
    )

    with open(nmr_path, "w") as f:
        f.write(nmr_content)

    print(f"\n  OK NMR input written to: {nmr_path}")
    print(f"    nprocs = {nprocs}  (adjust in file if needed)")
    print()
    print("Next steps:")
    print(f"  1. Check {nmr_path} looks correct")
    print(f"  2. Update the --job-name and paths in your submit script")
    print(f"  3. sbatch submit_{name}_nmr.sh")


if __name__ == "__main__":
    main()
