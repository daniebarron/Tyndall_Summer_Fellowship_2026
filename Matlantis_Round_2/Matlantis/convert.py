from ase import io
from ase.build import add_vacuum
import os
import numpy as np

systems = [
    # (input XDATCAR, output surface.traj)
    # Use first frame (index=0) = clean-ish starting structure
    ('MD_ZrN_110/295K/6O/XDATCAR',  'Matlantis/ZrN_110/295K/surface.traj'),
    ('MD_ZrN_110/363K/6O/XDATCAR',  'Matlantis/ZrN_110/363K/surface.traj'),
    # Also redo ZrN_100 properly from 6O first frame
    ('MD_ZrN_100/295K/6O/XDATCAR',  'Matlantis/ZrN_100/295K/surface.traj'),
    ('MD_ZrN_100/363K/6O/XDATCAR',  'Matlantis/ZrN_100/363K/surface.traj'),
]

for infile, outfile in systems:
    print(f'Converting {infile}...')
    atoms = io.read(infile, index=0)
    atoms.set_pbc([True, True, True])
    atoms.wrap()
    # Get z-coordinates in fractional coordinates (range 0 to 1)
    scaled_z = atoms.get_scaled_positions()[:, 2]

    # If atoms are split near 0.0 and 1.0, shift fractional positions by 0.5
    # to pull the split pieces together into the center of the unit cell
    if np.any(scaled_z < 0.2) and np.any(scaled_z > 0.8):
        scaled_z = (scaled_z + 0.5) % 1.0
        atoms.set_scaled_positions(np.column_stack([atoms.get_scaled_positions()[:, :2], scaled_z]))

    atoms.set_pbc([True, True, False])
    atoms.center(vacuum=12.5, axis=2)

    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    io.write(outfile, atoms)

    syms = atoms.get_chemical_symbols()
    counts = {s: syms.count(s) for s in set(syms)}
    print(f"  Formula: {atoms.get_chemical_formula()}")
    print(f"  Counts: {counts}")
    print(f"  Cell z: {atoms.get_cell()[2,2]:.2f} Å")
    print(f"  Written to {outfile}")
