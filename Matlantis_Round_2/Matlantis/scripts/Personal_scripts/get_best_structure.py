"""
get_best_structure.py — Find and save the last healthy structure before slab explosion
Usage: python get_best_structure.py /path/to/run/directory

Saves best_structure.traj and prints which round it came from.
"""

import sys
import os
import numpy as np
from ase import io

def is_healthy(atoms, min_atoms=60, max_z_range=30):
    """Check if slab is intact."""
    syms = atoms.get_chemical_symbols()
    pos  = atoms.get_positions()
    
    # Enough atoms remaining
    if len(atoms) < min_atoms:
        return False, f"too few atoms ({len(atoms)})"
    
    # Metal atoms not spread over huge range (explosion check)
    metal_idx = [i for i,s in enumerate(syms) if s in ('Zr', 'V')]
    if not metal_idx:
        return False, "no metal atoms"
    
    metal_z = pos[metal_idx, 2]
    z_range = metal_z.max() - metal_z.min()
    if z_range > max_z_range:
        return False, f"metal z-range too large ({z_range:.1f} Å)"
    
    # No atoms at extreme positions
    if pos[:,2].max() > 500 or pos[:,2].min() < -500:
        return False, "atoms at extreme z positions"
    
    return True, "ok"

def count_bound_o(atoms, cutoff=3.5):
    syms = atoms.get_chemical_symbols()
    pos  = atoms.get_positions()
    o_idx     = [i for i,s in enumerate(syms) if s=='O']
    metal_idx = [i for i,s in enumerate(syms) if s in ('Zr','V')]
    if not o_idx or not metal_idx:
        return 0
    return sum(1 for oi in o_idx
               if min(np.linalg.norm(pos[oi]-pos[mj]) for mj in metal_idx) < cutoff)

def main():
    if len(sys.argv) < 2:
        print("Usage: python get_best_structure.py /path/to/directory")
        sys.exit(0)
    
    directory = sys.argv[1].rstrip('/')
    
    # VN has 128+ atoms, ZrN has 72 atoms
    # Detect from first round
    first = os.path.join(directory, '00_MD.traj')
    if os.path.exists(first):
        atoms0 = io.read(first, index=-1)
        metal = 'V' if 'V' in atoms0.get_chemical_symbols() else 'Zr'
        min_atoms = 90 if metal == 'V' else 55
    else:
        min_atoms = 55

    print(f"Scanning {directory}")
    print(f"Min atoms threshold: {min_atoms}")
    print()
    
    best_round = None
    best_atoms = None
    best_bound = 0
    
    i = 0
    while True:
        # Check for _MD.traj first, fallback to _MD-trajectory.traj
        path = os.path.join(directory, f'{i:0>2d}_MD.traj')
        if not os.path.exists(path):
            path = os.path.join(directory, f'{i:0>2d}_MD-trajectory.traj')
            
        if not os.path.exists(path):
            break
        
        try:
            atoms = io.read(path, index=-1)
            healthy, reason = is_healthy(atoms, min_atoms=min_atoms)
            bound = count_bound_o(atoms)
            
            status = '✓' if healthy else f'✗ ({reason})'
            print(f"  Round {i:02d}: {atoms.get_chemical_formula():20s}  bound O: {bound:3d}  {status}")
            
            if healthy:
                best_round = i
                best_atoms = atoms
                best_bound = bound
        except Exception as e:
            print(f"  Round {i:02d}: ERROR - {e}")
        
        i += 1
    
    if best_atoms is not None:
        out = os.path.join(directory, 'best_structure.traj')
        io.write(out, best_atoms)
        print(f"\nBest healthy structure: Round {best_round}")
        print(f"Formula: {best_atoms.get_chemical_formula()}")
        print(f"Bound O: {best_bound}")
        print(f"Saved to: {out}")
        print(f"\nTo visualise in Matlantis:")
        print(f"  from ase import io")
        print(f"  from pfcc_extras.visualize.view import view_ngl")
        print(f"  atoms = io.read('{out}', index=-1)")
        print(f"  view_ngl([atoms], representations=['ball+stick'])")
    else:
        print("\nNo healthy structures found.")

if __name__ == '__main__':
    main()
