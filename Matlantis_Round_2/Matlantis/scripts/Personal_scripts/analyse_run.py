"""
analyse_run.py - Analysis of Matlantis ZrN/VN oxidation runs

Usage:
  python analyse_run.py /path/to/run/directory         # single run (or seed dir)
  python analyse_run.py --seeds /path/to/system/temp   # average across seed_1..seed_N
  python analyse_run.py --summary [base_dir]           # full table, all systems
  python analyse_run.py --summary-seeds [base_dir]     # full table with seed stats

Examples:
  python analyse_run.py /home/jovyan/Matlantis/ZrN_100/363K/seed_1
  python analyse_run.py --seeds /home/jovyan/Matlantis/ZrN_100/363K
  python analyse_run.py --summary /home/jovyan/Matlantis
  python analyse_run.py --summary-seeds /home/jovyan/Matlantis
"""

import sys
import os
import json
import numpy as np

try:
    from ase import io
except ImportError:
    print("ERROR: ASE not installed. Run: pip install ase")
    sys.exit(1)


# Ji reference values (Table 1)
JI_REFERENCE = {
    'ZrN_100': {'295': 8,  '363': 12, '1023': 28},
    'ZrN_110': {'295': 18, '363': 20, '1023': 32},
    'VN_111':  {'295': 11, '363': 18, '873':  12},
}


def get_metal(syms):
    if 'Zr' in syms: return 'Zr'
    if 'V'  in syms: return 'V'
    return None


def count_bound_o(atoms, bond_cutoff=3.5):
    syms = atoms.get_chemical_symbols()
    pos  = atoms.get_positions()
    o_idx     = [i for i,s in enumerate(syms) if s == 'O']
    metal_idx = [i for i,s in enumerate(syms) if s in ('Zr', 'V')]
    if not o_idx or not metal_idx:
        return 0
    return sum(1 for oi in o_idx
               if min(np.linalg.norm(pos[oi]-pos[mj]) for mj in metal_idx) < bond_cutoff)


def is_healthy(atoms, min_atoms=55):
    """Check slab is intact - not exploded."""
    syms = atoms.get_chemical_symbols()
    pos  = atoms.get_positions()
    if len(atoms) < min_atoms:
        return False
    metal_idx = [i for i,s in enumerate(syms) if s in ('Zr','V')]
    if not metal_idx:
        return False
    metal_z = pos[metal_idx, 2]
    if metal_z.max() - metal_z.min() > 30:
        return False
    if pos[:,2].max() > 500 or pos[:,2].min() < -500:
        return False
    return True


def analyse_md_log(log_path):
    """Parse MD.log - return per-round stats for most recent run only."""
    with open(log_path) as f:
        content = f.read()

    rounds = content.split('Time[ps]      Etot[eV]     Epot[eV]     Ekin[eV]    T[K]')
    if len(rounds) < 2:
        return []

    last_reset = 1
    for i, r in enumerate(rounds[1:], 1):
        lines = [l for l in r.strip().split('\n')
                 if l.strip() and not l.startswith('Time')]
        if not lines:
            continue
        try:
            if float(lines[0].split()[0]) == 0.0:
                last_reset = i
        except:
            pass

    stats = []
    for i, r in enumerate(rounds[last_reset:], 1):
        lines = [l for l in r.strip().split('\n')
                 if l.strip() and not l.startswith('Time')]
        if not lines:
            continue
        try:
            last  = lines[-1].split()
            temps = [float(l.split()[4]) for l in lines if len(l.split()) == 5]
            stats.append({
                'round':    i,
                'steps':    len(lines),
                't_end':    float(last[0]),
                'epot_end': float(last[2]),
                'T_avg':    np.mean(temps),
                'T_std':    np.std(temps),
            })
        except:
            pass
    return stats


def analyse_structures(directory):
    """Read all round _MD.traj files - return bound O progression."""
    results = []
    i = 0
    while True:
        path = os.path.join(directory, f'{i:0>2d}_MD.traj')
        if not os.path.exists(path):
            break
        try:
            atoms = io.read(path, index=-1)
            syms  = atoms.get_chemical_symbols()
            pos   = atoms.get_positions()
            metal = get_metal(syms)

            o_idx     = [j for j,s in enumerate(syms) if s == 'O']
            metal_idx = [j for j,s in enumerate(syms) if s in ('Zr', 'V')]
            n_idx     = [j for j,s in enumerate(syms) if s == 'N']
            n_start   = 36 if metal == 'Zr' else 64

            metal_z = pos[metal_idx, 2]
            bound   = count_bound_o(atoms)

            bonded_dists = []
            if o_idx and metal_idx:
                for oi in o_idx:
                    for mj in metal_idx:
                        d = np.linalg.norm(pos[oi] - pos[mj])
                        if d < 2.5:
                            bonded_dists.append(d)

            results.append({
                'round':       i,
                'formula':     atoms.get_chemical_formula(),
                'n_O':         len(o_idx),
                'bound_O':     bound,
                'N_lost':      n_start - len(n_idx),
                'metal_z_max': metal_z.max(),
                'metal_z_min': metal_z.min(),
                'healthy':     is_healthy(atoms, min_atoms=90 if metal=='V' else 55),
                'bond_min':    min(bonded_dists) if bonded_dists else None,
                'bond_max':    max(bonded_dists) if bonded_dists else None,
            })
        except Exception as e:
            print(f"  Warning: could not read {path}: {e}")
        i += 1
    return results


def best_bound_o(structs):
    """Return best bound O from healthy rounds only."""
    healthy = [s for s in structs if s['healthy']]
    if healthy:
        return max(s['bound_O'] for s in healthy)
    return max(s['bound_O'] for s in structs) if structs else None


def check_final_relaxed(directory):
    path = os.path.join(directory, 'final_relaxed.traj')
    if not os.path.exists(path):
        return None
    atoms = io.read(path, index=-1)
    syms  = atoms.get_chemical_symbols()
    metal = get_metal(syms)
    n_start = 36 if metal == 'Zr' else 64
    return {
        'formula': atoms.get_chemical_formula(),
        'bound_O': count_bound_o(atoms),
        'N_lost':  n_start - syms.count('N'),
    }


def check_relax_log(directory):
    path = os.path.join(directory, 'relax_final.log')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        content = f.read()
    blocks    = content.split('Step     Time          Energy          fmax')
    last_lines = [l for l in blocks[-1].strip().split('\n') if l.strip()]
    if not last_lines:
        return None
    try:
        last = last_lines[-1].split()
        return {
            'steps':     int(last[1]),
            'fmax':      float(last[4]),
            'converged': float(last[4]) < 0.05,
        }
    except:
        return None


def read_seed_info(seed_dir):
    path = os.path.join(seed_dir, 'seed_info.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def guess_system(directory):
    parts  = directory.replace('\\', '/').split('/')
    system = 'unknown'
    temp   = 'unknown'
    for p in parts:
        if any(x in p for x in ('ZrN_100', 'ZrN_110', 'VN_111')):
            system = p
        if p.endswith('K') and p[:-1].isdigit():
            temp = p[:-1]
    return system, temp


def find_seed_dirs(base_dir, max_seeds=10):
    """Find seed_N subdirectories under base_dir."""
    seeds = []
    for n in range(1, max_seeds + 1):
        d = os.path.join(base_dir, f'seed_{n}')
        if os.path.isdir(d):
            seeds.append((n, d))
    return seeds


# Single run analysis
def main(directory):
    system, temp = guess_system(directory)

    # Check if this is a seed dir
    info = read_seed_info(directory)
    seed_str = f"  Seed: {info['seed']}" if 'seed' in info else ''

    print(f"\n{'='*60}")
    print(f"ANALYSIS: {directory}")
    print(f"System: {system}  |  Temperature: {temp} K{seed_str}")
    print(f"{'='*60}")

    # MD log
    log_path = os.path.join(directory, 'MD.log')
    if os.path.exists(log_path):
        stats = analyse_md_log(log_path)
        if stats:
            print(f"\nMD LOG ({len(stats)} rounds):")
            print(f"  {'Rnd':>4} {'t(ps)':>7} {'Epot(eV)':>10} {'T_avg':>7} {'T_std':>6}")
            for s in stats:
                flag = ' *** LOW T' if s['T_avg'] < 100 else ''
                print(f"  {s['round']:>4} {s['t_end']:>7.2f} "
                      f"{s['epot_end']:>10.3f} {s['T_avg']:>7.0f} "
                      f"{s['T_std']:>6.0f}{flag}")
            epots   = [s['epot_end'] for s in stats]
            crashes = [s['round'] for i,s in enumerate(stats[1:],1)
                       if s['epot_end'] > stats[i-1]['epot_end'] + 20]
            print(f"\n  Epot drop: {epots[-1]-epots[0]:.3f} eV")
            print(f"  Crashes:   {crashes if crashes else 'None'}")
    else:
        print("\nMD LOG: not found")

    # Structures
    structs = analyse_structures(directory)
    if structs:
        print(f"\nSTRUCTURE PROGRESSION ({len(structs)} rounds):")
        print(f"  {'Rnd':>4} {'Formula':>18} {'TotalO':>7} {'BoundO':>7} "
              f"{'Nlost':>6} {'Healthy':>8} {'Bonds(Å)':>12}")
        for r in structs:
            bond_str = (f"{r['bond_min']:.3f}-{r['bond_max']:.3f}"
                        if r['bond_min'] else 'none')
            h = 'OK' if r['healthy'] else 'FAIL'
            print(f"  {r['round']:>4} {r['formula']:>18} {r['n_O']:>7} "
                  f"{r['bound_O']:>7} {r['N_lost']:>6} {h:>8} {bond_str:>12}")

        bound_hist = [r['bound_O'] for r in structs]
        print(f"\n  Bound O history: {bound_hist}")
        print(f"  Best (healthy rounds): {best_bound_o(structs)}")
    else:
        print("\nSTRUCTURES: no _MD.traj files found")

    # Final relaxed
    final = check_final_relaxed(directory)
    relax = check_relax_log(directory)
    if final:
        print(f"\nFINAL RELAXED: {final['formula']}  "
              f"bound O: {final['bound_O']}  N lost: {final['N_lost']}")
    if relax:
        status = 'CONVERGED' if relax['converged'] else 'NOT CONVERGED'
        print(f"  Relaxation: {status}, fmax={relax['fmax']:.4f} eV/Å")

    # Ji comparison
    ji_val   = JI_REFERENCE.get(system, {}).get(temp)
    your_val = best_bound_o(structs) if structs else None
    print(f"\nCOMPARISON TO JI:")
    if ji_val and your_val is not None:
        diff  = your_val - ji_val
        pct   = abs(diff) / ji_val * 100
        match = 'GOOD' if pct <= 25 else 'MODERATE' if pct <= 50 else 'POOR'
        print(f"  Ji: {ji_val}  |  PFP: {your_val}  |  "
              f"Diff: {diff:+d} ({pct:.0f}%)  -  {match}")
    elif ji_val:
        print(f"  Ji: {ji_val}  |  PFP: could not determine")
    else:
        print(f"  No Ji reference for {system} {temp}K")

    print(f"\n{'='*60}\n")


# Seed analysis
def analyse_seeds(base_dir):
    """Analyse all seed runs under base_dir and print mean +/- std."""
    system, temp = guess_system(base_dir)
    seed_dirs    = find_seed_dirs(base_dir)

    if not seed_dirs:
        print(f"No seed directories found under {base_dir}")
        print("Expected: seed_1, seed_2, ... subfolders")
        return

    print(f"\n{'='*65}")
    print(f"SEED ANALYSIS: {base_dir}")
    print(f"System: {system}  |  Temperature: {temp} K")
    print(f"Seeds found: {[n for n,_ in seed_dirs]}")
    print(f"{'='*65}")

    all_best   = []
    all_final  = []
    seed_data  = {}  # seed -> bound O history

    for n, d in seed_dirs:
        structs = analyse_structures(d)
        info    = read_seed_info(d)
        final   = check_final_relaxed(d)

        if structs:
            b = best_bound_o(structs)
            history = [r['bound_O'] for r in structs]
            crashed = not structs[-1]['healthy']
            all_best.append(b)
            seed_data[n] = history
            final_b = final['bound_O'] if final else history[-1]
            all_final.append(final_b)

            crash_str = ' [CRASHED]' if crashed else ''
            print(f"\n  Seed {n}{crash_str}")
            print(f"    Rounds completed: {len(structs)}")
            print(f"    Bound O history:  {history}")
            print(f"    Best bound O:     {b}")
            print(f"    Final bound O:    {final_b}")
        else:
            print(f"\n  Seed {n}: no data")

    if all_best:
        print(f"\n{'-'*65}")
        print(f"SUMMARY ACROSS {len(all_best)} SEEDS:")
        print(f"  Best bound O per seed:  {all_best}")
        print(f"  Mean +/- std (best):      {np.mean(all_best):.1f} +/- {np.std(all_best):.1f}")
        print(f"  Range:                  {min(all_best)} - {max(all_best)}")
        print(f"  Final bound O per seed: {all_final}")
        print(f"  Mean +/- std (final):     {np.mean(all_final):.1f} +/- {np.std(all_final):.1f}")

        ji_val = JI_REFERENCE.get(system, {}).get(temp)
        if ji_val:
            mean_b = np.mean(all_best)
            diff   = mean_b - ji_val
            pct    = abs(diff) / ji_val * 100
            match  = 'GOOD' if pct <= 25 else 'MODERATE' if pct <= 50 else 'POOR'
            print(f"\n  Ji DFT reference:       {ji_val} bound O")
            print(f"  PFP mean vs Ji:         {diff:+.1f} ({pct:.0f}%)  -  {match}")

        # Print aligned bound O table across seeds
        if seed_data:
            max_rounds = max(len(v) for v in seed_data.values())
            print(f"\n  Per-round bound O table (rows=rounds, cols=seeds):")
            header = f"  {'Rnd':>4} " + " ".join(f"{'S'+str(n):>6}" for n in seed_data) + \
                     f"  {'Mean':>6} {'Std':>5}"
            print(header)
            for rnd in range(max_rounds):
                vals = []
                row  = f"  {rnd+1:>4} "
                for n in seed_data:
                    hist = seed_data[n]
                    if rnd < len(hist):
                        row += f"{hist[rnd]:>6}"
                        vals.append(hist[rnd])
                    else:
                        row += f"{'-':>6}"
                if vals:
                    row += f"  {np.mean(vals):>6.1f} {np.std(vals):>5.1f}"
                print(row)

    print(f"\n{'='*65}\n")


# Full summary table
def summary(base_dir, use_seeds=False):
    systems = [
        ('ZrN_100', '295'), ('ZrN_100', '363'), ('ZrN_100', '1023'),
        ('ZrN_110', '295'), ('ZrN_110', '363'), ('ZrN_110', '1023'),
        ('VN_111',  '295'), ('VN_111',  '363'), ('VN_111',  '873'),
    ]

    mode = "SEED-AVERAGED" if use_seeds else "SINGLE RUN"
    print(f"\n{'='*75}")
    print(f"FULL SUMMARY ({mode}) - PFP vs Ji et al. J. Phys. Chem. C 2025")
    print(f"{'='*75}")

    if use_seeds:
        print(f"{'System':>10} {'Temp':>5} {'Ji':>5} {'Mean':>7} {'Std':>5} "
              f"{'Diff':>7} {'Match':>10} {'Seeds'}")
    else:
        print(f"{'System':>10} {'Temp':>5} {'Ji':>5} {'PFP':>5} "
              f"{'Diff':>6} {'Match':>10} {'Status'}")
    print(f"{'-'*75}")

    for sys_name, temp in systems:
        temp_dir  = os.path.join(base_dir, sys_name, f'{temp}K')
        ji_val    = JI_REFERENCE.get(sys_name, {}).get(temp)

        if not os.path.exists(temp_dir):
            row = f"{sys_name:>10} {temp:>5}K {str(ji_val or '-'):>5}"
            print(row + "  not found")
            continue

        if use_seeds:
            seed_dirs = find_seed_dirs(temp_dir)
            if seed_dirs:
                all_best = []
                for _, d in seed_dirs:
                    structs = analyse_structures(d)
                    if structs:
                        b = best_bound_o(structs)
                        if b is not None:
                            all_best.append(b)

                if all_best:
                    mean_b = np.mean(all_best)
                    std_b  = np.std(all_best)
                    if ji_val:
                        diff  = mean_b - ji_val
                        pct   = abs(diff) / ji_val * 100
                        match = 'GOOD' if pct<=25 else 'MODERATE' if pct<=50 else 'POOR'
                        print(f"{sys_name:>10} {temp:>5}K {ji_val:>5} "
                              f"{mean_b:>7.1f} {std_b:>5.1f} {diff:>+7.1f} "
                              f"{match:>10}  {len(all_best)} seeds")
                    else:
                        print(f"{sys_name:>10} {temp:>5}K {'-':>5} "
                              f"{mean_b:>7.1f} {std_b:>5.1f} {'-':>7} "
                              f"{'-':>10}  {len(all_best)} seeds")
                else:
                    print(f"{sys_name:>10} {temp:>5}K {str(ji_val or '-'):>5}"
                          f"  no data in seed dirs")
            else:
                # Fall back to non-seed run
                structs  = analyse_structures(temp_dir)
                your_val = best_bound_o(structs) if structs else None
                status   = 'no seeds found - using base dir'
                if ji_val and your_val is not None:
                    diff  = your_val - ji_val
                    pct   = abs(diff) / ji_val * 100
                    match = 'GOOD' if pct<=25 else 'MODERATE' if pct<=50 else 'POOR'
                    print(f"{sys_name:>10} {temp:>5}K {ji_val:>5} "
                          f"{your_val:>7} {'-':>5} {diff:>+7} {match:>10}  {status}")
                else:
                    print(f"{sys_name:>10} {temp:>5}K {str(ji_val or '-'):>5}"
                          f"  {status}")
        else:
            structs  = analyse_structures(temp_dir)
            your_val = best_bound_o(structs) if structs else None
            if structs:
                n_healthy = sum(1 for s in structs if s['healthy'])
                status = f'round {max(range(len(structs)), key=lambda i: structs[i]["bound_O"] if structs[i]["healthy"] else -1)+1} of {len(structs)}'
            else:
                status = 'no data'

            if ji_val and your_val is not None:
                diff  = your_val - ji_val
                pct   = abs(diff) / ji_val * 100
                match = 'GOOD' if pct<=25 else 'MODERATE' if pct<=50 else 'POOR'
                print(f"{sys_name:>10} {temp:>5}K {ji_val:>5} {your_val:>5} "
                      f"{diff:>+6} {match:>10}  {status}")
            elif your_val is not None:
                print(f"{sys_name:>10} {temp:>5}K {'-':>5} {your_val:>5} "
                      f"{'-':>6} {'-':>10}  {status}")
            else:
                print(f"{sys_name:>10} {temp:>5}K {str(ji_val or '-'):>5} "
                      f"{'-':>5} {'-':>6} {'-':>10}  {status}")

    print(f"{'='*75}\n")


# Entry point
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    arg = sys.argv[1]

    if arg == '--summary':
        base = sys.argv[2] if len(sys.argv) > 2 else '/home/jovyan/Matlantis'
        summary(base, use_seeds=False)
    elif arg == '--summary-seeds':
        base = sys.argv[2] if len(sys.argv) > 2 else '/home/jovyan/Matlantis'
        summary(base, use_seeds=True)
    elif arg == '--seeds':
        if len(sys.argv) < 3:
            print("Usage: python analyse_run.py --seeds /path/to/system/temp")
            sys.exit(1)
        analyse_seeds(sys.argv[2])
    else:
        main(sys.argv[1])
