"""
view_local.py — View Matlantis trajectory files locally using ASE GUI

Usage:
  python view_local.py /path/to/file.traj              # view single file
  python view_local.py /path/to/directory              # view all rounds as slideshow
  python view_local.py /path/to/directory --round 5    # view specific round
  python view_local.py /path/to/directory --final      # view final_relaxed.traj
  python view_local.py /path/to/directory --movie      # concatenated trajectory movie
  python view_local.py /path/to/directory --seeds      # compare best structures across seeds
  python view_local.py /path/to/directory --plot       # oxidation progression plot only

Requires: ase, matplotlib
  pip install ase matplotlib
"""

import sys
import os
import argparse
import numpy as np

try:
    from ase import io
    from ase.visualize import view
except ImportError:
    print("ERROR: ASE not installed. Run: pip install ase")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not installed — plots unavailable")


JI_REFERENCE = {
    'ZrN_100': {'295': 8,  '363': 12, '1023': 28},
    'ZrN_110': {'295': 18, '363': 20, '1023': 32},
    'VN_111':  {'295': 11, '363': 18, '873':  12},
}


def guess_system(path):
    parts  = path.replace('\\', '/').split('/')
    system = next((p for p in parts if any(x in p for x in
                   ('ZrN_100','ZrN_110','VN_111'))), 'unknown')
    temp   = next((p[:-1] for p in parts if p.endswith('K')
                   and p[:-1].isdigit()), 'unknown')
    return system, temp


def count_bound_o(atoms, cutoff=3.5):
    syms = atoms.get_chemical_symbols()
    pos  = atoms.get_positions()
    o_idx     = [i for i,s in enumerate(syms) if s=='O']
    metal_idx = [i for i,s in enumerate(syms) if s in ('Zr','V')]
    if not o_idx or not metal_idx:
        return 0
    return sum(1 for oi in o_idx
               if min(np.linalg.norm(pos[oi]-pos[mj]) for mj in metal_idx) < cutoff)


def is_healthy(atoms, min_atoms=55):
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


def load_rounds(directory):
    """Load all end-of-round _MD.traj files from a directory."""
    frames, labels = [], []
    i = 0
    while True:
        path = os.path.join(directory, f'{i:0>2d}_MD.traj')
        if not os.path.exists(path):
            break
        try:
            atoms  = io.read(path, index=-1)
            bound  = count_bound_o(atoms)
            health = '✓' if is_healthy(atoms) else '✗'
            labels.append(f'Round {i} | {atoms.get_chemical_formula()} | bound O: {bound} {health}')
            frames.append(atoms)
        except Exception as e:
            print(f"  Warning: round {i} unreadable — {e}")
        i += 1
    return frames, labels


def load_movie(directory):
    """Concatenate all MD trajectory files into one movie."""
    all_frames = []
    i = 0
    while True:
        path = os.path.join(directory, f'{i:0>2d}_MD-trajectory.traj')
        if not os.path.exists(path):
            break
        try:
            frames = io.read(path, index=':')
            all_frames.extend(frames)
            print(f"  Round {i:02d}: {len(frames)} frames")
        except Exception as e:
            print(f"  Round {i:02d}: error — {e}")
        i += 1
    return all_frames


def find_seed_dirs(base_dir, max_seeds=10):
    seeds = []
    for n in range(1, max_seeds + 1):
        d = os.path.join(base_dir, f'seed_{n}')
        if os.path.isdir(d):
            seeds.append((n, d))
    return seeds


def plot_oxidation(directory, save=False):
    """Plot bound O and total O progression, with Ji reference line."""
    if not HAS_MPL:
        print("matplotlib required for plotting")
        return

    system, temp = guess_system(directory)
    ji_val = JI_REFERENCE.get(system, {}).get(temp)

    rounds_list, bound_list, total_list = [], [], []
    i = 0
    while True:
        path = os.path.join(directory, f'{i:0>2d}_MD.traj')
        if not os.path.exists(path):
            break
        try:
            atoms = io.read(path, index=-1)
            if is_healthy(atoms):
                rounds_list.append(i + 1)
                bound_list.append(count_bound_o(atoms))
                total_list.append(atoms.get_chemical_symbols().count('O'))
        except:
            pass
        i += 1

    if not rounds_list:
        print("No healthy rounds found to plot")
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rounds_list, bound_list, 'o-', color='steelblue',
            label='PFP bound O', linewidth=2)
    ax.plot(rounds_list, total_list, 's--', color='lightblue',
            alpha=0.6, label='Total O in slab')
    if ji_val:
        ax.axhline(ji_val, color='red', linestyle='--', linewidth=1.5,
                   label=f'Ji DFT ({ji_val} bound O)')

    ax.set_xlabel('Round')
    ax.set_ylabel('O atoms')
    ax.set_title(f'{system} {temp}K — Oxidation progression')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        out = os.path.join(directory, f'{system}_{temp}K_oxidation.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved to {out}")
    plt.show()


def plot_seeds(base_dir, save=False):
    """Multi-seed overlay plot with mean ± std band."""
    if not HAS_MPL:
        print("matplotlib required for plotting")
        return

    system, temp = guess_system(base_dir)
    ji_val       = JI_REFERENCE.get(system, {}).get(temp)
    seed_dirs    = find_seed_dirs(base_dir)

    if not seed_dirs:
        print(f"No seed_N subdirectories found under {base_dir}")
        return

    all_bound = {}
    for n, d in seed_dirs:
        hist = []
        i = 0
        while True:
            path = os.path.join(d, f'{i:0>2d}_MD.traj')
            if not os.path.exists(path):
                break
            try:
                atoms = io.read(path, index=-1)
                if is_healthy(atoms):
                    hist.append(count_bound_o(atoms))
                else:
                    break
            except:
                break
            i += 1
        if hist:
            all_bound[n] = hist
            print(f"Seed {n}: {len(hist)} rounds, final={hist[-1]}")

    if not all_bound:
        print("No data found")
        return

    max_len = max(len(v) for v in all_bound.values())
    padded  = np.full((len(all_bound), max_len), np.nan)
    for i, b in enumerate(all_bound.values()):
        padded[i, :len(b)] = b

    mean_b = np.nanmean(padded, axis=0)
    std_b  = np.nanstd(padded, axis=0)
    rounds = np.arange(1, max_len + 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = plt.cm.Blues(np.linspace(0.35, 0.75, len(all_bound)))
    for (n, b), col in zip(all_bound.items(), colors):
        ax.plot(range(1, len(b)+1), b, color=col, alpha=0.4,
                linewidth=1, label=f'Seed {n}')

    ax.plot(rounds, mean_b, 'o-', color='steelblue',
            linewidth=2.5, markersize=5, label='Mean', zorder=5)
    ax.fill_between(rounds, mean_b - std_b, mean_b + std_b,
                    alpha=0.25, color='steelblue', label='±1 std')

    if ji_val:
        ax.axhline(ji_val, color='red', linestyle='--', linewidth=1.5,
                   label=f'Ji DFT ({ji_val} bound O)')

    final_vals = [b[-1] for b in all_bound.values()]
    stats_txt  = (f'Final bound O:\n'
                  f'  Mean: {np.mean(final_vals):.1f}\n'
                  f'  Std:  {np.std(final_vals):.1f}\n'
                  f'  Range: {min(final_vals)}–{max(final_vals)}')
    if ji_val:
        stats_txt += f'\n  Ji DFT: {ji_val}'
    ax.text(0.02, 0.97, stats_txt, transform=ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.set_xlabel('Round', fontsize=12)
    ax.set_ylabel('Bound O atoms', fontsize=12)
    ax.set_title(f'{system} {temp}K — {len(all_bound)} seeds', fontsize=13)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        out = os.path.join(base_dir, f'{system}_{temp}K_seeds.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved to {out}")
    plt.show()


def compare_seeds_structures(base_dir):
    """View best structure from each seed side by side as slideshow."""
    seed_dirs = find_seed_dirs(base_dir)
    if not seed_dirs:
        print(f"No seed directories found under {base_dir}")
        return

    frames, labels = [], []
    for n, d in seed_dirs:
        best_atoms = None
        best_bound = -1
        i = 0
        while True:
            path = os.path.join(d, f'{i:0>2d}_MD.traj')
            if not os.path.exists(path):
                break
            try:
                atoms = io.read(path, index=-1)
                if is_healthy(atoms):
                    b = count_bound_o(atoms)
                    if b > best_bound:
                        best_bound = b
                        best_atoms = atoms
                        best_round = i
            except:
                pass
            i += 1

        if best_atoms is not None:
            frames.append(best_atoms)
            labels.append(f'Seed {n} | Round {best_round} | '
                          f'{best_atoms.get_chemical_formula()} | bound O: {best_bound}')
            print(f"Seed {n}: best = round {best_round}, bound O = {best_bound}")
        else:
            print(f"Seed {n}: no healthy structures found")

    if frames:
        print(f"\nOpening {len(frames)} structures in ASE GUI...")
        print("Use arrow keys to step through seeds")
        for i, label in enumerate(labels):
            print(f"  Frame {i}: {label}")
        view(frames)
    else:
        print("No structures to display")


def main():
    parser = argparse.ArgumentParser(
        description='View Matlantis trajectory files locally',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('path', help='Path to .traj file or directory')
    parser.add_argument('--round', type=int, default=None,
                        help='View specific round number')
    parser.add_argument('--final', action='store_true',
                        help='View final_relaxed.traj')
    parser.add_argument('--movie', action='store_true',
                        help='Play concatenated MD trajectory as movie')
    parser.add_argument('--seeds', action='store_true',
                        help='Compare best structures across seed runs')
    parser.add_argument('--plot', action='store_true',
                        help='Show oxidation progression plot')
    parser.add_argument('--plot-seeds', action='store_true',
                        help='Show multi-seed overlay plot')
    parser.add_argument('--save', action='store_true',
                        help='Save plots to PNG files')
    args = parser.parse_args()

    path = args.path

    # ── Single .traj file ──────────────────────────────────────────────────────
    if path.endswith('.traj') and os.path.isfile(path):
        frames = io.read(path, index=':')
        if not isinstance(frames, list):
            frames = [frames]
        bound = count_bound_o(frames[-1])
        print(f"File: {path}")
        print(f"Frames: {len(frames)}")
        print(f"Final formula: {frames[-1].get_chemical_formula()}")
        print(f"Final bound O: {bound}")
        print("Opening ASE GUI...")
        view(frames)
        return

    # ── Directory modes ────────────────────────────────────────────────────────
    if not os.path.isdir(path):
        print(f"ERROR: {path} is not a file or directory")
        sys.exit(1)

    system, temp = guess_system(path)
    print(f"System: {system}  |  Temperature: {temp}K")

    if args.plot_seeds:
        plot_seeds(path, save=args.save)
        return

    if args.seeds:
        compare_seeds_structures(path)
        return

    if args.plot:
        plot_oxidation(path, save=args.save)
        return

    if args.final:
        fpath = os.path.join(path, 'final_relaxed.traj')
        if not os.path.exists(fpath):
            print(f"No final_relaxed.traj in {path}")
            sys.exit(1)
        atoms = io.read(fpath, index=-1)
        print(f"Final relaxed: {atoms.get_chemical_formula()}, "
              f"bound O: {count_bound_o(atoms)}")
        view([atoms])
        return

    if args.movie:
        print("Loading full trajectory movie (all rounds concatenated)...")
        frames = load_movie(path)
        if not frames:
            print("No trajectory files found")
            sys.exit(1)
        print(f"Total frames: {len(frames)}")
        print("Opening ASE GUI — use slider to scrub through trajectory")
        view(frames)
        return

    if args.round is not None:
        fpath = os.path.join(path, f'{args.round:0>2d}_MD.traj')
        if not os.path.exists(fpath):
            print(f"Round {args.round} not found: {fpath}")
            sys.exit(1)
        frames = io.read(fpath, index=':')
        if not isinstance(frames, list):
            frames = [frames]
        atoms  = frames[-1]
        print(f"Round {args.round}: {atoms.get_chemical_formula()}, "
              f"bound O: {count_bound_o(atoms)}")
        view(frames)
        return

    # ── Default: slideshow of all end-of-round structures ─────────────────────
    print("Loading end-of-round snapshots...")
    frames, labels = load_rounds(path)
    if not frames:
        print("No _MD.traj files found")
        sys.exit(1)

    print(f"\nLoaded {len(frames)} rounds:")
    for i, label in enumerate(labels):
        print(f"  {label}")

    print("\nOpening ASE GUI — use arrow keys to step through rounds")
    view(frames)


if __name__ == '__main__':
    main()
