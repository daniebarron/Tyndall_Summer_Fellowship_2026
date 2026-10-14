import os
import re

# Maximum allowed covalent bond length threshold (in Angstroms)
DIST_THRESHOLD = 2.1 

# Elements that should NOT form covalent bonds with Fluorine in standard organic/PFAS molecules
FORBIDDEN_F_BONDS = {'F', 'O', 'H'} 

def analyze_orca_output(filepath):
    suspicious_bonds = []
    in_redundant_block = False

    with open(filepath, 'r', errors='ignore') as f:
        for line in f:
            # Detect start of redundant internal coordinate block
            if "Redundant Internal Coordinates" in line:
                in_redundant_block = True
                continue
            
            # Detect end of the redundant coordinate definition block
            if in_redundant_block and ("DEFINED QUANTITIES" in line or "Basis Set Information" in line):
                break

            if in_redundant_block:
                # Matches patterns like: B(C 43, C 33) :   1.5400
                match = re.search(r"B\(([\w]+)\s+(\d+),\s*([\w]+)\s+(\d+)\)\s*:\s*([\d\.]+)", line)
                if match:
                    elem1, id1, elem2, id2, dist_str = match.groups()
                    dist = float(dist_str)

                    # Check 1: Bonds between Fluorine and non-carbon/non-metal atoms (e.g., F-F, F-O)
                    if (elem1 == 'F' and elem2 in FORBIDDEN_F_BONDS) or (elem2 == 'F' and elem1 in FORBIDDEN_F_BONDS):
                        suspicious_bonds.append(f"  [Forbidden Bond] {elem1}{id1}-{elem2}{id2} (Dist: {dist:.2f} Å)")

                    # Check 2: Covalent bonds initialized with absurdly large distances (>2.1 Å)
                    elif dist > DIST_THRESHOLD:
                        suspicious_bonds.append(f"  [Unusual Length] {elem1}{id1}-{elem2}{id2} (Dist: {dist:.2f} Å)")

    return suspicious_bonds

def main():
    print("=" * 60)
    print("Scanning ORCA Output Files for Topology / Ring Errors...")
    print("=" * 60)

    out_files = [f for f in os.listdir('.') if f.endswith('.out') or f.endswith('.log')]
    
    if not out_files:
        print("No .out or .log files found in the current directory.")
        return

    flagged_count = 0
    for file in sorted(out_files):
        issues = analyze_orca_output(file)
        if issues:
            flagged_count += 1
            print(f"\n⚠️  SUSPICIOUS FILE: {file}")
            for issue in issues[:10]: # Print first 10 bad bonds per file
                print(issue)
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more suspicious interactions.")
    
    print("\n" + "=" * 60)
    if flagged_count == 0:
        print("✅ Scan complete: All analyzed .out files look topologically sound!")
    else:
        print(f"❌ Scan complete: Flagged {flagged_count} / {len(out_files)} files with potential topological issues.")
    print("=" * 60)

if __name__ == "__main__":
    main()