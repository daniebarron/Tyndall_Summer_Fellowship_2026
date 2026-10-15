"""
check_imaginary_freqs.py
========================
Scans a directory of ORCA 6.x frequency output files (.out) and reports
which molecules have imaginary vibrational frequencies (indicating the
geometry did NOT converge to a true minimum).

Usage
-----
    python check_imaginary_freqs.py                  # scans current directory
    python check_imaginary_freqs.py path/to/out/dir  # scans a specific folder

Output
------
    - Console summary (pass / fail table + counts)
    - check_imaginary_freqs_results.csv written next to the script

Background
----------
ORCA prints imaginary frequencies as negative numbers in the
VIBRATIONAL FREQUENCIES block, e.g.:

     0:       0.00 cm**-1   <- translational/rotational (always ~0)
     6:    -45.32 cm**-1   <- imaginary (BAD - not a true minimum)
     7:    238.61 cm**-1   <- real mode (good)

The first 6 modes (indices 0-5) are always the 6 translational/rotational
zero modes and are ignored. Any remaining negative frequency is flagged.
"""

import os
import sys
import re
import csv
from pathlib import Path

# configurable thresholds
IMAGINARY_THRESHOLD = -5.0   # cm⁻¹  (modes more negative than this are flagged;
                              # small numerical noise near 0 is ignored)
LOW_FREQ_WARN       = 20.0   # cm⁻¹  (real but very low freqs - worth a look)


def parse_frequencies(filepath: Path):
    """
    Extract vibrational frequencies from an ORCA output file.

    Returns
    -------
    freqs : list[float]
        All frequencies (cm⁻¹) listed under VIBRATIONAL FREQUENCIES,
        including the six zero modes. Empty list if block not found.
    normal_termination : bool
        True if ORCA printed "ORCA TERMINATED NORMALLY"
    """
    freq_pattern   = re.compile(r"^\s*\d+:\s+(-?\d+\.\d+)\s+cm\*\*-1")
    block_header   = re.compile(r"VIBRATIONAL FREQUENCIES")
    term_pattern   = re.compile(r"ORCA TERMINATED NORMALLY")

    freqs              = []
    in_freq_block      = False
    normal_termination = False

    # ORCA on Windows sometimes writes UTF-16-LE files - detect and handle both
    raw = filepath.read_bytes()
    if raw[:2] in (b'\xff\xfe', b'\xfe\xff') or (len(raw) > 2 and raw[1] == 0):
        text = raw.decode("utf-16-le", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")

    for line in text.splitlines():
            if term_pattern.search(line):
                normal_termination = True
            if block_header.search(line):
                in_freq_block = True
                freqs = []          # reset - use LAST occurrence (opt+freq jobs)
                continue
            if in_freq_block:
                m = freq_pattern.match(line)
                if m:
                    freqs.append(float(m.group(1)))
                elif freqs and line.strip() == "":
                    # blank line after the frequency list signals end of block
                    in_freq_block = False

    return freqs, normal_termination


def check_file(filepath: Path):
    """
    Analyse one ORCA output file.

    Returns a dict with keys:
        name, path, status, imaginary_freqs, low_real_freqs,
        n_modes, normal_termination, notes
    """
    result = {
        "name"              : filepath.stem,
        "path"              : str(filepath),
        "status"            : "UNKNOWN",
        "imaginary_freqs"   : [],
        "low_real_freqs"    : [],
        "n_modes"           : 0,
        "normal_termination": False,
        "notes"             : "",
    }

    try:
        freqs, normal_term = parse_frequencies(filepath)
    except Exception as exc:
        result["status"] = "ERROR"
        result["notes"]  = f"Could not parse file: {exc}"
        return result

    result["normal_termination"] = normal_term

    if not freqs:
        result["status"] = "ERROR"
        result["notes"]  = "No VIBRATIONAL FREQUENCIES block found"
        return result

    result["n_modes"] = len(freqs)

    # Skip the first 6 translational/rotational modes (indices 0-5)
    vibrational = freqs[6:]

    imag = [f for f in vibrational if f < IMAGINARY_THRESHOLD]
    low  = [f for f in vibrational if IMAGINARY_THRESHOLD <= f < LOW_FREQ_WARN]

    result["imaginary_freqs"] = imag
    result["low_real_freqs"]  = low

    notes = []
    if not normal_term:
        notes.append("ORCA did not terminate normally")
    if low:
        notes.append(
            f"{len(low)} low real freq(s) (<{LOW_FREQ_WARN} cm⁻¹): "
            + ", ".join(f"{f:.2f}" for f in low)
        )

    result["notes"] = "; ".join(notes) if notes else ""

    if imag:
        result["status"] = "FAIL"
    elif not normal_term:
        result["status"] = "WARN"
    else:
        result["status"] = "PASS"

    return result


def print_summary(results):
    passes = [r for r in results if r["status"] == "PASS"]
    fails  = [r for r in results if r["status"] == "FAIL"]
    warns  = [r for r in results if r["status"] == "WARN"]
    errors = [r for r in results if r["status"] == "ERROR"]

    # header
    print("\n" + "=" * 70)
    print("  ORCA IMAGINARY FREQUENCY CHECK")
    print("=" * 70)
    print(f"  Files scanned : {len(results)}")
    print(f"  PASS          : {len(passes)}")
    print(f"  FAIL (imag.)  : {len(fails)}")
    print(f"  WARN          : {len(warns)}")
    print(f"  ERROR/unread  : {len(errors)}")
    print("=" * 70)

    # failures
    if fails:
        print("\nFAIL  FAIL - imaginary frequencies detected (geometry NOT a true minimum)")
        print("-" * 70)
        for r in sorted(fails, key=lambda x: x["name"]):
            imag_str = ", ".join(f"{f:.2f}" for f in r["imaginary_freqs"])
            print(f"  {r['name']:<40}  {imag_str} cm⁻¹")
            if r["notes"]:
                print(f"      -> {r['notes']}")
    else:
        print("\nOK  No imaginary frequencies found in any file.")

    # warnings
    if warns:
        print("\nWARN  WARN - job did not terminate normally")
        print("-" * 70)
        for r in sorted(warns, key=lambda x: x["name"]):
            print(f"  {r['name']:<40}  {r['notes']}")

    # errors
    if errors:
        print("\nWARN   ERROR - could not parse file")
        print("-" * 70)
        for r in sorted(errors, key=lambda x: x["name"]):
            print(f"  {r['name']:<40}  {r['notes']}")

    # passing list
    print("\nOK  PASS - all vibrational modes are real")
    print("-" * 70)
    for r in sorted(passes, key=lambda x: x["name"]):
        low_note = (
            f"  [WARN low freq: {', '.join(f'{f:.2f}' for f in r['low_real_freqs'])} cm⁻¹]"
            if r["low_real_freqs"] else ""
        )
        print(f"  {r['name']}{low_note}")

    print("\n" + "=" * 70 + "\n")


def write_csv(results, out_path: Path):
    fieldnames = [
        "name", "status", "n_imaginary", "imaginary_freqs_cm-1",
        "n_low_real", "low_real_freqs_cm-1", "n_modes",
        "normal_termination", "notes", "path",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in sorted(results, key=lambda x: x["name"]):
            writer.writerow({
                "name"                  : r["name"],
                "status"                : r["status"],
                "n_imaginary"           : len(r["imaginary_freqs"]),
                "imaginary_freqs_cm-1"  : "; ".join(f"{f:.2f}" for f in r["imaginary_freqs"]),
                "n_low_real"            : len(r["low_real_freqs"]),
                "low_real_freqs_cm-1"   : "; ".join(f"{f:.2f}" for f in r["low_real_freqs"]),
                "n_modes"               : r["n_modes"],
                "normal_termination"    : r["normal_termination"],
                "notes"                 : r["notes"],
                "path"                  : r["path"],
            })
    print(f"  Results saved -> {out_path}\n")


def main():
    scan_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")

    if not scan_dir.is_dir():
        print(f"ERROR: '{scan_dir}' is not a directory.")
        sys.exit(1)

    out_files = sorted(scan_dir.rglob("*.out"))

    if not out_files:
        print(f"No .out files found in '{scan_dir}'.")
        sys.exit(0)

    print(f"\nScanning {len(out_files)} .out file(s) in: {scan_dir.resolve()}")

    results = [check_file(f) for f in out_files]

    print_summary(results)

    csv_path = scan_dir / "check_imaginary_freqs_results.csv"
    write_csv(results, csv_path)


if __name__ == "__main__":
    main()
