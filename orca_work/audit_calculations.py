import os

CALCULATIONS_DIR = "pfas_calculations"

def audit_orca_runs():
    if not os.path.isdir(CALCULATIONS_DIR):
        print(f"ERROR: Cannot find '{CALCULATIONS_DIR}' folder.")
        return

    print("Scanning ORCA calculation folders...\n")
    
    successful = []
    failed_opt = []
    failed_freq = []
    incomplete = []
    no_output = []

    # Get all subfolders sorted alphabetically
    folders = sorted(os.listdir(CALCULATIONS_DIR))

    for folder_name in folders:
        folder_path = os.path.join(CALCULATIONS_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Look for .out files
        out_files = [f for f in os.listdir(folder_path) if f.endswith('.out')]
        
        if not out_files:
            no_output.append(folder_name)
            continue

        # Pick the largest/most recent .out file
        out_file = max(out_files, key=lambda f: os.path.getsize(os.path.join(folder_path, f)))
        out_path = os.path.join(folder_path, out_file)

        try:
            with open(out_path, 'rb') as f:
                raw = f.read()
            text = raw.decode('latin-1', errors='ignore')
        except Exception as e:
            print(f" Could not read {out_path}: {e}")
            continue

        # Check ORCA markers
        has_opt_success = 'HURRAY' in text
        has_freq_success = 'ORCA TERMINATED NORMALLY' in text
        is_aborted = 'ORCA TERMINATED ABNORMALLY' in text or 'ERROR' in text

        if has_opt_success and has_freq_success:
            successful.append(folder_name)
        elif not list(filter(None, [has_opt_success, has_freq_success, is_aborted])):
            # If none of these tags are present, the batch script likely crashed midway
            incomplete.append(folder_name)
        elif not has_opt_success:
            failed_opt.append(folder_name)
        elif has_opt_success and not has_freq_success:
            failed_freq.append(folder_name)

    # PRINT SUMMARY REPORT
    print(f"=== ORCA RUN AUDIT REPORT ===")
    print(f"Total Folders Scanned: {len(folders)}")
    print(f" Success (Opt + Freq done): {len(successful)}")
    print(f"FAIL Failed during Geometry Opt: {len(failed_opt)}")
    print(f"FAIL Failed during Freq Only:     {len(failed_freq)}")
    print(f"⏳ Incomplete / Cut Off mid-run: {len(incomplete)}")
    print(f" No .out files found:         {len(no_output)}")
    print("=============================\n")

    if failed_opt:
        print("FAIL FAILED AT GEOMETRY OPTIMISATION (Needs full rerun):")
        for f in failed_opt:
            print(f"  - {f}")
        print()

    if failed_freq:
        print("FAIL FAILED AT FREQUENCY (Can use your restart script):")
        for f in failed_freq:
            print(f"  - {f}")
        print()

    if incomplete:
        print("⏳ INCOMPLETE RUNS (Likely timed out, hit a power outage, or computer went to sleep):")
        for f in incomplete:
            print(f"  - {f}")
        print()

    if no_output:
        print(" NEVER STARTED (Input file exists, but batch script didn't reach it):")
        for f in no_output:
            print(f"  - {f}")

if __name__ == "__main__":
    audit_orca_runs()