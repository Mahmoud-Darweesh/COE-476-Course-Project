import os

RESULTS_DIR = "results"
OUTPUT_FILE = "all_experiment_reports.txt"

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for root, dirs, files in os.walk(RESULTS_DIR):
        # Look for the right report files in each subdirectory
        for fname in files:
            if fname in ("report.txt", "child_report.txt", "parent_report.txt"):
                fpath = os.path.join(root, fname)
                expt_label = os.path.relpath(fpath, RESULTS_DIR).replace("\\", "/")
                out.write("="*80 + "\n")
                out.write(f"EXPERIMENT: {expt_label}\n")
                out.write("="*80 + "\n\n")
                with open(fpath, "r", encoding="utf-8") as f:
                    out.write(f.read())
                    if not f.read().endswith("\n"):
                        out.write("\n")
                out.write("\n\n")

print(f"Merged all reports into {OUTPUT_FILE}")