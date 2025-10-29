"""Run only Bruker .d benchmark and merge with existing results."""
import csv
from pathlib import Path

import pandas as pd

from xenium_comparison import benchmark_bruker

OUTPUT_CSV = Path("results/xenium_comparison.csv")

def main():
    print("=" * 70)
    print("BRUKER .d ONLY - BENCHMARK UPDATE")
    print("=" * 70)

    # Load existing results
    existing_csv = OUTPUT_CSV
    if not existing_csv.exists():
        print(f"Error: {existing_csv} not found!")
        return

    print(f"\nLoading existing results from {existing_csv}...")
    df = pd.read_csv(existing_csv)
    print(f"Loaded {len(df)} rows")
    print(f"Formats in existing data: {df['format'].unique()}")

    # Remove old Bruker results
    df_filtered = df[df['format'] != 'Bruker .d'].copy()
    print(f"\nRemoved {len(df) - len(df_filtered)} Bruker .d rows")
    print(f"Remaining rows: {len(df_filtered)}")

    # Run new Bruker benchmark
    print("\n" + "=" * 70)
    bruker_results = benchmark_bruker()
    print("=" * 70)

    # Convert to DataFrame
    bruker_df = pd.DataFrame(bruker_results)
    print(f"\nNew Bruker results: {len(bruker_df)} rows")

    # Merge
    merged_df = pd.concat([df_filtered, bruker_df], ignore_index=True)
    print(f"Merged total: {len(merged_df)} rows")
    print(f"Formats in merged data: {sorted(merged_df['format'].unique())}")

    # Save
    merged_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved updated results to {OUTPUT_CSV}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY BY FORMAT")
    print("=" * 70)
    for fmt in sorted(merged_df['format'].unique()):
        count = len(merged_df[merged_df['format'] == fmt])
        print(f"{fmt:25s}: {count} measurements")

if __name__ == "__main__":
    main()
