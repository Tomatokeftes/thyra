"""Helper script to install dependencies and run benchmark."""

import subprocess
import sys

# Install required packages
required_packages = [
    "numpy>=2.0.0",
    "scipy>=1.7.0",
]

print("Installing required packages...")
for package in required_packages:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")
        sys.exit(1)

print("\nPackages installed successfully!")
print("="*70)
print()

# Now run the benchmark
print("Running benchmark...")
subprocess.check_call([sys.executable, "benchmarks/sparse_matrix_construction_benchmark.py"])
