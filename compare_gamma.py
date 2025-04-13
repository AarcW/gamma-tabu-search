import subprocess
import re
import pandas as pd
import time
from pathlib import Path

RESULTS_FILE = "comparison_results.xlsx"
GAMMA_DIR = Path("./src/GAMMA")

def run_gamma(command_args):
    """Run a Gamma command and return output + execution time"""
    start_time = time.time()
    result = subprocess.run(
        command_args,
        cwd=GAMMA_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    exec_time = time.time() - start_time
    return result.stdout, exec_time

def parse_metrics(output):
    """Extract metrics from Gamma output using regex"""
    metrics = {
        "Runtime (cycles)": r"Runtime: ([\d.e+-]+)\(cycles\)",
        "Energy (nJ)": r"Energy: ([\d.e+-]+) nJ",
        "Area (mm²)": r"Area: ([\d.e+-]+)\(mm2\)",
        "PE Area Ratio (%)": r"PE Area_ratio: ([\d.e+-]+)%",
        "L1 Size (elements)": r"L1 Buffer: ([\d.e+-]+)\(elements\)",
        "L2 Size (elements)": r"L2 Buffer: ([\d.e+-]+)\(elements\)",
    }
    
    results = {}
    for metric, pattern in metrics.items():
        match = re.search(pattern, output)
        if match:
            value = match.group(1)
            # Always convert to float to handle scientific notation
            results[metric] = float(value)
    return results

# Original Gamma configuration
original_stdout, original_time = run_gamma([
    "python", "main.py",
    "--fitness1", "latency",
    "--fitness2", "power",
    "--num_pe", "168",
    "--l1_size", "512",
    "--l2_size", "108000",
    "--NocBW", "81920000",
    "--epochs", "100",
    "--model", "vgg16",
    "--singlelayer", "1"
])

# Tabu Search configuration
tabu_stdout, tabu_time = run_gamma([
    "python", "d_m.py",
    "--fitness1", "latency",
    "--fitness2", "power", 
    "--l1_size", "128",
    "--l2_size", "27000",
    "--NocBW", "81920000",
    "--epochs", "100",
    "--model", "vgg16",
    "--singlelayer", "1",
    "--log_level", "1",
    "--num_pe", "168"
])

# Parse results
original_metrics = parse_metrics(original_stdout)
original_metrics["Execution Time (s)"] = original_time

tabu_metrics = parse_metrics(tabu_stdout)
tabu_metrics["Execution Time (s)"] = tabu_time

# Create comparison table
comparison_df = pd.DataFrame({
    "Original Gamma": original_metrics,
    "Tabu Search Gamma": tabu_metrics
}).transpose()

# Save to Excel
comparison_df.to_excel(RESULTS_FILE)
print(f"Results saved to {RESULTS_FILE}")