#!/usr/bin/env python3
import subprocess
import re
import pandas as pd
import time
from pathlib import Path
import ast

FULL_RESULTS_FILE = "all_model_comparison_results_full.xlsx"
COMPACT_RESULTS_FILE = "all_model_comparison_latency_energy.xlsx"
GAMMA_DIR = Path("./src/GAMMA")

model_names = [
    "BERT_m", "densenet", "dlrmRMC1_m",
    "googlenet", "mnasnet", "mobilenet_v2", "ncf_m", "resnet18",
    "resnet50", "resnet50_32x4d", "shufflenet_v2", "squeezenet",
    "T5_m", "transformer", "vgg16", "wide_resnet50"
]
# temporarily removed "ALBERT_m", "alexnet",

def run_gamma(command_args, timeout=180):
    start_time = time.time()
    try:
        result = subprocess.run(
            command_args,
            cwd=GAMMA_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout
        )
        exec_time = time.time() - start_time
        return result.stdout, exec_time
    except subprocess.TimeoutExpired:
        print(f"[Timeout] Command timed out: {' '.join(command_args)}")
        return "[Timeout]", timeout

def parse_metrics(output):
    metrics = {
        "Runtime (cycles)": r"Runtime: ([\d.e+-]+)\(cycles\)",
        "Energy (nJ)": r"Energy: ([\d.e+-]+) nJ",
        "L1 Size (elements)": r"L1 Buffer: ([\d.e+-]+)\(elements\)",
        "L2 Size (elements)": r"L2 Buffer: ([\d.e+-]+)\(elements\)",
        "Mapping": r"Mapping:\s*(\[\[.*?\]\])"
    }
    results = {}
    for metric, pattern in metrics.items():
        match = re.search(pattern, output, re.DOTALL if metric == "Mapping" else 0)
        if match:
            try:
                results[metric] = ast.literal_eval(match.group(1)) if metric == "Mapping" else float(match.group(1))
            except:
                results[metric] = match.group(1) if metric == "Mapping" else None
        else:
            results[metric] = None
    return results

full_results = []
compact_results = []

for model in model_names:
    print(f"Running for model: {model}")

    # Original Gamma
    orig_out, orig_time = run_gamma([
        "python", "main.py",
        "--fitness1", "latency", "--fitness2", "power",
        "--num_pe", "168", "--l1_size", "512", "--l2_size", "108000",
        "--NocBW", "81920000", "--epochs", "10",
        "--model", model, "--singlelayer", "1"
    ])
    if "[Timeout]" in orig_out:
        orig_metrics = {"Model": model, "Method": "Original Gamma", "Runtime (cycles)": None, "Energy (nJ)": None, "EDP (cycles·nJ)": None, "L1 Size (elements)": None, "L2 Size (elements)": None, "Mapping": None}
    else:
        orig_metrics = parse_metrics(orig_out)
        edp = orig_metrics["Runtime (cycles)"] * orig_metrics["Energy (nJ)"] if orig_metrics["Runtime (cycles)"] and orig_metrics["Energy (nJ)"] else None
        orig_metrics.update({"Model": model, "Method": "Original Gamma", "EDP (cycles·nJ)": edp})

    # Tabu Search Gamma
    tabu_out, tabu_time = run_gamma([
        "python", "d_m.py",
        "--fitness1", "latency", "--fitness2", "power",
        "--num_pe", "168", "--l1_size", "512", "--l2_size", "108000",
        "--NocBW", "81920000", "--epochs", "10",
        "--model", model, "--singlelayer", "1", "--log_level", "1"
    ])

    if "No valid solutions found" in tabu_out or "[Timeout]" in tabu_out:
        print(f"[Warning] Tabu Search Gamma failed or timed out for model: {model}")
        tabu_metrics = {"Model": model, "Method": "Tabu Search Gamma", "Runtime (cycles)": None, "Energy (nJ)": None, "EDP (cycles·nJ)": None, "L1 Size (elements)": None, "L2 Size (elements)": None, "Mapping": None}
    else:
        tabu_metrics = parse_metrics(tabu_out)
        edp = tabu_metrics["Runtime (cycles)"] * tabu_metrics["Energy (nJ)"] if tabu_metrics["Runtime (cycles)"] and tabu_metrics["Energy (nJ)"] else None
        tabu_metrics.update({"Model": model, "Method": "Tabu Search Gamma", "EDP (cycles·nJ)": edp})

    # Append to full results
    full_results.extend([orig_metrics, tabu_metrics])

    # Append to compact
    compact_results.append([model, "Original Gamma", orig_metrics["Runtime (cycles)"], orig_metrics["Energy (nJ)"]])
    compact_results.append([model, "Tabu Search Gamma", tabu_metrics["Runtime (cycles)"], tabu_metrics["Energy (nJ)"]])

# Save full dataframe in desired column order
full_df = pd.DataFrame(full_results)
full_columns = ["Model", "Method", "Runtime (cycles)", "Energy (nJ)", "EDP (cycles·nJ)", "L1 Size (elements)", "L2 Size (elements)", "Mapping"]
full_df = full_df[full_columns]
full_df.to_excel(FULL_RESULTS_FILE, index=False)

# Save compact dataframe
compact_df = pd.DataFrame(compact_results, columns=["Model", "Method", "Latency (cycles)", "Energy (nJ)"])
compact_df.to_excel(COMPACT_RESULTS_FILE, index=False)

print(f"\nSaved detailed results to {FULL_RESULTS_FILE}")
print(f"Saved compact results to {COMPACT_RESULTS_FILE}")
