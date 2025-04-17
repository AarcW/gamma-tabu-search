
import copy
import argparse
from datetime import datetime

import glob
import os, sys
script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '../'))
project_path = os.path.abspath(os.path.join(script_dir, '../../'))
if module_path not in sys.path:
    sys.path.insert(0,module_path)
if project_path not in sys.path:
    sys.path.insert(0,project_path)
from utils import *
import d_g as gamma
from math import ceil
import importlib
from shutil import copyfile
from multiprocessing import Pool, cpu_count
fitness_list = None
fitness = None
stage_idx = 0
prev_stage_value = []
tune_iter = 1
opt = None
MAC_AREA_MAESTRO=4470
MAC_AREA_INT8 = 282
BUF_AREA_perbit = 0.086
L2BUF_AREA_MAESTRO = 4161.536
L1BUF_AREA_MAESTRO = 4505.1889
L2BUF_UNIT = 32768
L1BUF_UNIT = 64

print("Pool imported:", Pool)  # Place after the import lines


# bias = {"par": {1: "K", 2:"C"}, "order":{1:["K", "C"]}, "tiles": {1:{"K":0.1, "C":0.2}, 2:{"K":0.3}}}
bias = {"par": {1: "K", 2:"C"}, "order":{1:["K", "C","Y", "X"], 2:["K", "C","Y", "X"]}}
# bias = {"par": {1: "K", 2:"C"}}
# bias = {"par": {1: "Y"}}


def get_pe_usage(env, sol, num_pe ):
    util_num_pe = num_pe
    baseline = env.get_indiv_info( sol, num_pe=num_pe)
    best_runtime, best_throughput, best_energy, best_area, best_l1_size, best_l2_size, best_mac, best_power, best_num_pe = baseline
    baseline = np.array(baseline)[:-2]
    for i in range(num_pe-1):
        util_num_pe -= 1
        cur = env.get_indiv_info(sol, num_pe=util_num_pe)
        best_runtime, best_throughput, best_energy, best_area, best_l1_size, best_l2_size, best_mac, best_power, best_num_pe = cur
        cur = np.array(cur)[:-2]
        if sum(baseline!=cur)>1:
            util_num_pe += 1
            break
    return util_num_pe

# Replace the train_model function in train.py with this version

def train_model(model_defs, input_arg, map_cstr=None, chkpt_file='./chkpt'):
    global opt
    opt = input_arg
    fitness = [opt.fitness1, opt.fitness2]
    
    for dimension in model_defs:
        env = gamma.GAMMA(
            dimension=dimension, 
            num_pe=opt.num_pe, 
            fitness=fitness, 
            par_RS=opt.parRS,
            l1_size=opt.l1_size,
            l2_size=opt.l2_size, 
            NocBW=opt.NocBW, 
            offchipBW=opt.offchipBW, 
            slevel_min=opt.slevel_min,
            slevel_max=opt.slevel_max,
            fixedCluster=opt.fixedCluster, 
            log_level=opt.log_level, 
            map_cstr=map_cstr
        )
        # Initialize best solution tracking
    best_overall_reward = float('-inf')
    best_overall_solution = None
    best_runtime = best_energy = best_area = best_l1 = best_l2 = best_pe = best_pe_ratio = None
    print(f"num_pe: {opt.num_pe}, l1_size: {opt.l1_size}, l2_size: {opt.l2_size}")
    for i in range(100):
        try:
            # Generate initial solution
            initial_solution = env.create_genome_fixedSL()
            env.comform_to_cstr([initial_solution])

            # Check validity before running Tabu Search
            if not env.validTo_external_mem_cstr(initial_solution, num_pe=opt.num_pe):
                print(f"Attempt {i+1}: Invalid initial solution, skipping...")
                continue
            runtime, _, energy, area, l1_size, l2_size, _, _, num_pe = env.get_indiv_info(initial_solution)

            print(f"Generated Valid Initial Solution: {initial_solution}")
            print(f"Iteration {i} Initial solution \nnum_pe: {num_pe}, l1_size: {l1_size}, l2_size: {l2_size}")

            # Run Tabu Search
            pool = Pool(cpu_count())
            current_solution, current_reward = env.tabu_search(
                pool=pool,
                initial_solution=initial_solution,
                max_iterations=opt.epochs,
                tabu_tenure=15
            )
            pool.close()

            # Get metrics for current solution
            runtime, _, energy, area, l1_size, l2_size, _, _, num_pe = env.get_indiv_info(current_solution)
            print(f"Iteration {i} num_pe: {num_pe}, l1_size: {l1_size}, l2_size: {l2_size}")
            pe_area = num_pe * MAC_AREA_INT8
            pe_area_ratio = (pe_area / area) * 100
            # Add POST-SEARCH VALIDATION
            if not env.validTo_external_mem_cstr(current_solution):
                print("Tabu Search produced invalid solution!")
                continue  # Skip to next iteration
            
            
            # Add theoretical minimum cycle check
            min_cycles = (dimension[0]*dimension[1]*dimension[4]*dimension[5])/opt.num_pe
            if runtime < min_cycles:
                print(f"Impossible runtime: {runtime} < {min_cycles}")
                continue
            print(f"Valid cycles")
            # Update best solution if improved
            if current_reward is None:
                print("No reward returned from Tabu Search")
                continue
            if current_reward[0] > best_overall_reward:
                best_overall_reward = current_reward[0]
                best_overall_solution = current_solution
                best_runtime, best_energy, best_area = runtime, energy, area
                best_l1, best_l2, best_pe = l1_size, l2_size, num_pe
                best_pe_ratio = pe_area_ratio

                print(f"\n🔥 New best at attempt {i+1}:")
                print(f"Reward: {best_overall_reward:.3e}")

        except Exception as e:
            print(f"Attempt {i+1} failed with error: {str(e)}")
            continue

    # After all iterations, output the best solution
    if best_overall_solution is not None:
        print("\n=== FINAL BEST SOLUTION ===")
        print("Mapping:", best_overall_solution)
        print(
            f"Reward: {best_overall_reward:.3e}, "
            f"Runtime: {best_runtime:.0f}(cycles), "
            f"Area: {best_area/1e6:.3f}(mm2), "
            f"Energy: {best_energy:.3e} nJ, "
            f"PE Area_ratio: {best_pe_ratio:.1f}%, "
            f"Num_PE: {best_pe:.0f}, "
            f"L1 Buffer: {best_l1:.0f}(elements), "
            f"L2 Buffer: {best_l2:.0f}(elements)"
        )
        
        # Save checkpoint
        chkpt = {
            "reward": best_overall_reward,
            "best_sol": best_overall_solution,
            "runtime": best_runtime,
            "area": best_area,
            "PE": best_pe,
            "L1_size": best_l1,
            "L2_size": best_l2
        }
        df = pd.DataFrame([chkpt])
        df.to_csv(chkpt_file[:-4] + ".csv")
        with open(chkpt_file, "wb") as fd:
            pickle.dump(chkpt, fd)
    else:
        print("\n❌ No valid solutions found in 100 attempts")
        exit(1)

def get_cstr_name(mapping_cstr):
    if mapping_cstr:
        cstr_name = mapping_cstr
    else:
        cstr_name = "free"
    return cstr_name

# def train_model

