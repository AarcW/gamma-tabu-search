import gamma_tabu as gamma
import numpy as np
import pickle
import os

MAC_AREA_INT8 = 282
BUF_AREA_perbit = 0.086

def train_model_tabu(model_defs, input_arg, map_cstr=None, chkpt_file='./chkpt'):
    """Train using Tabu Search instead of Genetic Algorithm."""
    opt = input_arg
    fitness = [opt.fitness1, opt.fitness2]
    dimension = model_defs[0]

    env = gamma.GAMMA(dimension=dimension, num_pe=opt.num_pe, fitness=fitness, l1_size=opt.l1_size, l2_size=opt.l2_size)
    
    constraints = {"area": opt.area_budget * 1e6}

    for dimension in model_defs:
        env.reset_dimension(fitness=fitness, constraints=constraints, dimension=dimension)
        env.reset_hw_parm(num_pe=opt.num_pe, l1_size=opt.l1_size, l2_size=opt.l2_size)

        chkpt = env.run_tabu_search(dimension, num_iterations=opt.epochs)

        best_sol = chkpt["best_sol"]
        best_runtime, best_area = env.evaluate_solution(best_sol), env.compute_area_external(opt.num_pe, opt.l1_size, opt.l2_size)
        # best_runtime, best_throughput, best_energy, best_area, best_l1_size, best_l2_size, best_mac, best_power, best_num_pe = env.get_indiv_info(best_sol, num_pe=None)

        print(f"Mapping: {best_sol}")
        print(f"Reward: {chkpt['best_reward']:.3e}, Runtime: {best_runtime}(cycles), Area: {best_area/1e6:.3f}(mm²)")
        # print(f"Reward: {chkpt['best_reward'][0]:.3e}, Runtime: {best_runtime:.0f}(cycles), Area: {best_area/1e6:.3f}(mm2), PE Area_ratio: {best_num_pe*MAC_AREA_INT8/best_area*100:.1f}%, Num_PE: {best_num_pe:.0f}, L1 Buffer: {best_l1_size:.0f}(elements), L2 Buffer: {best_l2_size:.0f}(elements)")

        
        os.makedirs(os.path.dirname(chkpt_file), exist_ok=True)

        with open(chkpt_file, "wb") as fd:
            pickle.dump(chkpt, fd)
