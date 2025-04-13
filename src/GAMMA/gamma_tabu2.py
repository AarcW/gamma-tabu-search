import numpy as np
import copy, random
import os
from multiprocessing.pool import Pool
from multiprocessing import cpu_count

class GAMMA(object):
    def __init__(self, dimension, num_pe=64, fitness="latency", l1_size=512, l2_size=108000, NocBW=81920000, slevel_min=2, slevel_max=2, fixedCluster=0):
        super(GAMMA, self).__init__()
        self.dimension = dimension
        self.num_pe = num_pe
        self.fitness_objective = fitness
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.NocBW = NocBW
        self.slevel_min = slevel_min
        self.slevel_max = slevel_max
        self.fixedCluster = fixedCluster
        self.best_sol = None
        self.best_reward = -float("inf")

    def get_indiv_info(self, individual, num_pe=None, l1_size=None, l2_size=None, NocBW=None):
        self.oberserve_maestro(individual,num_pe=num_pe, l1_size=l1_size, l2_size=l2_size, NocBW=NocBW)
        return self.observation
    
    def get_dimension_factors(self, dimension_dict):
        dimension_factors = dict()
        for key, value in dimension_dict.items():
            if key != "T":
                factors = self.get_factors(value)
                dimension_factors[key] = {"set":factors, "array":np.array(list(factors))}
        return dimension_factors
    
    def reset_dimension(self, dimension=None, fitness=None, constraints=None, constraint_class=None, external_mem_cstr=None):
        if dimension is not None:
            self.dimension = dimension
        if fitness is not None:
            self.fitness_objective =  fitness
        if constraints is not None:
            self.constraints = constraints
        if constraint_class is not None:
            self.constraint_class = constraint_class
        if external_mem_cstr is not None:
            self.external_mem_cstr = external_mem_cstr
        self.use_ranking = True if self.fitness_objective[0] == "ranking" else False
        self.dimension_dict = {"K": self.dimension[0], "C": self.dimension[1], "Y": self.dimension[2], "X": self.dimension[3], "R": self.dimension[4],"S": self.dimension[5], "T": self.dimension[6]}
        self.dimension_factors = self.get_dimension_factors(self.dimension_dict)

    def evaluate_solution(self, solution):
        """Evaluate a single solution using the existing evaluate function."""
        pool = None  # No multiprocessing
        reward_activ_list = self.thread_fun(solution)
        reward, _ = reward_activ_list
        if reward is None or any(np.array(reward) >= 0):
            return float("-inf")  # Invalid solution
        return reward[self.stage_idx]

    def generate_neighbors(self, solution):
        """Generate neighbors by tweaking parameters like tile sizes and PE count."""
        neighbors = []
        for _ in range(5):  # Generate 5 neighbor solutions
            new_solution = copy.deepcopy(solution)
            move_type = random.choice(["tile", "pe", "buffer"])
            if move_type == "tile":
                idx = random.randint(0, len(new_solution) - 1)
                if isinstance(new_solution[idx][1], int):
                    new_solution[idx][1] = max(1, int(new_solution[idx][1] * random.uniform(0.9, 1.1)))
            elif move_type == "pe":
                pe_idx = next((i for i, x in enumerate(new_solution) if x[0] == "PE"), None)
                if pe_idx is not None:
                    new_solution[pe_idx][1] = max(1, new_solution[pe_idx][1] + random.choice([-1, 1]))
            elif move_type == "buffer":
                l1_idx = next((i for i, x in enumerate(new_solution) if x[0] == "L1"), None)
                l2_idx = next((i for i, x in enumerate(new_solution) if x[0] == "L2"), None)
                if l1_idx is not None and l2_idx is not None:
                    shift_amount = random.randint(-100, 100)
                    new_solution[l1_idx][1] = max(1, new_solution[l1_idx][1] + shift_amount)
                    new_solution[l2_idx][1] = max(1, new_solution[l2_idx][1] - shift_amount)
            neighbors.append(new_solution)
        return neighbors

    def run_tabu_search(self, dimension, num_iterations=100, tabu_size=10):
        """Tabu Search optimization replacing GA."""
        self.stage_idx = 0  # Setting the search stage
        current_solution = self.create_genome_fixedSL()
        best_solution = copy.deepcopy(current_solution)
        best_fitness = self.evaluate_solution(current_solution)

        tabu_list = []

        for _ in range(num_iterations):
            neighbors = self.generate_neighbors(current_solution)
            best_candidate = None
            best_candidate_score = float("-inf")

            for neighbor in neighbors:
                if neighbor in tabu_list:
                    continue
                score = self.evaluate_solution(neighbor)
                if score > best_candidate_score:
                    best_candidate = neighbor
                    best_candidate_score = score

            if best_candidate is None:
                break  # No valid moves

            current_solution = best_candidate
            tabu_list.append(copy.deepcopy(best_candidate))
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)  # Maintain tabu list size

            if best_candidate_score > best_fitness:
                best_solution = best_candidate
                best_fitness = best_candidate_score

        self.best_sol = best_solution
        self.best_reward = best_fitness
        return {"best_sol": best_solution, "best_reward": best_fitness}
