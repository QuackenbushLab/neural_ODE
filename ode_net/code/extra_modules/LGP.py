import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import mpmath
import sys
import random

from copy import deepcopy
from sympy import Symbol, sympify, sin, cos, Abs
from solve_eq import generate_grid
from datahandler import DataHandler
from odenet import ODENet
from termcolor import colored

instruction_length = 4

def init_population(size, max_starting_instructions, num_operators, num_var_reg, num_const_reg):
    '''
        Initialize the population
    '''
    population = []
    tot_reg = num_var_reg + num_const_reg

    for i in range(size):
        num_instructions = np.random.randint(max_starting_instructions)
        individual = [0 for x in range(instruction_length*num_instructions)]
        for j in range(num_instructions):
            operator = np.random.randint(num_operators)
            dest_reg = np.random.randint(num_var_reg)
            operand1 = np.random.randint(tot_reg)
            operand2 = np.random.randint(tot_reg)

            start = j * instruction_length
            individual[start] = operator
            individual[start + 1] = dest_reg
            individual[start + 2] = operand1
            individual[start + 3] = operand2
        
        population.append(individual)
    return population

def decode_chromosome(individual, const_reg, num_var_reg, data, data_dim):

    # Initialize registers
    chrom_len = len(individual)
    num_instructions = int(chrom_len / instruction_length)
    num_data_points = len(data)
    var_reg = [0 for x in range(num_var_reg)]
    combined_reg = var_reg + const_reg + [0 for x in range(data_dim)]
    reg_mat = np.array([combined_reg.copy() for x in range(num_data_points)])
    for i in range(data_dim):
        reg_mat[:,i] = data[:,i]
        reg_mat[:,-i - 1] = data[:,i]
    
    # Decode the chromosome
    for i in range(num_instructions):
        start = i*instruction_length
        operator = individual[start]
        dest = individual[start + 1]
        operand1_indx = individual[start + 2]
        operand2_indx = individual[start + 3]
        operand1 = reg_mat[:, operand1_indx]
        operand2 = reg_mat[:, operand2_indx]

        if operator == 0:
            reg_mat[:, dest] = operand1 + operand2
        elif operator == 1:
            reg_mat[:, dest] = operand1 - operand2
        elif operator == 2:
            reg_mat[:, dest] = operand1 * operand2
        elif operator == 3:
            for i in range(len(reg_mat[:,dest])):
                if operand2[i] == 0:
                    reg_mat[i, dest] = 99999999
                else:
                    reg_mat[i, dest] = operand1[i] / operand2[i]
        elif operator == 4:
            reg_mat[:, dest] = np.sin(operand1)
        elif operator == 5:
            reg_mat[:, dest] = np.cos(operand1)
        elif operator == 6:
            reg_mat[:, dest] = np.abs(operand1)
        else:
            print("Invalid operator: {}".format(operator))

        if np.isnan(reg_mat[:,dest]).any():
            print("Vafan händer")
    
    decoded_val = np.zeros((num_data_points, data_dim))
    for i in range(data_dim):
        decoded_val[:,i] = reg_mat[:,i]

    return decoded_val

def evaluate_individual(decoded_chromosome, target):
    ''' Currently using MSE for evaluation '''
    mse = np.mean(np.square(decoded_chromosome - target))
    return 1/mse

def tournament_select(population_fitness, tourn_prob, tourn_size):
    pop_size = len(population_fitness)

    random_indx = np.zeros(tourn_size)
    selected_fitness = np.zeros(tourn_size)

    for i in range(tourn_size):
        indx = int(np.random.rand() * pop_size)
        random_indx[i] = indx
        selected_fitness[i] = population_fitness[indx]

    current_vec_len = tourn_size
    chromosome_selected = False

    while current_vec_len > 1 and not chromosome_selected:
        rnd = np.random.rand()
        i = np.argmax(selected_fitness)
        if rnd < tourn_prob:
            selected_indx = random_indx[i]
            chromosome_selected = True
        else:
            random_indx = np.delete(random_indx, i)
            selected_fitness = np.delete(selected_fitness, i)
            current_vec_len = current_vec_len - 1

    # No index selected, select the last one
    if not chromosome_selected:
        selected_indx = random_indx[0]
    
    return int(selected_indx)

def crossover(chrom1, chrom2, max_instructions):
    ''' Perform crossover with two chromosomes '''
    max_length = max_instructions * instruction_length
    chrom1_len = len(chrom1)
    chrom2_len = len(chrom2)
    chrom1_instructions = int(chrom1_len / instruction_length)
    chrom2_instructions = int(chrom2_len / instruction_length)

    if chrom1_instructions > 0:
        r = np.random.randint(chrom1_instructions, size=2)
        chrom1_crosspt1 = instruction_length * np.min(r)
        chrom1_crosspt2 = instruction_length * np.max(r)
    else:
        chrom1_crosspt1 = 0
        chrom1_crosspt2 = 0

    if chrom2_instructions > 0:
        r = np.random.randint(chrom2_instructions, size=2)
        chrom2_crosspt1 = instruction_length * np.min(r)
        chrom2_crosspt2 = instruction_length * np.max(r)
    else:
        chrom2_crosspt1 = 0
        chrom2_crosspt2 = 0

    updated_chrom1 = []
    updated_chrom1 = updated_chrom1 + chrom1[0:chrom1_crosspt1]
    updated_chrom1 = updated_chrom1 + chrom2[chrom2_crosspt1:chrom2_crosspt2] + chrom1[chrom1_crosspt2::]

    updated_chrom2 = []
    updated_chrom2 = updated_chrom2 + chrom2[0:chrom2_crosspt1]
    updated_chrom2 = updated_chrom2 + chrom1[chrom1_crosspt1:chrom1_crosspt2] + chrom2[chrom2_crosspt2::]

    if max_length > 0:
        if len(updated_chrom1) > max_length:
            updated_chrom1 = updated_chrom1[0:max_length]

        if len(updated_chrom2) > max_length:
            updated_chrom2 = updated_chrom2[0:max_length]

    return updated_chrom1, updated_chrom2

def mutate(chromosome, mutate_prob, num_operators, num_var_reg, num_const_reg):
    ''' Mutate a chromosome '''
    tot_reg = num_var_reg + num_const_reg
    num_genes = len(chromosome)
    mutated_chromosome = deepcopy(chromosome)

    for i in range(num_genes):
        r = np.random.rand()
        if r < mutate_prob:
            curr_genome = i % instruction_length
            if curr_genome == 0:
                mutated_chromosome[i] = np.random.randint(num_operators)
            elif curr_genome == 1:
                mutated_chromosome[i] = np.random.randint(num_var_reg)
            else:
                mutated_chromosome[i] = np.random.randint(tot_reg)

    return mutated_chromosome

def insert_best_individual(population, best_chromosome, num_copies_to_insert):
    for i in range(num_copies_to_insert):
        population[i] = deepcopy(best_chromosome)
    return population

def calc_target(odenet, data_handler):
    with torch.no_grad():
        data = data_handler.data_pt
        t = data_handler.time_pt
        data_vec = torch.zeros((data_handler.datasize, data_handler.dim))
        target = torch.zeros((data_handler.datasize, data_handler.dim))
        startingpt = 0
        for i in range(len(data)):
            data_vec[startingpt:startingpt+t[i].shape[0]] = data[i].squeeze()
            target[startingpt:startingpt+t[i].shape[0]] = odenet.forward(t[i], data[i]).squeeze()
            startingpt = startingpt + t[i].shape[0]
        return data_vec.numpy(), target.numpy(), data_handler.dim

def get_symbolic_expression(chromosome, const_reg, num_var_reg, dim):
    ''' Decode chromosome and find symbolic expression '''
    chrom_length = len(chromosome)
    num_instructions = int(chrom_length / instruction_length)
    tot_num_reg = num_var_reg + len(const_reg)
    combined_reg = [0 for x in range(num_var_reg)] + const_reg + [0 for x in range(dim)]
    for i in range(dim):
        x = Symbol('x{}'.format(i))
        combined_reg[i] = x
        combined_reg[-i - 1] = x
    
    for i in range(num_instructions):
        start = i*instruction_length
        operator = chromosome[start]
        dest_reg = chromosome[start + 1]
        operand1_indx = chromosome[start + 2]
        operand2_indx = chromosome[start + 3]
        operand1 = combined_reg[operand1_indx]
        operand2 = combined_reg[operand2_indx]

        if operator == 0:
            combined_reg[dest_reg] = operand1 + operand2
        elif operator == 1:
            combined_reg[dest_reg] = operand1 - operand2
        elif operator == 2:
            combined_reg[dest_reg] = operand1 * operand2
        elif operator == 3 and not operand2 == 0:
            combined_reg[dest_reg] = operand1 / operand2
        elif operator == 3 and operand2 == 0:
            combined_reg[dest_reg] = mpmath.inf
        elif operator == 4:
            combined_reg[dest_reg] = sin(operand1)
        elif operator == 5:
            combined_reg[dest_reg] = cos(operand1)
        elif operator == 6:
            combined_reg[dest_reg] = Abs(operand1)
        else:
            print("Unused operator found: {}".format(operator))

    if combined_reg[dest_reg] == np.nan:
        print("Vafan händer")

    simplified_eq = []
    for i in range(dim):
        simplified_eq.append(sympify(combined_reg[i]))
        #simplified_eq[i] = simplified_eq[i].replace('**', '^')
    return simplified_eq

parser = argparse.ArgumentParser('Testing')
parser.add_argument('--data', type=str, default='.\data\lotka_volterra_2_traj.csv')
parser.add_argument('--model', type=str, default='D:\\Skola\\MSc-Thesis\\Base\\output\\2019-5-14(12;45)_lotka_volterra_2_traj_20epochs\\best_model.pt')
args = parser.parse_args()

if __name__ == "__main__":
    print("LGP for ODENet function fitting v1.0")
    
    data_handler = DataHandler.fromcsv(args.data, 'cpu', 0.0)
    odenet = ODENet('cpu', 4)
    odenet.load(args.model)

    data, target, data_dim = calc_target(odenet, data_handler)

    error = np.infty
    target_error = 0.001
    num_gen = 1

    pop_size = 50
    max_starting_instructions = 50
    max_instructions = 500
    num_operators = 7
    num_var_reg = 8
    num_float_reg = 4
    num_int_reg = 4
    int_reg_range = 10
    num_decimals = 3
    const_reg = [0] + [np.around(np.random.rand()*0.1, decimals=num_decimals)
                ] + [np.around(np.random.rand(), decimals=num_decimals) for x in range(num_float_reg - 1)
                ] + random.sample(range(1, int_reg_range), num_int_reg)
    print("Constant register variables:")
    print(const_reg)
    #const_reg = [0.005, 0.1, 0.3, 0.5, 1, 3, 7]
    num_const_reg = len(const_reg) + data_dim

    tourn_prob = 0.8
    tourn_size = 2
    cross_prob = 0.8
    mutate_prob = 0.05
    num_copies_to_insert = 3

    # Init population
    population = init_population(pop_size, max_starting_instructions, num_operators, num_var_reg, num_const_reg)
    new_population = deepcopy(population)
    decoded_pop = deepcopy(population)
    population_fitness = [0 for x in range(pop_size)]

    best_fit_total = 0

    while error > target_error:
        # Evaluation phase
        best_indx = 1
        best_fitness = 0
        for i in range(pop_size):
            individual = population[i]
            # Decode individual
            #print("Decode")
            decoded_chromosome = decode_chromosome(individual, const_reg, num_var_reg, data, data_dim)
            decoded_pop[i] = decoded_chromosome

            #print("Evaluate")
            fitness = evaluate_individual(decoded_chromosome, target)
            population_fitness[i] = fitness
            if fitness > best_fitness:
                best_indx = i
                best_fitness = fitness
                error = 1/fitness
                if fitness > best_fit_total:
                    best_fit_total = fitness
                    print("New best fit found at generation {} with error {:.8f}".format(num_gen, error))
                    output = get_symbolic_expression(individual, const_reg, num_var_reg, data_dim)
                    for i in range(len(output)):
                        print(colored("\tx{}_dot = {}".format(i, output[i]), 'yellow'))

        best_chromosome = deepcopy(population[best_indx])

        # Mutation phase
        for i in range(0, pop_size, 2):
            #print("Tournament")
            if tourn_size > 0:
                chromosome_1_indx = tournament_select(population_fitness, tourn_prob, tourn_size)
                chromosome_2_indx = tournament_select(population_fitness, tourn_prob, tourn_size)

            chrom1 = population[chromosome_1_indx]
            chrom2 = population[chromosome_2_indx]

            rnd = np.random.rand()
            #print("Cross")
            if rnd <= cross_prob:
                new_chrom1, new_chrom2 = crossover(chrom1, chrom2, max_instructions)
            else:
                new_chrom1 = chrom1
                new_chrom2 = chrom2
            #print("Mutate")
            mutated_chrom1 = mutate(new_chrom1, mutate_prob, num_operators, num_var_reg, num_const_reg)
            mutated_chrom2 = mutate(new_chrom2, mutate_prob, num_operators, num_var_reg, num_const_reg)

            new_population[i] = mutated_chrom1
            if i + 1 < pop_size:
                new_population[i + 1] = mutated_chrom2

        # Update population phase
        #print("Insert")
        if num_copies_to_insert <= pop_size:
            population = insert_best_individual(new_population, best_chromosome, num_copies_to_insert)

        if num_gen % 100 == 0:
            print("Finished {} generations.".format(num_gen))
        num_gen = num_gen + 1

    output = get_symbolic_expression(best_chromosome, const_reg, num_var_reg, data_dim)
    print("Fit with error less than {} found after {} generations:".format(target_error, num_gen - 1))
    for i in range(len(output)):
        print(colored("\tx{}_dot = {}".format(i, output[i]), 'green'))

            