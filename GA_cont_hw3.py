# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:06:08 2017

@author: zeman
"""
import random

'''Global Variables'''
num_generation=100
size_population=50
chromosome_len=64
Pc=0.9
Pm=0.04

'''Compare the probability with random number '''
def draw_prob(probability):
    return random.random() < probability

'''Create random population string for starting'''
def create_random_initial_pop(size, chrom_length):
    string_rand = ''.join('0' if draw_prob(0.5) else '1' for _ in range(chrom_length))
    return [string_rand for _ in range(size)]

'''Selection of parents from the population based on normalized roulette rule'''
def ga_selection(population, fitness_function): #, min_or_max=MAX):

    #ROULETTE RULES --for weights of each individuals
    # Compute probability for each individual in the population
    min_fitness_val = min(fitness_function(p) for p in population)
    
    def calc_weight(p):
        fitness = fitness_function(p)

        return 1 / (fitness - min_fitness_val + 1) # since we are looking for a minimum fitness

    # Since this is minimization problem, we normalize each individual's weights with respect to the total weight
    pop_weight = [(p, calc_weight(p)) for p in population]
    tot_weight = sum(weight for p, weight in pop_weight)
    cumulat_prob = [(p, weight/tot_weight) for p, weight in pop_weight]

    # Select individuals according to their weight (roulette) and store them in new population (same number as population)
    new_population = []
    for n in range(len(population)):
        randm = random.random()
        partial_cumul = 0
        for member, member_norm_weight in cumulat_prob: #member_norm_weight = memb_weight/total_weight
            partial_cumul += member_norm_weight
            if randm <= partial_cumul:
                new_population.append(member)
                break                               
    
    return new_population

'''cross over on two parents '''
def cros_over(parent1, parent2, cross_idx):
    head_p1, tail_p1 = parent1[:cross_idx], parent1[cross_idx:]
    head_p2, tail_p2 = parent2[:cross_idx], parent2[cross_idx:]
    return head_p1 + tail_p2, head_p2 + tail_p1

'''cross over population with random probability Pc'''
def cross_over_population(population, Pc):
    cross_pairs = []
    new_generation = []
    while len(population) > 1:
        cross_pairs.append((population.pop(), population.pop()))  #cut off the last member twice and then append them
    if len(population) == 1:
        new_generation.append(population.pop())
        
    for parent1, parent2 in cross_pairs:
        if draw_prob(Pc) == False: 
            # Take the parents without doing crossover
            new_generation += [parent1, parent2]
            continue
        cross_index = random.randint(1, len(parent1)-1)           # rondomly select cross over point
        child1, child2 = cros_over(parent1, parent2, cross_index)
        new_generation.append(child1)
        new_generation.append(child2)
    return new_generation

'''mutation on individual - bit flipping - called by population_mutation '''
def mutation_flip(individual, probability):
    flip = lambda b: '1' if b is '0' else '0'
    bits = (flip(bit) if draw_prob(probability) else bit for bit in individual)
    return ''.join(bits)

'''mutation on entire population based on Pm '''
def population_mutation(population, Pm):
   return [mutation_flip(p, Pm) for p in population] 

'''Run the main GA iteration'''
def ga_run_main():  
    
    # fitness function - objective function calculates fitness of decoded string 
    fitnes_func = lambda code: obj_func(*phenotype_decoder(code))
    
    # Generate innitial population
    population = create_random_initial_pop(size_population,chromosome_len)
    
    # store the population to the current population
    cur_populations = [population] # initialize with first population

    # The for loop until maximum number of generation
    for g in range(num_generation):
        population = ga_selection(population, fitnes_func ) 
        population = cross_over_population(population, Pc)
        population = population_mutation(population, Pm)
        cur_populations.append(population)                      # append to the current population
    
    return cur_populations

'''define the objective function'''
def obj_func(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

'''Decode the binary bit strings code and return decimal X and y to feed in to the obj_function'''
def phenotype_decoder(code):
    # Split the binary code into half for X and y
    middle = int(len(code)/2)  # 
    binary_x, binary_y = code[:middle], code[middle:]
    # Calculate the corresponding decimal value for x & y
    deciml_x, deciml_y = int(binary_x, 2), int(binary_y, 2)  # decimal = int(binary, 2)
    # The scale is 2^len -1,  for length for both x and y
    scale = (2**len(binary_x)) -1    # len(binary_x) = len(binary_y) 
    upper = 6
    lower = -6
    precision = (upper - lower)/(scale) 
    x = lower + deciml_x*precision
    y = lower + deciml_y*precision
    return x, y

'''Display the final solution on console'''
def display_result():

    # Run GA and collect the resulting populations
    populations = ga_run_main()
    
    # define anonymous fitness function
    fitnes_func = lambda code: obj_func(*phenotype_decoder(code))
    
    # Determine the final best solution from all the chromosomes in population
    chromosomes = {code for pop in populations for code in pop}
    best_solution = min(chromosomes, key=fitnes_func)
    best_fitness = fitnes_func(best_solution)
    
    # Display the final solution on console
    print("Best solution:", best_solution)
    print("Best fitness:", best_fitness)
    print("Decoded x & Y:", phenotype_decoder(best_solution))  

display_result() # start the program running
