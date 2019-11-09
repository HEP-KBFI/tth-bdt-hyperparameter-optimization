import numpy as np
import xgboost as xgb
import docopt, random, os
from tthAnalysis.bdtHyperparameterOptimization.xgb_tools import prepare_run_params
from tthAnalysis.bdtHyperparameterOptimization.universal import ensemble_fitnesses
from tthAnalysis.bdtHyperparameterOptimization.universal import calculate_improvement_wAVG

# Selection of two parents from a population based on the tournament method.
def selection(pop, fitnesses, t_size = 3, t_prob = 0.75):

    # Initialization
    parents = []
    while len(parents) < 2:
        tournament = []
        t_fitness = []

        # Randomly select tournament members
        while len(tournament) < t_size:
            select = random.randint(0, len(pop)-1)
            tournament.append(pop[select])
            t_fitness.append(fitnesses[select])

        while len(tournament) >= 1:

            # Member with highest fitness will be selected with probability of t_prob
            if random.random() < t_prob:
                parents.append(tournament[np.argmax(t_fitness)])
                break

            # Last remaining member of tournament will be selected
            elif len(tournament) == 1:
                parents.append(tournament[0])
                break

            # If member with highest fitness was not selected, then it is removed from tournament
            else:
                tournament.remove(tournament[np.argmax(t_fitness)])
                t_fitness.remove(t_fitness[np.argmax(t_fitness)])

    return parents


# crossover of parents based on grouping
def crossover(parents, mutation_chance, value_dicts):
    parent1, true_corr = Grouping(parents[0], value_dicts)
    parent2, true_corr = Grouping(parents[1], value_dicts)
    offspring = []
    for group in range(len(parent1)):
        cointoss = random.random()
        if cointoss < 0.5:
            if cointoss < (mutation_chance/2):
                mutated = Mutate(
                    parent1[group],
                    mutation_chance,
                    cointoss,
                    true_corr[group]
                )
                offspring.append(mutated)
            else:
                offspring.append(parent1[group])
        else:
            if cointoss > (1 - mutation_chance/2):
                mutated = Mutate(
                    parent2[group],
                    mutation_chance,
                    (1 - cointoss),
                    true_corr[group]
                )
                offspring.append(mutated)
            else:
                offspring.append(parent2[group])
    offspring = Degroup(offspring)
    for i, pm in enumerate(value_dicts):
        key = pm['p_name']
        if pm['true_int'] == 'True':
            offspring[key] = int(offspring[key])
        if offspring[key] < pm['range_start']:
            offspring[key] = pm['range_start']
        elif offspring[key] > pm['range_end']:
            offspring[key] = pm['range_end']
    return offspring


# group elements in a parent for crossover
def Grouping(parent, value_dicts):
    grouped_parent = []
    group = []
    new_dict = {}
    true_corr = []
    for i, param in enumerate(value_dicts):
        key = param['p_name']
        new_dict[key] = parent[key]
        if i + 1 > (len(value_dicts) - 1):
            grouped_parent.append(new_dict)
            true_corr.append(param['true_corr'])
        elif value_dicts[i]['group_nr'] != value_dicts[i + 1]['group_nr']:
            grouped_parent.append(new_dict)
            true_corr.append(param['true_corr'])
            group = []
            new_dict = {}
    return grouped_parent, true_corr


# degroup elements in offspring after crossover
def Degroup(offspring):
    degrouped_offspring = {}
    for group in offspring:
            degrouped_offspring.update(group)
    return degrouped_offspring


# mutation of a single group
def Mutate(group, mutation_chance, cointoss, pos_corr):
    mutation = {}
    if pos_corr == 'True':
        for i, key in enumerate(group):
            if random.random() < 0.5:
                mutation[key] = group[key] + (mutation_chance - cointoss)*group[key]
            else:
                mutation[key] = group[key] - (mutation_chance - cointoss)*group[key]
    else:
        for i, key in enumerate(group):
            if random.random() < 0.5:
                mutation[key] = (
                    group[key]
                    + ((-1)**i)*(mutation_chance - cointoss)*group[key]
                )
            else:
                mutation[key] = (
                    group[key]
                    - ((-1)**i)*(mutation_chance - cointoss)*group[key]
                )
    return mutation

# Preserve either a chosen amount or a chosen fraction of best performing members of the previous generation.
def elitism(population, fitnesses, elites):
    
    # Create copies of data 
    population = population[:]
    fitnesses = fitnesses[:]

    # Initialization
    elite = []

    # Set num as the number of elite members to perserve
    if elites >= 1:
        # Given is a number of elite members
        assert type(elites) == int, 'Invalid parameter for elitism'
        num = elites

    elif elites < 1 and elites > 0:
        # Given is a fraction of the population to be selected as elite members
        num = int(round(len(population)*elites))

    else: 
        # If given 0 or negative number, select no elite members
        return elite

    # Select num amount of elite members from the population
    while num > 0:
        index = np.argmax(fitnesses)
        del fitnesses[index]
        elite.append(population.pop(index))
        num -= 1

    return elite

# Add missing parameters to an offspring
def add_parameters(offspring, nthread):
    params = {
        'silent': 1,
        'objective': 'multi:softprob',
        'num_class': 10,
        'nthread': nthread,
        'seed': 1,
    }
    offspring.update(params)
    return offspring

# Create the next generation population
def new_population(population, fitnesses, settings, parameters):

    # Add best members of previous population into the new population
    new_population = elitism(population, fitnesses, settings['elites'])

    # Generate offspring to fill the new generation
    while len(new_population) < len(population):
        parents = selection(population, fitnesses)
        offspring = add_parameters(crossover(parents, settings['mut_chance'], parameters), settings['nthread'])

        # No duplicate members
        if offspring not in new_population:
            new_population.append(offspring)

    return new_population

# Create num amount of subpopulations
def create_subpopulations(settings, parameters):

    # Initialization
    subpopulations = []
    size = settings['pop_size']
    num = settings['sub_pops']

    # Return empty list in case of invalid settings (for testing purposes only)
    if num > size:
        return subpopulations

    # Divide population into num amount of subpopulations
    for i in range(num):
        if i == 0:
            sub_size = size//num + size%num
        else:
            sub_size = size//num

        # Generate subpopulation
        sub_population = prepare_run_params(settings['nthread'], parameters, sub_size)
        subpopulations.append(sub_population)

    return subpopulations

# Evolve subpopulations separately and then merge them into one
def sub_evolution(subpopulations, settings, data, parameters):

    # Initialization
    best_scores = {}
    avg_scores = {}
    worst_scores = {}
    merged_population = []
    sub_iteration = 1

    # Evolution for each subpopulation
    for population in subpopulations:
        print('\n::::: Subpopulation: ', sub_iteration, ' :::::')
        final_population, scores_dict = evolve(population, settings, data, parameters)

        # Saving results in dictionaries (key indicates the subpopulation)
        best_scores[sub_iteration] = scores_dict['best_scores']
        avg_scores[sub_iteration] = scores_dict['avg_scores']
        worst_scores[sub_iteration] = scores_dict['worst_scores']

        # Gather final generations of each subpopulation
        merged_population += final_population
        sub_iteration += 1

    # Collect results
    scores_dict = {'best_scores': best_scores, 'avg_scores': avg_scores, 'worst_scores': worst_scores}

    return merged_population, scores_dict

# Evolve a population until reaching the threshold or maximum number of iterations
def evolve(population, settings, data, parameters, final = False):

    # Initialization
    best_scores = []
    avg_scores = []
    worst_scores = []
    improvement = 1
    improvements = []
    iteration = 0

    # Evolution loop
    while improvement > settings['threshold'] and iteration <= settings['iterations']:
        
        # Generate a new population
        if iteration != 0:
            print('::::: Iteration:     ', iteration, ' :::::')
            population = new_population(population, fitnesses, settings, parameters)

        # Calculate fitness of the population
        fitnesses, pred_trains, pred_tests = (
            ensemble_fitnesses(population, data)
        )

        # Save results
        best_scores.append(max(fitnesses))
        avg_scores.append(np.mean(fitnesses))
        worst_scores.append(min(fitnesses))

        # Calculate improvement
        improvements, improvement = calculate_improvement_wAVG(
            avg_scores, improvements, settings['threshold'])

        iteration += 1

    # Collect results
    scores_dict = {'best_scores': best_scores, 'avg_scores': avg_scores, 'worst_scores': worst_scores}

    if final:
        return population, scores_dict, fitnesses, pred_trains, pred_tests

    return population, scores_dict


def evolution(settings, data, parameters):

    if settings['sub_pops'] > 1:

        # Checking settings for validity
        assert settings['pop_size'] > settings['sub_pops'], 'Invalid parameters for subpopulation creation'

        # Create subpopulations
        subpopulations = create_subpopulations(settings, parameters)

        # Evolve subpopulations
        merged_population, scores_dict = sub_evolution(subpopulations, settings, data, parameters)

        # Evolve merged population
        print(('\n::::: Merged population  :::::'))
        population, final_scores_dict, fitnesses, pred_trains, pred_tests = evolve(
            merged_population, settings, data, parameters, True)

        scores_dict['best_scores'].update({'final': final_scores_dict['best_scores']})
        scores_dict['avg_scores'].update({'final': final_scores_dict['avg_scores']})
        scores_dict['worst_scores'].update({'final': final_scores_dict['worst_scores']})

    else:

        # Create one population
        population = prepare_run_params(settings['nthread'], parameters, settings['pop_size'])

        # Evolve population
        population, scores_dicts, fitnesses, pred_trains, pred_tests = evolve(
            population, settings, data, parameters, True)

    # Finalize results
    index = np.argmax(fitnesses)
    best_parameters = population[index]
    pred_train = pred_trains[index]
    pred_test = pred_tests[index]
    result = {
        'best_parameters': best_parameters,
        'best_scores': scores_dict['best_scores'],
        'avg_scores': scores_dict['avg_scores'],
        'worst_scores': scores_dict['worst_scores'],
        'pred_test': pred_test,
        'pred_train': pred_train,
        'data_dict': data
    }
    return result
