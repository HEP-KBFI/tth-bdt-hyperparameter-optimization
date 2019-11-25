'''Main functions for the genetic algorithm'''
from __future__ import division
import random
import numpy as np
from tthAnalysis.bdtHyperparameterOptimization import xgb_tools as xt
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import ga_selection as select


def float_encoding(value, accuracy=5):
    '''Encode a float into a bit string.

    Parameters
    ----------
    value : float
        Float to be encoded
    accuracy : int
        Number of decimals to be preserved

    Returns
    -------
    encoded : string
        Encoded value of the float with a given accuracy
    '''
    integer = int(round(value * (10 ** accuracy)))
    encoded = '{:032b}'.format(integer)
    return encoded


def float_decoding(encoded, accuracy=5):
    '''Decode an encoded float back into a float.

    Parameters
    ----------
    encoded : string
        Encoded value
    accuracy : int
        Number of decimals preserved in the float

    Returns
    -------
    value : float
        Decoded value of the float
    '''
    decoded = int(encoded, 2)
    value = decoded / accuracy
    return value


def int_encoding(value):
    '''Encode an int into a bit string.

    Parameters
    ----------
    value : int
        Integer to be encoded

    Returns
    -------
    encoded : string
        Encoded value of the integer
    '''
    encoded = '{:032b}'.format(value)
    return encoded


def int_decoding(encoded):
    '''Decode an encoded int back into a n int.

    Parameters
    ----------
    encoded : string
        Encoded value

    Returns
    -------
    decoded : int
        Decoded value of the int
    '''
    decoded = int(encoded, 2)
    return decoded


def encode_parent(parent, parameters):
    '''Encode an individual for crossover and mutation.

    Parameters
    ----------
    parent : dict
        Individual to be encoded
    parameters : dict
        xgboost settings used in the genetic algorithm

    Returns
    -------
    encoded_parent : dict
        Individual with encoded values
    '''

    # Initialization
    encoded_parent = {}

    # Encoding values in parent
    for parameter in parameters:
        key = parameter['p_name']
        true_int = parameter['true_int']
        if true_int == 'True':
            encoded_parent[key] = int_encoding(parent[key])
        else:
            encoded_parent[key] = float_encoding(parent[key])

    return encoded_parent


def decode_offspring(offspring, parameters):
    '''Decode an individual.

    Parameters
    ----------
    offspring : dict
        Individual to be decoded
    parameters : dict
        xgboost settings used in the genetic algorithm

    Returns
    -------
    decoded_offspring : dict
        Individual with decoded values
    '''

    # Initialization
    decoded_offspring = {}

    # Decoding values in offspring
    for parameter in parameters:
        key = parameter['p_name']
        true_int = parameter['true_int']
        if true_int == 'True':
            decoded_offspring[key] = int_decoding(offspring[key])
        else:
            decoded_offspring[key] = float_decoding(offspring[key])

    return decoded_offspring


def kpoint_crossover(parents, parameters, mutation_chance=0, k=1):
    '''Crossover of parents via k-point crossover.
    Default is single-point crossover.

    Parameters
    ----------
    parents : list
        Two parents to be used in the crossover
    parameters : dict
        xgboost settings used in the genetic algorithm
    mutation_chance : float
        probability of mutation for each bit
    k : int
        number of points in k-point crossover

    Returns
    -------
    offspring : dict
        New individual
    '''

    # Initialization
    points = []
    offspring = {}
    parent1 = encode_parent(parents[0], parameters)
    parent2 = encode_parent(parents[1], parameters)

    # Choose crossover points
    for i in range(k):
        point = round(random.random() * len(parent1))
        points.append(point)

    # k-point crossover of parents
    for key in parent1:
        parent1_chromosome = parent1[key]
        parent2_chromosome = parent2[key]
        curr_chromosome = parent1_chromosome
        offspring_chromosome = ''
        for i in range(len(parent1_chromosome)):
            offspring_chromosome += curr_chromosome[i]
            if i in points:
                if curr_chromosome == parent1_chromosome:
                    curr_chromosome = parent2_chromosome
                else:
                    curr_chromosome = parent1_chromosome

        # Mutation
        if mutation_chance != 0:
            offspring_chromosome = chromosome_mutate(
                offspring_chromosome, mutation_chance)

        offspring[key] = offspring_chromosome

    # Finalize offspring
    offspring = decode_offspring(offspring, parameters)
    offspring = mutation_fix(offspring, parameters)

    return offspring


def uniform_crossover(parents, parameters, mutation_chance=0):
    '''Crossover of parents via uniform crossover.

    Parameters
    ----------
    parents : list
        Two parents to be used in the crossover
    parameters : dict
        xgboost settings used in the genetic algorithm
    mutation_chance : float
        probability of mutation for each bit

    Returns
    -------
    offspring : dict
        New individual
    '''

    # Initialization
    offspring = []
    parent1 = encode_parent(parents[0], parameters)
    parent2 = encode_parent(parents[1], parameters)

    # Uniform crossover of parents
    for key in parent1:
        parent1_chromosome = parent1[key]
        parent2_chromosome = parent2[key]
        offspring_chromosome = ''
        for i, chromosome in enumerate(parent1_chromosome):
            cointoss = random.random()
            if cointoss < 0.5:
                offspring_chromosome += chromosome
            else:
                offspring_chromosome += parent2_chromosome[i]

        # Mutation
        if mutation_chance != 0:
            offspring_chromosome = chromosome_mutate(
                offspring_chromosome, mutation_chance)

        offspring[key] = offspring_chromosome

    # Finalize offspring
    offspring = decode_offspring(offspring, parameters)
    offspring = mutation_fix(offspring, parameters)

    return offspring


def chromosome_mutate(chromosome, mutation_chance):
    '''Mutation of a single chromosome.

    Parameters
    ----------
    chromosome : string
        A single chromosome of an individual to be mutated
    mutation_chance : float
        Probability of mutation for a single bit in the chromosome

    Returns
    -------
    mutated_chromosome : string
        A mutated chromosome
    '''

    # Initialization
    mutated_chromosome = {}

    # Random mutation based on mutation_chance
    for gene in chromosome:
        if random.random() < mutation_chance:
            mutated_chromosome += str(abs(int(gene) - 1))
        else:
            mutated_chromosome += gene

    return mutated_chromosome


def group_crossover(parents, parameters, mutation_chance):
    '''Crossover of parents based on grouping.

    Parameters
    ----------
    parents : list
        Two parents to be used in the crossover
    parameters : dict
        xgboost settings used in the genetic algorithm
    mutation_chance : float
        probability of mutation for each bit

    Returns
    -------
    offspring : dict
        New individual
    '''

    # Initialization
    parent1, true_corr = grouping(parents[0], parameters)
    parent2, true_corr = grouping(parents[1], parameters)
    offspring = []

    # Crossover and mutation
    for i, group in enumerate(parent1):
        cointoss = random.random()
        if cointoss < 0.5:
            if cointoss < (mutation_chance/2):
                mutated = group_mutate(
                    group,
                    mutation_chance,
                    cointoss,
                    true_corr[i]
                )
                offspring.append(mutated)
            else:
                offspring.append(group)
        else:
            if cointoss > (1 - mutation_chance/2):
                mutated = group_mutate(
                    parent2[i],
                    mutation_chance,
                    (1 - cointoss),
                    true_corr[i]
                )
                offspring.append(mutated)
            else:
                offspring.append(parent2[i])

    # Finalize offspring
    offspring = degroup(offspring)
    offspring = mutation_fix(offspring, parameters)

    return offspring

def mutation_fix(offspring, parameters):
    '''
    Fixes mutated offspring in case of parameter violation.

    Parameters
    ---------
    offspring : dict
        Mutated offspring
    parameters : dict
        xgboost settings used in the genetic algorithm

    Returns
    -------
    offspring : dict
        Offspring with fixed parameter values
    '''
    for parameter in parameters:
        # Current parameter
        key = parameter['p_name']

        # Forces int parameters to have integer values
        if parameter['true_int'] == 'True':
            offspring[key] = int(round(offspring[key]))

        # Forces parameter values not to be lower
        # than range start value
        if offspring[key] < parameter['range_start']:
            offspring[key] = parameter['range_start']

        # Forces parameter values not to exceed range end value
        elif offspring[key] > parameter['range_end']:
            offspring[key] = parameter['range_end']

    return offspring


def grouping(parent, parameters):
    '''Group elements in a parent for crossover

    Parameters
    ----------
    parent : dict
        Parent to be grouped
    parameters : dict
        xgboost settings used in the genetic algorithm

    Returns
    -------
    grouped_parent : list
        Parent with grouped parameter values
    true_corr : list
        Correlation for each group
    '''

    # Initialization
    grouped_parent = []
    group = {}
    true_corr = []

    # Grouping
    for i, param in enumerate(parameters):
        key = param['p_name']
        group[key] = parent[key]
        if i + 1 > (len(parameters) - 1):
            grouped_parent.append(group)
            true_corr.append(param['true_corr'])
        elif parameters[i]['group_nr'] != parameters[i + 1]['group_nr']:
            grouped_parent.append(group)
            true_corr.append(param['true_corr'])
            group = {}

    return grouped_parent, true_corr


def degroup(offspring):
    '''Degroup elements in offspring after crossover

    Parameters
    ----------
    offspring : list
        Grouped individual to be degrouped

    Returns
    degrouped_offspring : dict
        Degrouped individual
    '''
    degrouped_offspring = {}
    for group in offspring:
        degrouped_offspring.update(group)
    return degrouped_offspring


def group_mutate(group, mutation_chance, cointoss, pos_corr):
    '''Mutation of a single group.

    Parameters
    ----------
    group : dict
        Group of parameter values to be mutated together
    mutation_chance : float
        Probability of mutation occuring
    cointoss : float
        Random number indicating current probability value
    pos_corr : bool
        Group is correlated either positively (True)
        or negatively (False)

    Returns
    -------
    mutation : dict
        Group of mutated parameter values
    '''

    # Initialization
    mutation = {}

    if pos_corr == 'True':

        # Positively correlated mutation
        for i, key in enumerate(group):
            if random.random() < 0.5:
                mutation[key] = (group[key]
                                 + (mutation_chance - cointoss)*group[key])
            else:
                mutation[key] = (group[key]
                                 - (mutation_chance - cointoss)*group[key])
    else:

        # Negatively correlated mutation
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


def set_num(amount, population):
    '''Set num as the amount indicated for the given population.
    If given a float between 0 and 1, num is set
    as a given fraction of the population.
    If given a int larger than 1, num is set as that int.
    If given any other number, num is set as 0.

    Parameters
    ----------
    amount : float or int
        Given number
    population : list
        Current population

    Returns
    -------
    num : int
        Number of members of the population indicated
        by the amount given.
    '''

    # Given is a number
    if amount >= 1 and isinstance(amount, int):
        num = amount

    # Given is a fraction
    elif 0 < amount < 1:
        num = int(round(len(population)*amount))

    # Given 0 or a negative number
    else:
        num = 0

    return num


def elitism(population, fitnesses, elites):
    '''Preserve best performing members of the previous generation

    Parameters
    ----------
    population : list
        A group of individuals
    fitnesses : list
        Fitness scores corresponding to the given population
    elites : float
        Amount of elite members to select

    Returns
    -------
    elite : list
        Best performing members of the given population
    '''

    # Create copies of data
    population = population[:]
    fitnesses = fitnesses[:]

    # Initialization
    elite = []

    # Set num as the number of elite members to perserve
    num = set_num(elites, population)

    # Select num amount of elite members from the population
    while num > 0:
        index = np.argmax(fitnesses)
        del fitnesses[index]
        elite.append(population.pop(index))
        num -= 1

    return elite


def culling(population, fitnesses, settings, data, parameters):
    '''Cull worst performing members
    and replace them with random new ones

    Parameters
    ----------
    population : list
        A group of individuals
    fitnesses : list
        Fitness scores corresponding to the given population
    settings : dict
        Settings of the genetic algorithm
    data : dict
        Training and testing data sets
    parameters : dict
        Descriptions of the xgboost parameters

    Returns
    -------
    population : list
        New population with worst performing members replaced
    fitnesses : list
        Fitness scores corresponding to the new population
    '''

    # Set num as the number of members to destroy
    num = set_num(settings['culling'], population)

    # Population size to be replaced
    size = num

    # Destroy num amount of members from the population
    while num > 0:
        index = np.argmin(fitnesses)
        del fitnesses[index]
        del population[index]
        num -= 1

    # Replace destroyed members
    new_members = xt.prepare_run_params(
        settings['nthread'], parameters, size)
    population += new_members
    fitnesses += xt.ensemble_fitnesses(
        new_members, data, settings)[0]

    return population, fitnesses


def add_parameters(offspring, nthread):
    '''Add missing parameters to an offspring.

    Parameters
    ----------
    offspring : dict
        An individual with missing parameters
    nthread : int
        Number of threads

    Returns
    -------
    offspring : dict
        An individual with a complete set of parameters
    '''
    params = {
        'silent': 1,
        'objective': 'multi:softprob',
        'num_classes': 10,
        'nthread': nthread,
        'seed': 1,
    }
    offspring.update(params)
    return offspring


def new_population(population, fitnesses, settings, parameters):
    '''Create the next generation population.

    Parameters
    ----------
    population : list
        Current set of individuals
    fitnesses : list
        Fitness scores corresponding to the given population
    settings : dict
        Settings of the genetic algorithm
    parameters: dict
        Descriptions of the xgboost parameters

    Returns
    -------
    next_population : list
        Newly generated set of individuals
    '''

    # Add best members of previous population into the new population
    next_population = elitism(population, fitnesses, settings['elites'])

    # Generate offspring to fill the new generation
    while len(next_population) < len(population):
        parents = select.tournament(population, fitnesses)
        offspring = add_parameters(
            group_crossover(parents, parameters, settings['mut_chance'], ),
            settings['nthread'])

        # No duplicate members
        if offspring not in next_population:
            next_population.append(offspring)

    return next_population


def create_subpopulations(settings, parameters):
    '''Create num amount of subpopulations

    Parameters
    ----------
    settings : dict
        Settings of the genetic algorithm
    parameters: dict
        Descriptions of the xgboost parameters

    Returns
    -------
    subpopulations : list
        Randomly generated subpopulations
    '''

    # Initialization
    subpopulations = []
    size = settings['pop_size']
    num = settings['sub_pops']

    # Return empty list in case of invalid settings
    if num > size:
        return subpopulations

    # Divide population into num amount of subpopulations
    for i in range(num):
        if i == 0:
            sub_size = size//num + size % num
        else:
            sub_size = size//num

        # Generate subpopulation
        sub_population = xt.prepare_run_params(
            settings['nthread'], parameters, sub_size)
        subpopulations.append(sub_population)

    return subpopulations


def sub_evolution(subpopulations, settings, data, parameters):
    '''Evolve subpopulations separately and then merge them into one

    Parameters
    ----------
    subpopulations : list
        Initial subpopulations
    settings : dict
        Settings of the genetic algorithm
    data: dict
        Data sets for testing and training
    parameters: dict
        Descriptions of the xgboost parameters

    Returns
    -------
    merged_population : list
        Final subpopulations merged into one
    scores_dict : dict
        History of score improvements
    '''

    # Initialization
    best_scores = {}
    avg_scores = {}
    worst_scores = {}
    merged_population = []
    sub_iteration = 1

    # Evolution for each subpopulation
    for population in subpopulations:
        print('\n::::: Subpopulation: ' + str(sub_iteration) + ' :::::')
        final_population, scores_dict = evolve(
            population, settings, data, parameters)

        # Saving results in dictionaries
        # (key indicates the subpopulation)
        best_scores[sub_iteration] = scores_dict['best_scores']
        avg_scores[sub_iteration] = scores_dict['avg_scores']
        worst_scores[sub_iteration] = scores_dict['worst_scores']

        # Gather final generations of each subpopulation
        merged_population += final_population
        sub_iteration += 1

    # Collect results
    scores_dict = {
        'best_scores': best_scores,
        'avg_scores': avg_scores,
        'worst_scores': worst_scores
    }

    return merged_population, scores_dict


def evolve(population, settings, data, parameters, final=False):
    '''Evolve a population until reaching the threshold
    or maximum number of iterations

    Parameters
    ----------
    population : list
        Initial population
    settings : dict
        Settings of the genetic algorithm
    data: dict
        Data sets for testing and training
    parameters: dict
        Descriptions of the xgboost parameters
    final : bool
        Whether the evolution is the last one

    Returns
    -------
    population : list
        Final population
    scores_dict : dict
        History of score improvements
    fitnesses : list
        Fitness scores corresponding to the final population
    pred_trains : list
    pred_tests : list
    '''

    # Initialization
    fitnesses = []
    best_scores = []
    avg_scores = []
    worst_scores = []
    improvement = 1
    improvements = []
    iteration = 0

    # Evolution loop
    while (improvement > settings['threshold']
           and iteration <= settings['iterations']):

        # Generate a new population
        if iteration != 0:
            print('::::: Iteration:     ' + str(iteration) + ' :::::')
            population, fitnesses = culling(
                population, fitnesses, settings, data, parameters)
            population = new_population(
                population, fitnesses, settings, parameters)

        # Calculate fitness of the population
        fitnesses, pred_trains, pred_tests = (
            xt.ensemble_fitnesses(
                population, data, settings)
        )

        # Save results
        best_scores.append(max(fitnesses))
        avg_scores.append(np.mean(fitnesses))
        worst_scores.append(min(fitnesses))

        # Calculate improvement
        improvements, improvement = universal.calculate_improvement_wAVG(
            avg_scores, improvements, settings['threshold'])

        iteration += 1

    # Collect results
    scores_dict = {
        'best_scores': best_scores,
        'avg_scores': avg_scores,
        'worst_scores': worst_scores
    }

    if final:
        return population, scores_dict, fitnesses, pred_trains, pred_tests

    return population, scores_dict


def evolution(settings, data, parameters):
    '''Evolution of the parameter values

    Parameters
    ----------
    settings : dict
        Settings of the genetic algorithm
    data: dict
        Data sets for testing and training
    parameters: dict
        Descriptions of the xgboost parameters

    Returns
    -------
    result : dict
        Result of the run of the genetic algorithm
    '''

    if settings['sub_pops'] > 1:

        # Checking settings for validity
        assert settings['pop_size'] > settings['sub_pops'], \
            'Invalid parameters for subpopulation creation'

        # Create subpopulations
        print('::::::: Creating subpopulations ::::::::')
        subpopulations = create_subpopulations(settings, parameters)

        # Evolve subpopulations
        merged_population, scores_dict = sub_evolution(
            subpopulations, settings, data, parameters)

        # Evolve merged population
        print(('\n::::: Merged population:::::'))
        population, final_scores_dict, fitnesses, pred_trains, pred_tests = \
            evolve(merged_population, settings, data, parameters, True)

        scores_dict['best_scores'].update(
            {'final': final_scores_dict['best_scores']})
        scores_dict['avg_scores'].update(
            {'final': final_scores_dict['avg_scores']})
        scores_dict['worst_scores'].update(
            {'final': final_scores_dict['worst_scores']})

    else:

        # Create one population
        print('::::::: Creating population ::::::::\n')
        population = xt.prepare_run_params(
            settings['nthread'], parameters, settings['pop_size'])

        # Evolve population
        population, scores_dict, fitnesses, pred_trains, pred_tests = evolve(
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
