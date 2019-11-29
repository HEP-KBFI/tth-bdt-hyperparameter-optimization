'''Main functions for the genetic algorithm'''
from __future__ import division
import numpy as np
from tthAnalysis.bdtHyperparameterOptimization import xgb_tools as xt
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import ga_selection as select
from tthAnalysis.bdtHyperparameterOptimization import ga_crossover as gc


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
    new_members = xt.prepare_run_params(parameters, size)
    population += new_members
    fitnesses += universal.fitness_to_list(
        xt.ensemble_fitnesses(new_members, data, settings)[0])

    return population, fitnesses


# NOT IN USE ANYMORE
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
        offspring = gc.kpoint_crossover(
            parents, parameters, settings['mut_chance'])

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
        sub_population = xt.prepare_run_params(parameters, sub_size)
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
            print("Population: " + str(population))

        # Calculate fitness of the population
        score_dicts, pred_trains, pred_tests = xt.ensemble_fitnesses(
                population, data, settings)
        fitnesses = universal.fitness_to_list(score_dicts)

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
        output = evolve(merged_population, settings, data, parameters, True)

        scores_dict['best_scores'].update(
            {'final': output[1]['best_scores']})
        scores_dict['avg_scores'].update(
            {'final': output[1]['avg_scores']})
        scores_dict['worst_scores'].update(
            {'final': output[1]['worst_scores']})
        output[1] = scores_dict

    else:

        # Create one population
        print('::::::: Creating population ::::::::\n')
        population = xt.prepare_run_params(parameters, settings['pop_size'])
        print("Initial population: " + str(population))

        # Evolve population
        output = evolve(population, settings, data, parameters, True)

    # Finalize results
    index = np.argmax(output[2])
    result = {
        'best_parameters': output[0][index],
        'best_scores': output[1]['best_scores'],
        'avg_scores': output[1]['avg_scores'],
        'worst_scores': output[1]['worst_scores'],
        'pred_train': output[3][index],
        'pred_test': output[4][index],
        'data_dict': data
    }
    return result
