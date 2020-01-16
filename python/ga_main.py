'''Main functions for the genetic algorithm'''
from __future__ import division
import inspect
import numpy as np
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


def culling(population, fitnesses, settings, data, parameters, create_set, evaluate):
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

    # If num = 0, then no culling takes place
    if num == 0:
        return population, fitnesses

    # Population size to be replaced
    size = num

    # Destroy num amount of members from the population
    while num > 0:
        index = np.argmin(fitnesses)
        del fitnesses[index]
        del population[index]
        num -= 1

    # Replace destroyed members
    new_members = create_set(parameters, size)
    population += new_members
    args = inspect.getargspec(evaluate)
    if len(args[0]) == 3:
        fitnesses += universal.fitness_to_list(
            evaluate(new_members, data, settings)[0],
            fitness_key=settings['fitness_fn'])
    elif len(args[0]) == 4:
        fitnesses += universal.fitness_to_list(
            evaluate(new_members, data, settings, size)[0],
            fitness_key=settings['fitness_fn'])

    return population, fitnesses


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


def create_subpopulations(settings, parameters, create_set):
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
    size = settings['sample_size']
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
        sub_population = create_set(parameters, sub_size)
        subpopulations.append(sub_population)

    return subpopulations


def sub_evolution(subpopulations, settings, data, parameters, create_set, evaluate):
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
    tracker = {}

    # Evolution for each subpopulation
    for population in subpopulations:
        print('\n::::: Subpopulation: ' + str(sub_iteration) + ' :::::')
        final_population, sub_tracker = evolve(
            population, settings, data, parameters, create_set, evaluate)

        # Saving results in dictionaries
        # (key indicates the subpopulation)
        # best_scores[sub_iteration] = scores_dict['best_scores']
        # avg_scores[sub_iteration] = scores_dict['avg_scores']
        # worst_scores[sub_iteration] = scores_dict['worst_scores']

        for key in sub_tracker:
            # try:
            tracker[key].update({sub_iteration: sub_tracker[key]})
            # except:
            #     tracker[key] = {}
            #     tracker[key].update({sub_iteration: sub_tracker[key]})

        # Gather final generations of each subpopulation
        merged_population += final_population
        sub_iteration += 1

    # Collect results
    # scores_dict = {
    #     'best_scores': best_scores,
    #     'avg_scores': avg_scores,
    #     'worst_scores': worst_scores
    # }

    return merged_population, tracker


def evolve(population, settings, data, parameters, create_set, evaluate, final=False):
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
    # best_scores = []
    # avg_scores = []
    # worst_scores = []
    improvement = 1
    improvements = []
    compactnesses = []
    iteration = 0
    tracker = {}

    # Evolution loop
    while (improvement > settings['threshold']
           and iteration <= settings['iterations']):

        # Generate a new population
        if iteration != 0:
            print('::::: Iteration:     ' + str(iteration) + ' :::::')
            population, fitnesses = culling(
                population, fitnesses, settings, data, parameters, create_set, evaluate)
            population = new_population(
                population, fitnesses, settings, parameters)

        # Calculate fitness of the population
        args = inspect.getargspec(evaluate)
        if len(args[0]) == 3:
            scores, pred_trains, pred_tests, feature_importances = evaluate(
                population, data, settings)
        elif len(args[0]) == 4:
            scores, pred_trains, pred_tests, feature_importances = evaluate(
                population, data, settings, len(population))

        fitnesses = universal.fitness_to_list(
            scores, fitness_key=settings['fitness_fn'])

        # Save results

        ### WORK IN PROGRESS
        if iteration == 0:
            tracker = score_tracker(tracker, scores, fitnesses, initialize=True)
        else:
            tracker = score_tracker(tracker, scores, fitnesses)
        ###

        # best_scores.append(max(fitnesses))
        # avg_scores.append(np.mean(fitnesses))
        # worst_scores.append(min(fitnesses))

        # Calculate stopping criteria
        improvements, improvement = universal.calculate_improvement_wAVG(
            tracker['avg_scores'], improvements, settings['threshold'])
        compactness = universal.calculate_compactness(population)
        compactnesses.append(compactness)

        iteration += 1

    # Collect results
    # scores_dict = {
    #     'best_scores': best_scores,
    #     'avg_scores': avg_scores,
    #     'worst_scores': worst_scores
    # }

    if final:
        # print("Tracker: ")
        # print(tracker)
        output = {
            'population': population,
            'scores': tracker,
            'fitnesses': fitnesses,
            'compactnesses': compactnesses,
            'pred_trains': pred_trains,
            'pred_tests': pred_tests,
            'feature_importances': feature_importances
        }
        return output

    return population, tracker

### WORK IN PROGRESS

def score_tracker(tracker, scores, fitnesses, initialize=False, append=True):
    '''Tracks best scores of each iteration

    Parameters
    ----------
    tracker : dict
        Dictionary of best scores
    scores : dict
        Dictionary of scores of the current population
    fitnesses : list
        List of fitness scores of the current population
    initialize : bool
        Whether to initialize the dictionary
    append : bool
        Whether to append new results

    Returns
    -------
    tracker : dict
        Dictionary of best scores from each iteration
    '''
    keys = ['g_score', 'f1_score', 'd_score', 'test_auc', 'train_auc']
    index = np.argmax(fitnesses)

    for key in keys:
        key_name = 'best_' + key + 's'
        if initialize:
            tracker[key_name] = []
        if append:
            tracker[key_name].append(scores[index][key])

    if initialize:
        tracker['avg_scores'] = []
        tracker['best_fitnesses'] = []
        # tracker['compactness'] = []

    if append:
        tracker['avg_scores'].append(np.mean(fitnesses))
        tracker['best_fitnesses'].append(max(fitnesses))
        # tracker['compactness'] = []

    return tracker


def finalize_results(output, data):
    '''Creates a dictionary of results

    Parameters
    ----------
    output : dict
        Output from the final run of evolve function
    data : dict
        Data sets for testing and training

    Returns
    -------
    result : dict
        Result of the run of the genetic algorithm
    '''

    # Initialization
    keys = ['g_score', 'f1_score', 'd_score', 'test_auc', 'train_auc']
    index = np.argmax(output['fitnesses'])

    # Create the dictionary
    result = {
        'best_parameters': output['population'][index],
        #'best_scores': output['scores']['best_scores'],
        'best_fitnesses': output['scores']['best_fitnesses'],
        'avg_scores': output['scores']['avg_scores'],
        #'worst_scores': output['scores']['worst_scores'],
        'compactnesses': output['compactnesses'],
        'pred_train': output['pred_trains'][index],
        'pred_test': output['pred_tests'][index],
        'feature_importances': output['feature_importances'][index],
        'data_dict': data
    }

    # Add tracked scores to result dictionary
    for key in keys:
        key_name = 'best_' + key + 's'
        result[key_name] = output['scores'][key_name]

    return result

###


def evolution(settings, data, parameters, create_set, evaluate):
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
        assert settings['sample_size'] > settings['sub_pops'], \
            'Invalid parameters for subpopulation creation'

        # Create subpopulations
        print('::::::: Creating subpopulations ::::::::')
        subpopulations = create_subpopulations(settings, parameters, create_set)

        # Evolve subpopulations
        merged_population, tracker = sub_evolution(
            subpopulations, settings, data, parameters, create_set, evaluate)

        # Evolve merged population
        print(('\n::::: Merged population:::::'))
        output = evolve(merged_population, settings, data, parameters, create_set, evaluate, True)

        for key in tracker:
            tracker[key].update({'final': output['scores']})

        print(tracker)

        output['scores'] = tracker

        # scores_dict['best_scores'].update(
        #     {'final': output[1]['best_scores']})
        # scores_dict['avg_scores'].update(
        #     {'final': output[1]['avg_scores']})
        # scores_dict['worst_scores'].update(
        #     {'final': output[1]['worst_scores']})
        # output[1] = scores_dict

    else:

        # Create one population
        print('::::::: Creating population ::::::::\n')
        population = create_set(parameters, settings['sample_size'])

        # Evolve population
        output = evolve(population, settings, data, parameters, create_set, evaluate, True)

    return finalize_results(output, data)
