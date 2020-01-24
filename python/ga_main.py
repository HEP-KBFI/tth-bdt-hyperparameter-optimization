'''Main functions for the genetic algorithm'''
from __future__ import division
import copy
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


def fitness_calculation(population, settings, data, evaluate):
    '''Calculate the fitness scores of the given generation

    Parameters
    ----------
    population : list
        A group of individuals to be evaluated
    settings : dict
        Settings of the genetic algorithm
    data : dict
        Training and testing data sets
    evaluate : function
        Function used to calculate scores

    Returns
    -------
    fitnesses : list
        Fitness scores corresponding to the given population
    score_dicts : list
        List of score_dicts of each individual
    pred_trains : list
        List of pred_trains corresponding to each individual
    pred_tests : list
        List of pred_tests corresponding to each individual
    feature_importances : list
        List of feature importances corresponding to each individual
    '''
    args = inspect.getargspec(evaluate)

    if len(args[0]) == 3:
        scores, pred_trains, pred_tests, feature_importances = evaluate(
            population, data, settings)
    elif len(args[0]) == 4:
        scores, pred_trains, pred_tests, feature_importances = evaluate(
            population, data, settings, len(population))

    fitnesses = universal.fitness_to_list(
        scores, fitness_key=settings['fitness_fn'])

    return fitnesses, scores, pred_trains, pred_tests, feature_importances


def elitism(population, pop_data, elites):
    '''Preserve best performing members of the previous generation

    Parameters
    ----------
    population : list
        A group of individuals
    pop_data : dict
        Data about the given population
    elites : float
        Amount of elite members to select

    Returns
    -------
    elite : list
        Best performing members of the given population
    '''
    # Create copies of data
    population = population[:]
    fitnesses = pop_data['fitnesses'][:]
    pop_data = copy.deepcopy(pop_data)

    # Initialization
    elite = []
    elite_data = {}
    for key in pop_data:
        elite_data[key] = []

    # Set num as the number of elite members to perserve
    num = set_num(elites, population)

    # Select num amount of elite members from the population
    while num > 0:
        index = np.argmax(fitnesses)
        del fitnesses[index]
        for key in pop_data:
            elite_data[key].append(pop_data[key].pop(index))
        elite.append(population.pop(index))
        num -= 1

    return elite, elite_data


def culling(
        population,
        fitnesses,
        settings,
        data,
        parameters,
        create_set,
        evaluate
):
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
    create_set : function
        Function used to generate a population
    evaluate : function
        Function used to calculate scores

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
    fitnesses += fitness_calculation(
        new_members, settings, data, evaluate)[0]

    return population, fitnesses


def new_population(population, pop_data, settings, parameters):
    '''Create the next generation population.

    Parameters
    ----------
    population : list
        Current set of individuals
    pop_data : dict
        Data about the given population
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
    next_population, next_pop_data = elitism(
        population, pop_data, settings['elites'])

    # Generate offspring to fill the new generation
    while len(next_population) < len(population):
        parents = select.tournament(population, pop_data['fitnesses'])
        offspring = gc.kpoint_crossover(
            parents, parameters, settings['mut_chance'])

        # No duplicate members
        if offspring not in next_population:
            next_population.append(offspring)

    return next_population, next_pop_data


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


def sub_evolution(
        subpopulations,
        settings,
        data,
        parameters,
        create_set,
        evaluate
):
    '''Evolve subpopulations separately and then merge them into one

    Parameters
    ----------
    subpopulations : list
        Initial subpopulations
    settings : dict
        Settings of the genetic algorithm
    data : dict
        Data sets for testing and training
    parameters : dict
        Descriptions of the xgboost parameters
    create_set : function
        Function used to generate a population
    evaluate : function
        Function used to calculate scores

    Returns
    -------
    merged_population : list
        Final subpopulations merged into one
    tracker : dict
        History of best scores from all iterations
    compactness : dict
        History of compactness scores from all iterations
    '''
    compactnesses = {}
    merged_population = []
    sub_iteration = 1
    tracker = {}

    # Evolution for each subpopulation
    for population in subpopulations:
        print('\n::::: Subpopulation: ' + str(sub_iteration) + ' :::::')
        output = evolve(
            population, settings, data, parameters, create_set, evaluate)

        # Saving results in dictionaries
        for key in output['scores']:
            try:
                tracker[key].update({sub_iteration: output['scores'][key]})
            except KeyError:
                tracker[key] = {}
                tracker[key].update({sub_iteration: output['scores'][key]})
        compactnesses.update({sub_iteration: output['compactnesses']})

        # Gather final generations of each subpopulation
        merged_population += output['population']
        sub_iteration += 1

    return merged_population, tracker, compactnesses


# WORK IN PROGRESS
# def arrange_population(subpopulations, pop_data):
#     '''Arrange populations into separate lists in preparation for evaluation

#     Parameters
#     ----------
#     subpopulations : list
#         List of subpopulations
#     pop_data : dictionary
#         Information about the current subpopulations

#     Returns
#     -------
#     eval_pop : list
#         Members from all subpopulations to be evaluated
#     rest_pop : list
#         Members from all subpopulations already evaluated
#     sub_data : dict
#         Information about which subpopulation does a given member belong to
#     '''
#     eval_pop = []
#     rest_pop = []
#     sub_data = {
#         'eval_pop': [],
#         'rest_pop': []
#     }

#     for i, population in enumerate(subpopulations):
#         try:
#             if pop_data['fitnesses'][i]:
#                 for j, member in enumerate(population):
#                     if j in range(len(pop_data['fitnesses'][i])):
#                         rest_pop.append(member)
#                         sub_data['rest_pop'].append(i)
#                     else:
#                         eval_pop.append(member)
#                         sub_data['eval_pop'].append(i)
#             else:
#                 eval_pop = population
#                 for member in population:
#                     sub_data['eval_pop'].append(i)
#         except KeyError:
#             eval_pop = population
#             for member in population:
#                 sub_data['eval_pop'].append(i)

#     return eval_pop, rest_pop, sub_data


# def evolve_v2(population):
#     iteration = 0
#     improvement = 1
#     fitnesses = []
#     improvements = []
#     compactnesses = []
#     tracker = {}
#     pop_data = {}

#     # Evolution loop
#     while (improvement > settings['threshold']
#            and iteration <= settings['iterations']):

#         # Generate a new population

#         ###

#         # Arrange population
#         eval_pop, rest_pop, sub_data = arrange_population(subpopulations, pop_data)

#         # Calculate fitness scores
#         fitnesses, scores, pred_trains, pred_tests, feature_importances = \
#             fitness_calculation(eval_pop, settings, data, evaluate)

#         if iteration == 0:
#             tracker = score_tracker(
#                 tracker, scores, fitnesses, initialize=True)
#         else:
#             tracker = score_tracker(tracker, scores, fitnesses)

#         pop_data = {
#             'scores': scores,
#             'pred_trains': pred_trains,
#             'pred_tests': pred_tests,
#             'feature_importances': feature_importances,
#             'fitnesses': fitnesses
#         }

###

def evolve(population, settings, data, parameters, create_set, evaluate):
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
    create_set : function
        Function used to generate a population
    evaluate : function
        Function used to calculate scores

    Returns
    -------
    output : dict
        All gathered information about the final population
    '''
    fitnesses = []
    improvement = 1
    improvements = []
    compactnesses = []
    iteration = 0
    tracker = {}
    pop_data = {}

    # Evolution loop
    while (improvement > settings['threshold']
           and iteration <= settings['iterations']):

        # Generate a new population
        if iteration != 0:
            print('::::: Iteration:     ' + str(iteration) + ' :::::')
            population, fitnesses = culling(
                population,
                fitnesses,
                settings,
                data,
                parameters,
                create_set,
                evaluate
            )
            population, pop_data = new_population(
                population, pop_data, settings, parameters)

        # Separate population into two parts for time efficiency
        try:
            if pop_data['fitnesses']:
                eval_pop = [] # population members to be evaluated
                rest_pop = [] # population members already evaluated
                for i, member in enumerate(population):
                    if i in range(len(pop_data['fitnesses'])):
                        rest_pop.append(member)
                    else:
                        eval_pop.append(member)
            else:
                eval_pop = population
        except KeyError:
            eval_pop = population

        # Calculate fitness scores of the population
        fitnesses, scores, pred_trains, pred_tests, feature_importances = \
            fitness_calculation(eval_pop, settings, data, evaluate)

        # Gather and save results
        try:
            if rest_pop:
                scores = pop_data['scores'] + scores
                pred_trains = pop_data['pred_trains'] + pred_trains
                pred_tests = pop_data['pred_tests'] + pred_tests
                feature_importances = (pop_data['feature_importances']
                                       + feature_importances)
                fitnesses = pop_data['fitnesses'] + fitnesses
        except UnboundLocalError:
            pass

        if iteration == 0:
            tracker = score_tracker(
                tracker, scores, fitnesses, initialize=True)
        else:
            tracker = score_tracker(tracker, scores, fitnesses)

        pop_data = {
            'scores': scores,
            'pred_trains': pred_trains,
            'pred_tests': pred_tests,
            'feature_importances': feature_importances,
            'fitnesses': fitnesses
        }

        # Calculate stopping criteria
        improvements, improvement = universal.calculate_improvement_wAVG(
            tracker['avg_scores'], improvements, settings['threshold'])
        compactness = universal.calculate_compactness(population)
        compactnesses.append(compactness)

        iteration += 1

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


def score_tracker(tracker, scores, fitnesses, initialize=False):
    '''Tracks best scores of each iteration

    Parameters
    ----------
    tracker : dict
        Dictionary of best scores
    scores : list
        List of dictionaries with scores of the current population
    fitnesses : list
        List of fitness scores of the current population
    initialize : bool
        Whether to initialize the dictionary

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
        tracker[key_name].append(scores[index][key])

    if initialize:
        tracker['avg_scores'] = []
        tracker['best_fitnesses'] = []

    tracker['avg_scores'].append(np.mean(fitnesses))
    tracker['best_fitnesses'].append(max(fitnesses))

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
    keys = ['g_score', 'f1_score', 'd_score', 'test_auc', 'train_auc']
    index = np.argmax(output['fitnesses'])

    # Create the dictionary
    result = {
        'best_parameters': output['population'][index],
        'best_fitnesses': output['scores']['best_fitnesses'],
        'avg_scores': output['scores']['avg_scores'],
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
    create_set : function
        Function used to generate a population
    evaluate : function
        Function used to calculate scores

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
        subpopulations = create_subpopulations(
            settings, parameters, create_set)

        # Evolve subpopulations
        merged_population, tracker, compactnesses = sub_evolution(
            subpopulations, settings, data, parameters, create_set, evaluate)

        # Evolve merged population
        print(('\n::::: Merged population:::::'))
        output = evolve(
            merged_population,
            settings,
            data,
            parameters,
            create_set,
            evaluate
        )

        for key in tracker:
            tracker[key].update({'final': output['scores'][key]})

        compactnesses.update({'final': output['compactnesses']})

        output['scores'] = tracker
        output['compactnesses'] = compactnesses

    else:

        # Create one population
        print('::::::: Creating population ::::::::\n')
        population = create_set(parameters, settings['sample_size'])

        # Evolve population
        output = evolve(
            population, settings, data, parameters, create_set, evaluate)

    return finalize_results(output, data)
