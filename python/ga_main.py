'''Main functions for the genetic algorithm'''
from __future__ import division
import copy
import inspect
import numpy as np
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import ga_selection as select
from tthAnalysis.bdtHyperparameterOptimization import ga_crossover as gc

class Individual:
    ''' A class used to represent an individual member of a population

    Attributes
    ----------
    values : dict
        Set of parameter values
    subpop : int
        Number denoting the subpopulation to which the individual
        belongs
    score : dict
        Dictionary of scores of the individual's performance
    pred_train : list
        List of numpy arrays containing probabilities for all labels
        for the training sample
    pred_test : list
        List of numpy arrays containing probabilities for all labels
        for the testing sample
    feature_importance
    fitness : float
        Fitness score of the individual

    Methods
    -------
    add_result(score, pred_train, pred_test, feature_importance,
    settings)
        Adds the the evaluation results as attributes to the individual
    merge()
        Assignes the subpopulation number to be 0
    '''
    def __init__(self, values, subpop):
        self.values = values
        self.subpop = subpop
        self.score = None
        self.pred_train = None
        self.pred_test = None
        self.feature_importance = None
        self.fitness = None

    def __eq__(self, other):
        return self.values == other.values

    def add_result(
            self, score, pred_train, pred_test, feature_importance, settings):
        '''Adds the the evaluation results as attributes to the
        individual'''
        self.score = score
        self.pred_train = pred_train
        self.pred_test = pred_test
        self.feature_importance = feature_importance
        self.fitness = universal.fitness_to_list(
            score, fitness_key=settings['fitness_fn'])

    def merge(self):
        '''Assignes the subpopulation number to be 0'''
        self.subpop = 0


def assign_individuals(population, subpop):
    '''Assigns generated values to members of the class Individual

    Parameters
    ----------
    population : list
        List of generated values for the population
    subpop : int
        Number denoting the current subpopulation

    Returns
    -------
    individuals : list
        The population as a list of individuals
    '''
    individuals = []
    for member in population:
        individual = Individual(member, subpop)
        individuals.append(individual)
    return individuals


def create_population(settings, parameters, create_set):
    '''Creates a randomly generated population

    Parameters
    ----------
    settings : dict
        Settings of the genetic algorithm
    parameters: dict
        Descriptions of the xgboost parameters
    create_set : function
        Function used to generate a population

    Returns
    -------
    population : list
        Randomly generated population
    '''
    population = []
    size = settings['sample_size']
    num = settings['sub_pops']
    # Return empty list in case of invalid settings
    if num > size:
        return population
    # Divide population into num amount of subpopulations
    for i in range(num):
        if i == 0:
            sub_size = size//num + size % num
        else:
            sub_size = size//num
        # Generate population
        sub_population = create_set(parameters, sub_size)
        sub_population = assign_individuals(sub_population, i)
        population += sub_population
    return population


def separate_subpopulations(population, settings):
    '''Separate the population into subpopulations

    Parameters
    ----------
    population : list
        The entire population
    settings : dict
        Settings of the genetic algorithm

    Returns
    -------
    subpopulations : list
        List of subpopulations
    '''
    subpopulations = []
    # Create empty subpopulations
    for i in range(settings['sub_pops']):
        subpopulations.append([])
    # Add individuals to correct subpopulations
    for member in population:
        index = member.subpop
        subpopulations[index].append(member)
    return subpopulations


def unite_subpopulations(subpopulations):
    '''Reunite separated subpopulations

    Parameters
    ----------
    subpopulations : list
        List of subpopulations

    Returns
    -------
    population : list
        The entire population
    '''
    population = []
    for subpopulation in subpopulations:
        population += subpopulation
    return population


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
        num = int(round(len(population) * amount))
    # Given an invalid number
    else:
        num = 0
    return num


def fitness_list(population):
    '''Generate a list of fitness scores for the given population

    Parameters
    ----------
    population : list
        The given population

    Returns
    -------
    fitnesses : list
        List of fitness scores for the given population
    '''
    fitnesses = []
    for member in population:
        fitnesses.append(member.fitness)
    return fitnesses


def population_list(population):
    '''Convert the population from a list of individuals to a list of
    values

    Parameters
    ----------
    population : list
        The given population

    Returns
    value_list : list
        The given population as a list of values
    '''
    value_list = []
    for member in population:
        value_list.append(member.values)
    return value_list


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
    population : list
        A group of individuals
    '''
    args = inspect.getargspec(evaluate)
    if len(args[0]) == 3:
        scores, pred_trains, pred_tests, feature_importances = evaluate(
            population, data, settings)
    elif len(args[0]) == 4:
        scores, pred_trains, pred_tests, feature_importances = evaluate(
            population, data, settings, len(population))
    for i, member in enumerate(population):
        member.add_result(
            scores[i],
            pred_trains[i],
            pred_tests[i],
            feature_importances[i],
            settings
        )
    return population


def elitism(population, settings):
    '''Preserve best performing members of the previous generation

    Parameters
    ----------
    population : list
        A group of individuals
    settings : dict
        Settings of the genetic algorithm

    Returns
    -------
    elite : list
        Best performing members of the given population
    '''
    num = set_num(settings['elites'], population)
    population = population[:]
    fitnesses = fitness_list(population)
    elite = []
    # Select members
    while num > 0:
        index = np.argmax(fitnesses)
        del fitnesses[index]
        elite.append(population.pop(index))
        num -= 1
    return elite


def culling(
        population,
        settings,
        parameters,
        data,
        create_set,
        evaluate,
        subpop=0
):
    '''Cull worst performing members
    and replace them with random new ones

    Parameters
    ----------
    population : list
        Population to be culled
    settings : dict
        Settings of the genetic algorithm
    parameters : dict
        Descriptions of the xgboost parameters
    data : dict
        Training and testing data sets
    create_set : function
        Function used to generate a population
    evaluate : function
        Function used to calculate scores
    subpop : int (optional)
        Which subpopulation is being culled

    Returns
    -------
    population : list
        New population with worst performing members replaced
    '''
    num = set_num(settings['culling'], population)
    fitnesses = fitness_list(population)
    size = num
    # In case of no culling
    if num == 0:
        return population
    # Destroy members
    while num > 0:
        index = np.argmin(fitnesses)
        del fitnesses[index]
        del population[index]
        num -= 1
    # Replace destroyed members
    new_members = create_set(parameters, size)
    new_members = assign_individuals(new_members, subpop)
    new_members = fitness_calculation(new_members, settings, data, evaluate)
    population += new_members
    return population


def new_population(population, settings, parameters, subpop=0):
    '''Create the next generation population.

    Parameters
    ----------
    population : list
        Current set of individuals
    settings : dict
        Settings of the genetic algorithm
    parameters: dict
        Descriptions of the xgboost parameters
    subpop : int (optional)
        Which subpopulation is being updated

    Returns
    -------
    next_population : list
        Newly generated set of individuals
    '''
    # Add best members of previous population into the new population
    offsprings = []
    next_population = elitism(population, settings)
    # Generate offspring to fill the new generation
    while len(next_population) < len(population):
        fitnesses = fitness_list(population)
        value_list = population_list(population)
        parents = select.tournament(value_list, fitnesses)
        offspring = gc.kpoint_crossover(
            parents, parameters, settings['mut_chance'])
        # No duplicate members
        if offspring not in next_population:
            offsprings.append(offspring)
    next_population += assign_individuals(offsprings, subpop)
    return next_population


# def create_subpopulations(settings, parameters, create_set):
#     '''Create num amount of subpopulations

#     Parameters
#     ----------
#     settings : dict
#         Settings of the genetic algorithm
#     parameters: dict
#         Descriptions of the xgboost parameters

#     Returns
#     -------
#     subpopulations : list
#         Randomly generated subpopulations
#     '''
#     subpopulations = []
#     size = settings['sample_size']
#     num = settings['sub_pops']

#     # Return empty list in case of invalid settings
#     if num > size:
#         return subpopulations

#     # Divide population into num amount of subpopulations
#     for i in range(num):
#         if i == 0:
#             sub_size = size//num + size % num
#         else:
#             sub_size = size//num

#         # Generate subpopulation
#         sub_population = create_set(parameters, sub_size)
#         subpopulations.append(sub_population)

#     return subpopulations


# def sub_evolution(
#         subpopulations,
#         settings,
#         data,
#         parameters,
#         create_set,
#         evaluate
# ):
#     '''Evolve subpopulations separately and then merge them into one

#     Parameters
#     ----------
#     subpopulations : list
#         Initial subpopulations
#     settings : dict
#         Settings of the genetic algorithm
#     data : dict
#         Data sets for testing and training
#     parameters : dict
#         Descriptions of the xgboost parameters
#     create_set : function
#         Function used to generate a population
#     evaluate : function
#         Function used to calculate scores

#     Returns
#     -------
#     merged_population : list
#         Final subpopulations merged into one
#     tracker : dict
#         History of best scores from all iterations
#     compactness : dict
#         History of compactness scores from all iterations
#     '''
#     compactnesses = {}
#     merged_population = []
#     sub_iteration = 1
#     tracker = {}

#     # Evolution for each subpopulation
#     for population in subpopulations:
#         print('\n::::: Subpopulation: ' + str(sub_iteration) + ' :::::')
#         output = evolve(
#             population, settings, data, parameters, create_set, evaluate)

#         # Saving results in dictionaries
#         for key in output['scores']:
#             try:
#                 tracker[key].update({sub_iteration: output['scores'][key]})
#             except KeyError:
#                 tracker[key] = {}
#                 tracker[key].update({sub_iteration: output['scores'][key]})
#         compactnesses.update({sub_iteration: output['compactnesses']})

#         # Gather final generations of each subpopulation
#         merged_population += output['population']
#         sub_iteration += 1

#     return merged_population, tracker, compactnesses


def arrange_population(population):
    '''Arrange population into separate lists in preparation for
    evaluation

    Parameters
    ----------
    population : list
        Population to be arranged

    Returns
    -------
    eval_pop : list
        Members from all subpopulations to be evaluated
    rest_pop : list
        Members from all subpopulations already evaluated
    '''
    eval_pop = []
    rest_pop = []
    for member in population:
        # Find members not yet evaluated
        if member.fitness is None:
            eval_pop.append(member)
        # Find member already evaluated
        else:
            rest_pop.append(member)
    return eval_pop, rest_pop


def merge_subpopulations(subpopulations):
    '''Merge subpopulations into one population

    Parameters
    ----------
    subpopulations : list
        List of subpopulations to merge

    Returns
    -------
    population : list
        A merged population
    '''
    population = unite_subpopulations(subpopulations)
    for member in population:
        member.merge()
    return population


def evolve(population, settings, parameters, data, create_set, evaluate):
    '''Evolve a population until reaching the threshold
    or maximum number of iterations. In case of subpopulations, first
    evolve all subpopulations until reaching either criteria, then
    evolve the merged population until reaching either criteria

    Parameters
    ----------
    population : list
        Initial population
    settings : dict
        Settings of the genetic algorithm
    parameters: dict
        Descriptions of the xgboost parameters
    data: dict
        Data sets for testing and training
    create_set : function
        Function used to generate a population
    evaluate : function
        Function used to calculate scores

    Returns
    -------
    output : dict
        All gathered information
    '''
    iteration = 0
    curr_improvement = []
    improvements = {}
    compactnesses = {}
    tracker = {}
    # Evolution loop for subpopulations
    if settings['sub_pops'] > 1:
        while (iteration <= settings['iterations']
               and population):
            # Generate a new population
            if iteration != 0:
                print('::::: Iteration:     ' + str(iteration) + ' :::::')
                for i, subpopulation in enumerate(subpopulations):
                    subpopulation = culling(
                        subpopulation,
                        settings,
                        parameters,
                        data,
                        create_set,
                        evaluate,
                        i
                    )
                    subpopulation = new_population(
                        subpopulation, settings, parameters, i)
                population = unite_subpopulations(subpopulations)
            # Separate population for time efficiency
            eval_pop, rest_pop = arrange_population(population)
            # Calculate fitness scores
            eval_pop = fitness_calculation(eval_pop, settings, data, evaluate)
            population = eval_pop + rest_pop
            subpopulations = separate_subpopulations(population, settings)
            # Track scores and calculate stopping criteria
            for i, subpopulation in enumerate(subpopulations):
                if iteration == 0:
                    tracker[i] = {}
                    improvements[i] = []
                    compactnesses[i] = []
                    curr_improvement.append(1)
                    tracker[i] = score_tracker(tracker[i], subpopulation, True)
                else:
                    tracker[i] = score_tracker(tracker[i], subpopulation)
                improvements[i], curr_improvement[i] = \
                    universal.calculate_improvement_wAVG(
                        tracker[i]['avg_scores'],
                        improvements[i],
                        settings['threshold']
                    )
                compactness = universal.calculate_compactness(subpopulation)
                compactnesses[i].append(compactness)
            # Remove a subpopulation that has reached a stopping
            # criterium
            for i, improvement in enumerate(curr_improvement):
                if improvement <= settings['threshold']:
                    removed_populations += subpopulations.pop(i)
            iteration += 1
        # Merge subpopulations into one
        print('::::: Merging subpopulations :::::')
        subpopulations += removed_populations
        population = merge_subpopulations(subpopulations)
        iteration = 1

    improvement = 1
    final_improvements = []
    final_compactnesses = []
    final_tracker = {}
    # Evolution loop for single population or merged population
    while (iteration <= settings['iterations']
           and improvement > settings['threshold']):
        # Generate new population
        if iteration != 0:
            print('::::: Iteration:     ' + str(iteration) + ' :::::')
            population = culling(
                population, settings, parameters, data, create_set, evaluate)
            population = new_population(population, settings, parameters)
        # Separate population for time efficiency
        eval_pop, rest_pop = arrange_population(population)
        # Calculate fitness scores 
        eval_pop = fitness_calculation(eval_pop, settings, data, evaluate)
        population = eval_pop + rest_pop
        # Track scores and calculate stopping criteria
        if iteration == 0:
            final_tracker = score_tracker(final_tracker, population, True)
        else:
            final_tracker = score_tracker(final_tracker, population)
        final_improvements, improvement = universal.calculate_improvement_wAVG(
            final_tracker['avg_scores'],
            final_improvements,
            settings['threshold']
        )
        compactness = universal.calculate_compactness(population)
        final_compactnesses.append(compactness)
        iteration += 1

    tracker.update({'final': final_tracker})
    # improvements.update({'final': final_improvements})
    compactnesses.update({'final': final_compactnesses})

    output = {
        'population': population,
        'scores': tracker,
        'fitnesses': fitness_list(population),
        'compactnesses': compactnesses,
    }
    return output


def score_tracker(tracker, population, initialize=False):
    '''Tracks best scores of each iteration

    Parameters
    ----------
    tracker : dict
        Dictionary of best scores
    population : list
        Current population
    initialize : bool
        Whether to initialize the dictionary

    Returns
    -------
    tracker : dict
        Dictionary of best scores from each iteration
    '''
    keys = ['g_score', 'f1_score', 'd_score', 'test_auc', 'train_auc']
    fitnesses = fitness_list(population)
    index = np.argmax(fitnesses)
    for key in keys:
        key_name = 'best_' + key + 's'
        if initialize:
            tracker[key_name] = []
        tracker[key_name].append(population[index].score[key])
    if initialize:
        tracker['avg_scores'] = []
        tracker['best_fitnesses'] = []
    tracker['avg_scores'].append(np.mean(fitnesses))
    tracker['best_fitnesses'].append(max(fitnesses))
    return tracker


def find_result(tracker, key):
    ''' Create a list or dictionary of scores with a given key

    Parameters
    ----------
    tracker : dict
        A dictionary of all the tracked information
    key : string
        The key for the scores to be searched for

    Returns
    ------
    result : dict or list
        All of the scores from the tracker with the given key
    '''
    if len(tracker) > 1:
        result = {}
        for subpop in tracker:
            if subpop == 'final':
                result.update({'final': tracker[subpop][key]})
            else:
                result.update({subpop: tracker[subpop][key]})
    else:
        result += tracker['final'][key]
    return result


def finalize_result(output, data):
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
    result = {
        'best_parameters': output['population'][index],
        'best_fitnesses': find_result(output['scores'], 'best_fitnesses'),
        'avg_scores': find_result(output['scores'], 'avg_scores'),
        'compactnesses': output['compactnesses'],
        'pred_train': output['population'][index].pred_train,
        'pred_test': output['population'][index].pred_test,
        'feature_importances':
            output['population'][index].feature_importance,
        'data_dict': data
    }
    for key in keys:
        key_name = 'best_' + key + 's'
        result[key_name] = find_result(output['scores'], key_name)
    return result


def evolution(settings, parameters, data, create_set, evaluate):
    '''Evolution of the parameter values

    Parameters
    ----------
    settings : dict
        Settings of the genetic algorithm
    parameters: dict
        Descriptions of the xgboost parameters
    data: dict
        Data sets for testing and training
    create_set : function
        Function used to generate a population
    evaluate : function
        Function used to calculate scores

    Returns
    -------
    result : dict
        Result of the run of the genetic algorithm
    '''
    population = create_population(settings, parameters, create_set)
    output = evolve(
        population, settings, parameters, data, create_set, evaluate)
    result = finalize_result(output, data)
    return result
