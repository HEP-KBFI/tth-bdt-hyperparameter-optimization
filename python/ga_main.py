'''Main functions for the genetic algorithm'''
from __future__ import division
import inspect
import numpy as np
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import ga_selection as select
from tthAnalysis.bdtHyperparameterOptimization import ga_crossover as gc


# KEYS = ['g_score', 'f1_score', 'd_score', 'test_auc', 'train_auc']
KEYS = ['d_ams']


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
    fitness)
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
            self,
            fitness,
            score=None,
            pred_train=None,
            pred_test=None,
            feature_importance=None):
        '''Adds the the evaluation results as attributes to the
        individual'''
        self.fitness = fitness
        self.score = score
        self.pred_train = pred_train
        self.pred_test = pred_test
        self.feature_importance = feature_importance

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
    separated : list
        List of separated subpopulations
    '''
    subpopulations = []
    separated = []
    # Create empty subpopulations
    for i in range(settings['sub_pops']):
        subpopulations.append([])
    # Add individuals to correct subpopulations
    for member in population:
        index = member.subpop
        subpopulations[index].append(member)
    # Select subpopulations that are not empty
    for subpopulation in subpopulations:
        if subpopulation:
            separated.append(subpopulation)
    return separated


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


def rosenbrock_fitnesses(fitnesses):
    '''Turn rosenbrock fitness score into an appropriate score for genetic algorithm

    Parameters
    ----------
    fitnesses : list
        List of fitness scores for the population

    Returns
    -------
    new_fitnesses : list
        List of appropriate fitness scores for the population
    '''
    new_fitnesses = []
    for fitness in fitnesses:
        fitness = np.log10(fitness) / 38
        new_fitnesses.append(1 - fitness)
    return new_fitnesses


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
    # Separate population for time efficiency
    eval_pop, rest_pop = arrange_population(population)
    # Evaluate unevaluated members of population
    if eval_pop:
        args = inspect.getargspec(evaluate)
        if len(args[0]) == 2: #rosenbrock function optimization
            fitnesses = evaluate(population_list(eval_pop), data)
            fitnesses = rosenbrock_fitnesses(fitnesses)
            for i, member in enumerate(eval_pop):
                member.add_result(fitnesses[i])
        else: #hyperparameter optimization
            if len(args[0]) == 3:
                scores, pred_trains, pred_tests, feature_importances = evaluate(
                    population_list(eval_pop), data, settings)
            elif len(args[0]) == 4:
                scores, pred_trains, pred_tests, feature_importances = evaluate(
                    population_list(eval_pop), data, settings, len(eval_pop))
            fitnesses = universal.fitness_to_list(
                scores, fitness_key=settings['fitness_fn'])
            # Assign scores to the individuals
            for i, member in enumerate(eval_pop):
                member.add_result(
                    fitnesses[i],
                    scores[i],
                    pred_trains[i],
                    pred_tests[i],
                    feature_importances[i],
                )
        # Reunite population
        population = eval_pop + rest_pop
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


def new_population(
        population,
        settings,
        parameters,
        data,
        create_set,
        evaluate,
        subpop=0
    ):
    '''Create the next generation population.

    Parameters
    ----------
    population : list
        Current set of individuals
    settings : dict
        Settings of the genetic algorithm
    parameters: dict
        Descriptions of the xgboost parameters
    data : dict
        Training and testing data sets
    create_set : function
        Function used to generate a population
    evaluate : function
        Function used to calculate scores
    subpop : int (optional)
        Which subpopulation is being updated

    Returns
    -------
    next_population : list
        Newly generated set of individuals
    '''
    # Cull worst members of previous population
    population = culling(
        population, settings, parameters, data, create_set, evaluate, subpop)
    # Add best members of previous population into the new population
    offsprings = []
    next_population = elitism(population, settings)
    fitnesses = fitness_list(population)
    # Generate offspring to fill the new generation
    while len(offsprings) < (len(population) - len(next_population)):
        parents = select.tournament(population_list(population), fitnesses)
        offspring = gc.kpoint_crossover(
            parents, parameters, settings['mut_chance'])
        # No duplicate members
        if offspring not in next_population:
            offsprings.append(offspring)
    next_population += assign_individuals(offsprings, subpop)
    return next_population


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


def finish_subpopulation(subpopulations, finished_subpopulations, improvements, threshold):
    '''Separate out a subpopulation that has reached the improvement threshold

    Parameters
    ----------
    subpopulations : list
        List of all current subpopulations
    finished_subpopulations : list
        List of already finished subpopulations
    improvements : list
        Improvement scores for the current subpopulations
    threshold : float
        Threshold value for the improvement score

    Returns
    -------
    finished_subpopulations : list
        Updated list of finished subpopulations
    remaining_subpopulations : list
        Subpopulations to continue evolving
    '''
    remaining_subpopulations = []
    for i, improvement in enumerate(improvements):
        if improvement <= threshold:
            finished_subpopulations.append(subpopulations[i])
        else:
            remaining_subpopulations.append(subpopulations[i])
    return finished_subpopulations, remaining_subpopulations


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
        finished_subpopulations = []
        while (iteration <= settings['iterations']
               and population):
            # Generate a new population
            if iteration != 0:
                print('::::: Iteration:     ' + str(iteration) + ' :::::')
                new_subpopulations = []
                for subpopulation in subpopulations:
                    new_subpopulation = new_population(
                        subpopulation,
                        settings,
                        parameters,
                        data,
                        create_set,
                        evaluate,
                        subpopulation[0].subpop
                    )
                    new_subpopulations.append(new_subpopulation)
                population = unite_subpopulations(new_subpopulations)
            # Calculate fitness scores
            population = fitness_calculation(
                population, settings, data, evaluate)
            subpopulations = separate_subpopulations(population, settings)
            # Track scores and calculate stopping criteria
            curr_improvements = []
            for i, subpopulation in enumerate(subpopulations):
                index = subpopulation[0].subpop
                if iteration == 0:
                    tracker[index] = {}
                    improvements[index] = []
                    compactnesses[index] = []
                    tracker[index] = score_tracker(
                        tracker[index], subpopulation, True)
                else:
                    tracker[index] = score_tracker(tracker[index], subpopulation)
                improvements[index], curr_improvement = \
                    universal.calculate_improvement_wAVG(
                        tracker[index]['avg_scores'],
                        improvements[index],
                        settings['threshold']
                    )
                curr_improvements.append(curr_improvement)
                compactness = universal.calculate_compactness(
                    population_list(subpopulation))
                compactnesses[index].append(compactness)
            # Remove a subpopulation that has reached a stopping
            # criterium
            finished_subpopulations, subpopulations = finish_subpopulation(
                subpopulations, finished_subpopulations, curr_improvements, settings['threshold'])
            population = unite_subpopulations(subpopulations)
            iteration += 1
        # Merge subpopulations into one
        print('::::: Merging subpopulations :::::')
        subpopulations += finished_subpopulations
        population = merge_subpopulations(subpopulations)
        iteration = 0

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
            population = new_population(
                population, settings, parameters, data, create_set, evaluate)
        # Calculate fitness scores
        population = fitness_calculation(
            population, settings, data, evaluate)
        # Track scores and calculate stopping criteria
        if iteration == 0:
            final_tracker = score_tracker(final_tracker, population, True)
        else:
            final_tracker = score_tracker(final_tracker, population)
        final_improvements, improvement = \
            universal.calculate_improvement_wAVG(
                final_tracker['avg_scores'],
                final_improvements,
                settings['threshold']
            )
        compactness = universal.calculate_compactness(
            population_list(population))
        final_compactnesses.append(compactness)
        iteration += 1

    tracker.update({'final': final_tracker})
    if compactnesses:
        compactnesses.update({'final': final_compactnesses})
    else:
        compactnesses = final_compactnesses

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
    fitnesses = fitness_list(population)
    index = np.argmax(fitnesses)
    for key in KEYS:
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
        result = []
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
    index = np.argmax(output['fitnesses'])
    result = {
        'best_fitness': max(output['fitnesses']),
        'best_parameters': population_list(output['population'])[index],
        'best_fitnesses': find_result(output['scores'], 'best_fitnesses'),
        'avg_scores': find_result(output['scores'], 'avg_scores'),
        'compactnesses': output['compactnesses'],
        'pred_train': output['population'][index].pred_train,
        'pred_test': output['population'][index].pred_test,
        'feature_importances':
            output['population'][index].feature_importance,
        'data_dict': data
    }
    for key in KEYS:
        key_name = 'best_' + key
        result[key_name] = output['population'][index].score[key]
        list_key = key_name + 's'
        result[list_key] = find_result(output['scores'], list_key)
    return result


def evolution(settings, parameters, data, create_set, evaluate):
    '''Evolution of the parameter values for hyperparameter optimization

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
    print('\n::::: Generating initial population :::::')
    population = create_population(settings, parameters, create_set)
    output = evolve(
        population, settings, parameters, data, create_set, evaluate)
    print('\n::::: Finalizing results :::::')
    result = finalize_result(output, data)
    return result


def evolution_rosenbrock(settings, parameters, data, create_set, evaluate):
    '''Evolution of the parameter values for the Rosenbrock function

    Parameters
    ----------
    settings : dict
        Settings of the genetic algorithm
    parameters: dict
        Descriptions of the xgboost parameters
    data: dict
        Parameter values for the Rosenbrock function
    create_set : function
        Function used to generate a population
    evaluate : function
        Function used to calculate scores

    Returns
    -------
    result : dict
        Result of the run of the genetic algorithm
    '''
    print('\n::::: Generating initial population :::::')
    population = create_population(settings, parameters, create_set)

    iteration = 0
    curr_improvement = []
    improvements = {}
    avg_scores = {}

    result = {}
    subpopulation_iterations = int(np.ceil(0.9 * settings['iterations']))
    merged_iterations = settings['iterations'] - subpopulation_iterations

    # Evolution loop for subpopulations
    if settings['sub_pops'] > 1:
        finished_subpopulations = []
        while (iteration <= subpopulation_iterations
               and population):
            # Generate a new population
            if iteration != 0:
                print('::::: Iteration:     ' + str(iteration) + ' :::::')
                new_subpopulations = []
                for subpopulation in subpopulations:
                    new_subpopulation = new_population(
                        subpopulation,
                        settings,
                        parameters,
                        data,
                        create_set,
                        evaluate,
                        subpopulation[0].subpop
                    )
                    new_subpopulations.append(new_subpopulation)
                population = unite_subpopulations(new_subpopulations)
            # Calculate fitness scores
            population = fitness_calculation(
                population, settings, data, evaluate)
            subpopulations = separate_subpopulations(population, settings)
            # Track results and calculate stopping criteria
            curr_improvements = []
            for i, subpopulation in enumerate(subpopulations):
                index = subpopulation[0].subpop
                fitnesses = fitness_list(subpopulation)
                if iteration == 0:
                    avg_scores[index] = []
                    improvements[index] = []
                    result['best_fitness'] = max(fitnesses)
                    result['best_parameters'] = population_list(subpopulation)[np.argmax(fitnesses)]
                    result['list_of_old_bests'] = [result['best_parameters']]
                    result['list_of_best_fitnesses'] = [result['best_fitness']]
                avg_scores[index].append(np.mean(fitnesses))
                improvements[index], curr_improvement = \
                    universal.calculate_improvement_wAVG(
                        avg_scores[index],
                        improvements[index],
                        settings['threshold']
                    )
                curr_improvements.append(curr_improvement)
                if max(fitnesses) > result['best_fitness']:
                    result['best_fitness'] = max(fitnesses)
                    result['best_parameters'] = population_list(subpopulation)[np.argmax(fitnesses)]
                    result['list_of_old_bests'].append(result['best_parameters'])
                    result['list_of_best_fitnesses'].append(result['best_fitness'])
            # Remove a subpopulation that has reached a stopping
            # criterium
            # finished_subpopulations, subpopulations = finish_subpopulation(
            #     subpopulations, finished_subpopulations, curr_improvements, settings['threshold'])
            population = unite_subpopulations(subpopulations)
            iteration += 1
        # Merge subpopulations into one
        print('::::: Merging subpopulations :::::')
        # subpopulations += finished_subpopulations
        population = merge_subpopulations(subpopulations)

    iteration = 0
    improvement = 1
    improvements = []
    avg_scores = []

    # Evolution loop for single population or merged population
    while (iteration <= merge_subpopulations):
           # and improvement > settings['threshold']):
        # Generate new population
        if iteration != 0:
            print('::::: Iteration:     ' + str(iteration) + ' :::::')
            population = new_population(
                population, settings, parameters, data, create_set, evaluate)
        # Calculate fitness scores
        population = fitness_calculation(
            population, settings, data, evaluate)
        # Track scores and calculate stopping criteria
        fitnesses = fitness_list(population)
        index = np.argmax(fitnesses)
        if iteration == 0:
            avg_scores = []
            improvements = []
            result['best_fitness'] = max(fitnesses)
            result['best_parameters'] = population_list(population)[np.argmax(fitnesses)]
            result['list_of_old_bests'] = [result['best_parameters']]
            result['list_of_best_fitnesses'] = [result['best_fitness']]
        avg_scores.append(np.mean(fitnesses))
        improvements, improvement = \
            universal.calculate_improvement_wAVG(
                avg_scores,
                improvements,
                settings['threshold']
            )
        if max(fitnesses) > result['best_fitness']:
            result['best_fitness'] = max(fitnesses)
            result['best_parameters'] = population_list(population)[np.argmax(fitnesses)]
            result['list_of_old_bests'].append(result['best_parameters'])
            result['list_of_best_fitnesses'].append(result['best_fitness'])
        iteration += 1

    return result
