'''
Testing the main functions of the genetic algorithm.
Missing tests for the following functions:
    sub_evolution
    evolve
    evolution
'''
import os
from tthAnalysis.bdtHyperparameterOptimization import ga_main as gm
from tthAnalysis.bdtHyperparameterOptimization import xgb_tools as xt
from tthAnalysis.bdtHyperparameterOptimization import mnist_filereader as mf

# parameters and settings for testing
PARAMETERS = [
    {
        'p_name': 'num_boost_round',
        'range_start': 1,
        'range_end': 500,
        'true_int': 1,
        'group_nr': 1,
        'true_corr': 0
    },
    {
        'p_name': 'learning_rate',
        'range_start': 0,
        'range_end': 0.3,
        'true_int': 0,
        'group_nr': 1,
        'true_corr': 0
    },
    {
        'p_name': 'max_depth',
        'range_start': 1,
        'range_end': 10,
        'true_int': 1,
        'group_nr': 2,
        'true_corr': 0
    },
    {
        'p_name': 'gamma',
        'range_start': 0,
        'range_end': 5,
        'true_int': 0,
        'group_nr': 2,
        'true_corr': 0
    }
]

SETTINGS = {
    'num_classes': 10,
    'pop_size': 3,
    'iterations': 1,
    'threshold': 0.001,
    'mut_chance': 0.03,
    'elites': 1,
    'culling': 1,
    'nthread': 8
}

POPULATION = [
    {
        'num_boost_round': 300,
        'learning_rate': 0.20323,
        'max_depth': 1,
        'gamma': 0.31544
    },
    {
        'num_boost_round': 55,
        'learning_rate': 0.07981,
        'max_depth': 8,
        'gamma': 4.12071
    },
    {
        'num_boost_round': 481,
        'learning_rate': 0.00411,
        'max_depth': 4,
        'gamma': 0.62212
    }
]

FITNESSES = [0.4, 0.6, 0.8]

# temporary solution for data during testing
DATA = mf.create_datasets(os.path.expandvars('$HOME/MNIST'), 8)

def test_set_num():
    '''Testing the set_num function'''
    initial = [1, 2, 3]
    nums = [0.5, 1, 2, 3]
    excpected = [2, 1, 2, 3]
    result = []
    for num in nums:
        result.append(gm.set_num(num, initial))
    assert result == excpected, 'test_set_num failed'


def test_elitism():
    '''Testing the elitism function'''
    initial = [1, 2, 3]
    nums = [0.5, 1, 2, 3]
    excpected = [[3, 2], [3], [3, 2], [3, 2, 1]]
    result = []
    for num in nums:
        result.append(gm.elitism(initial, FITNESSES, num))
    assert result == excpected, 'test_elitism failed'


def test_culling():
    '''Testing the culling function'''
    result = gm.culling(
        POPULATION,
        FITNESSES,
        SETTINGS,
        DATA,
        PARAMETERS,
        xt.prepare_run_params,
        xt.ensemble_fitnesses
    )
    assert len(result) == 2, 'test_culling failed'
    for element in result:
        assert len(element) == len(POPULATION), 'test_culling failed'


def test_new_population():
    '''Testing the new_population function'''
    result = gm.new_population(
        POPULATION, FITNESSES, SETTINGS, PARAMETERS)
    assert len(result) == len(POPULATION), \
        'test_new_population failed'


def test_create_subpopulations():
    '''Testing the create_subpopulations function'''
    nums = [1, 2, 3]
    excpected = [[1], [2, 1], [1, 1, 1]]
    i = 0
    for num in nums:
        j = 0
        SETTINGS.update({'sub_pops': num})
        result = gm.create_subpopulations(
            SETTINGS, PARAMETERS, xt.prepare_run_params)
        assert len(result) == len(excpected[i]), \
            'test_create_subpopulations failed'
        for element in result:
            assert len(element) == excpected[i][j], \
                'test_create_subpopulations failed'
            j += 1
        i += 1


# def test_sub_evolution():
#     '''Testing the sub_evolution function'''
#     result = gm.sub_evolution(
#         [POPULATION[0:2], POPULATION[2]],
#         SETTINGS,
#         DATA,
#         PARAMETERS,
#         xt.prepare_run_params,
#         xt.ensemble_fitnesses
#     )
#     assert len(result) == len(POPULATION)


# def test_evolve():
#     '''Testing the evolve function'''
#     result = gm.evolve(
#         POPULATION,
#         SETTINGS,
#         DATA,
#         PARAMETERS,
#         xt.prepare_run_params,
#         xt.ensemble_fitnesses,
#         True
#         )
#     assert len(result[0]) == len(POPULATION), 'test_evolve failed'
#     for key in result[1]:
#         assert len(result[1][key] == SETTINGS['iterations']), 'test_evolve failed'
#     assert len(result[2]) == len(result[0]), 'test_evolve failed'
