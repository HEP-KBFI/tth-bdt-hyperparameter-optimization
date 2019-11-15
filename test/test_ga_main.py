from __future__ import division
import numpy as np
from tthAnalysis.bdtHyperparameterOptimization import ga_main as gm
import random


# parameters and settings for testing
parameters = [
    {
        'p_name': 'num_boost_round',
        'range_start': 1,
        'range_end': 500,
        'true_int': 'True',
        'group_nr': 1,
        'true_corr': 'False'
    },
    {
        'p_name': 'learning_rate',
        'range_start': 0,
        'range_end': 0.3,
        'true_int': 'False',
        'group_nr': 1,
        'true_corr': 'False'
    },
    {
        'p_name': 'max_depth',
        'range_start': 1,
        'range_end': 10,
        'true_int': 'True',
        'group_nr': 2,
        'true_corr': 'False'
    },
    {
        'p_name': 'gamma',
        'range_start': 0,
        'range_end': 5,
        'true_int': 'False',
        'group_nr': 2,
        'true_corr': 'False'
    },
    {
        'p_name': 'min_child_weight',
        'range_start': 0,
        'range_end': 500,
        'true_int': 'False',
        'group_nr': 3,
        'true_corr': 'True'
    },
    {
        'p_name': 'subsample',
        'range_start': 0.8,
        'range_end': 1,
        'true_int': 'False',
        'group_nr': 4,
        'true_corr': 'True'
    },
    {
        'p_name': 'colsample_bytree',
        'range_start': 0.3,
        'range_end': 1,
        'true_int': 'False',
        'group_nr': 5,
        'true_corr': 'True'
    }
]

settings = {
    "iterations": 2, 
    "threshold": 0.002, 
    "mut_chance": 0.1, 
    "elites": 2, 
    "nthread": 8
}

sample_population = [
    {
        'num_boost_round': 38, 
        'learning_rate': 0.29915544328166055, 
        'max_depth': 9, 
        'gamma': 0.0005718740867244332, 
        'min_child_weight': 151.16628631591988, 
        'subsample': 0.8293511781634226, 
        'colsample_bytree': 0.3646370163381584, 
        'silent': 1, 
        'objective': 'multi:softprob', 
        'num_class': 10, 'nthread': 8, 'seed': 1
    }, 
    {
        'num_boost_round': 461, 
        'learning_rate': 0.11637322234860221, 
        'max_depth': 7, 
        'gamma': 4.677695354030159, 
        'min_child_weight': 423.15545834300855, 
        'subsample': 0.8626547033864551, 
        'colsample_bytree': 0.66718371170101, 
        'silent': 1, 
        'objective': 'multi:softprob', 
        'num_class': 10, 
        'nthread': 8, 
        'seed': 1
    }, 
    {
        'num_boost_round': 253, 
        'learning_rate': 0.2634352309172836, 
        'max_depth': 3, 
        'gamma': 4.569810122896165, 
        'min_child_weight': 228.60240399349414, 
        'subsample': 0.8861397134369418, 
        'colsample_bytree': 0.9573894525965381, 
        'silent': 1, 
        'objective': 'multi:softprob', 
        'num_class': 10, 
        'nthread': 8, 
        'seed': 1
    }
]

simple_population = [1, 2, 3]
fitnesses = [0.4, 0.6, 0.8]
nums = [0.5, 1, 2, 3]

# helper function for testing
def grouped_sample_population():
    grouped = []
    pos_corr = []
    for element in sample_population:
        grouped.append(gm.grouping(element, parameters)[0])
        pos_corr.append(gm.grouping(element, parameters)[1])
    return grouped, pos_corr


# TESTS
# def test_selection():
#     calculated = gm.selection(simple_population, fitnesses)
#     assert len(calculated) == 2, 'test_selection failed'
#     for parent in calculated:
#         assert parent in simple_population, 'test_selection failed'


def test_crossover():
    parents = sample_population[0:2]
    calculated = gm.add_parameters(
        gm.crossover(parents, settings['mut_chance'], parameters), settings['nthread'])
    assert len(calculated) == len(parents[0]), 'test_crossover failed'


def test_grouping():
    for dictionary in sample_population:
        calculated = gm.grouping(dictionary, parameters)
        for element in calculated:
            assert len(element) == 5, 'test_grouping failed'
        for element in calculated[0]:
            for key in element.keys():
                assert element[key] == dictionary[key], 'test_grouping failed'
        for element in calculated[1]:
            i = 0
            try:
                if element != parameters[i]['true_corr']:
                    i += 1
                else:
                    assert element == parameters[i]['true_corr'], 'test_grouping failed'
                    i += 1
            except:
                raise AssertionError('test_grouping failed')


def test_degroup():
    grouped, pos_corr = grouped_sample_population()
    calculated = []
    for element in grouped:
        calculated.append(gm.add_parameters(
            gm.degroup(element), settings['nthread']))
    assert calculated == sample_population, 'test_degroup failed'


def test_mutate():
    grouped, pos_corr = grouped_sample_population()
    for i, element in enumerate(grouped):
        total_calc = []
        for j, group in enumerate(element):
            calculated = gm.mutate(group, settings['mut_chance'], random.random(), pos_corr[i][j])
            assert len(calculated) == len(group), 'test_mutate failed'
            total_calc.append(calculated)
        assert len(total_calc) == len(element), 'test_mutate failed'


def test_set_num():
    result = [2, 1, 2, 3]
    calculated = []
    for num in nums:
        calculated.append(gm.set_num(num, simple_population))
    assert result == calculated, 'test_set_num failed'


def test_elitism():
    result = [[3, 2], [3], [3, 2], [3, 2, 1]]
    calculated = []
    for num in nums:
        calculated.append(gm.elitism(simple_population, fitnesses, num))
    assert result == calculated, 'test_elitism failed'


# def test_culling():
#     result = [1, 2, 1, 0]
#     for i, num in enumerate(nums):
#         settings.update({'culling':num})
#         calculated = gm.culling(sample_population, fitnesses, settings, parameters)
#         assert len(calculated[0]) == len(sample_population), 'test_culling failed'
#         assert len(calculated[1]) == len(sample_population), 'test_culling failed'
#         counter = 0
#         for member in calculated[0]:
#             if member in sample_population:
#                 counter += 1
#         assert counter == result[i], 'test_culling failed'


def test_add_parameters():
    dictionary = {'test': 1}
    nthread = 8
    calculated = gm.add_parameters(dictionary, nthread)
    result = {
        'test': 1,
        'silent': 1,
        'objective': 'multi:softprob',
        'num_class': 10,
        'nthread': nthread,
        'seed': 1,
    }
    assert result == calculated, 'test_add_parameters failed'


def test_new_population():
    settings.update({'pop_size':3})
    calculated = gm.new_population(sample_population, fitnesses, settings, parameters)
    assert len(calculated) == len(sample_population), 'test_new_population failed'


def test_create_subpopulations():
    nums = [1, 2, 3]
    sizes = [1, 2, 4, 7, 11]
    result = [
        [1], [], [],
        [2], [1, 1], [],
        [4], [2, 2], [2, 1, 1],
        [7], [4, 3], [3, 2, 2],
        [11], [6, 5], [5, 3, 3]
    ]
    i = 0
    for size in sizes:
        for num in nums:
            j = 0
            settings.update({'pop_size':size, 'sub_pops':num})
            calculated = gm.create_subpopulations(settings, parameters)
            assert len(calculated) == len(result[i]), "test_create_subpopulations failed"
            for element in calculated:
                assert len(element) == result[i][j], "test_create_subpopulations failed"
                j += 1
            i += 1
