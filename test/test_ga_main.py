"""
Testing the main functions of the genetic algorithm.
Missing tests for the following functions:
    culling
    sub_evolution
    evolve
    evolution
"""
from __future__ import division
import random
from tthAnalysis.bdtHyperparameterOptimization import ga_main as gm


# parameters and settings for testing
PARAMETERS = [
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

SETTINGS = {
    'iterations': 2,
    'threshold': 0.002,
    'mut_chance': 0.1,
    'elites': 2,
    'nthread': 8
}

SAMPLE = [
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
        'num_class': 10,
        'nthread': 8,
        'seed': 1
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

SIMPLE = [1, 2, 3]
FITNESSES = [0.4, 0.6, 0.8]
NUMS = [0.5, 1, 2, 3]


def grouped_sample():
    """Temporary function for grouping the sample population"""
    grouped = []
    pos_corr = []
    for element in SAMPLE:
        grouped.append(gm.grouping(element, PARAMETERS)[0])
        pos_corr.append(gm.grouping(element, PARAMETERS)[1])
    return grouped, pos_corr


def test_crossover():
    """Testing the crossover function"""
    parents = SAMPLE[0:2]
    calculated = gm.add_parameters(
        gm.crossover(
            parents,
            SETTINGS['mut_chance'],
            PARAMETERS),
        SETTINGS['nthread'])
    assert len(calculated) == len(parents[0]), 'test_crossover failed'


def test_grouping():
    """Testing the grouping function"""
    for dictionary in SAMPLE:
        calculated = gm.grouping(dictionary, PARAMETERS)
        for element in calculated:
            assert len(element) == 5, 'test_grouping failed'
        for element in calculated[0]:
            for key in element.keys():
                assert element[key] == dictionary[key], 'test_grouping failed'
        for element in calculated[1]:
            i = 0
            try:
                if element != PARAMETERS[i]['true_corr']:
                    i += 1
                else:
                    assert element == PARAMETERS[i]['true_corr'], \
                        'test_grouping failed'
                    i += 1
            except:
                raise AssertionError('test_grouping failed')


def test_degroup():
    """Testing the degroup function"""
    grouped = grouped_sample()[0]
    calculated = []
    for element in grouped:
        calculated.append(gm.add_parameters(
            gm.degroup(element), SETTINGS['nthread']))
    assert calculated == SAMPLE, 'test_degroup failed'


def test_mutate():
    """Testing the mutate function"""
    grouped, pos_corr = grouped_sample()
    for i, element in enumerate(grouped):
        total_calc = []
        for j, group in enumerate(element):
            calculated = gm.mutate(
                group, SETTINGS['mut_chance'], random.random(), pos_corr[i][j])
            assert len(calculated) == len(group), 'test_mutate failed'
            total_calc.append(calculated)
        assert len(total_calc) == len(element), 'test_mutate failed'


def test_set_num():
    """Testing the set_num function"""
    result = [2, 1, 2, 3]
    calculated = []
    for num in NUMS:
        calculated.append(gm.set_num(num, SIMPLE))
    assert result == calculated, 'test_set_num failed'


def test_elitism():
    """Testing the elitism function"""
    result = [[3, 2], [3], [3, 2], [3, 2, 1]]
    calculated = []
    for num in NUMS:
        calculated.append(gm.elitism(SIMPLE, FITNESSES, num))
    assert result == calculated, 'test_elitism failed'


def test_add_parameters():
    """Testing the add_parameters function"""
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
    """Testing the new_population function"""
    SETTINGS.update({'pop_size': 3})
    calculated = gm.new_population(
        SAMPLE, FITNESSES, SETTINGS, PARAMETERS)
    assert len(calculated) == len(SAMPLE), \
        'test_new_population failed'


def test_create_subpopulations():
    """Testing the create_subpopulations function"""
    nums = NUMS[1:]
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
            SETTINGS.update({'pop_size': size, 'sub_pops': num})
            calculated = gm.create_subpopulations(SETTINGS, PARAMETERS)
            assert len(calculated) == len(result[i]), \
                'test_create_subpopulations failed'
            for element in calculated:
                assert len(element) == result[i][j], \
                    'test_create_subpopulations failed'
                j += 1
            i += 1
