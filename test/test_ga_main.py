'''Testing the main functions of the genetic algorithm.'''
import os
import gzip
import shutil
import urllib
import pytest
from tthAnalysis.bdtHyperparameterOptimization import ga_main as gm
from tthAnalysis.bdtHyperparameterOptimization import xgb_tools as xt
from tthAnalysis.bdtHyperparameterOptimization import mnist_filereader as mf

dir_path = os.path.dirname(os.path.realpath(__file__))
resources_dir = os.path.join(dir_path, 'resources')
tmp_folder = os.path.join(resources_dir, 'tmp')
if not os.path.exists(tmp_folder):
    os.makedirs(tmp_folder)

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
    'sample_size': 3,
    'iterations': 1,
    'threshold': 0.001,
    'mut_chance': 0.03,
    'elites': 1,
    'culling': 1,
    'nthread': 28,
    'fitness_fn': 'test_auc'
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

# downloading data for testing
main_url = 'http://yann.lecun.com/exdb/mnist/'
train_images = 'train-images-idx3-ubyte'
train_labels = 'train-labels-idx1-ubyte'
test_images = 't10k-images-idx3-ubyte'
test_labels = 't10k-labels-idx1-ubyte'
file_list = [train_labels, train_images, test_labels, test_images]
sample_dir = os.path.join(tmp_folder, 'samples_mnist')
nthread = 2
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
for file in file_list:
    file_loc = os.path.join(sample_dir, file)
    file_url = os.path.join(main_url, file + '.gz')
    urllib.urlretrieve(file_url, file_loc + '.gz')
    with gzip.open(file_loc + '.gz', 'rb') as f_in:
        with open(file_loc, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
DATA = mf.create_datasets(sample_dir, 16)


def test_set_num():
    '''Testing the set_num function'''
    initial = [1, 2, 3]
    nums = [0.5, 1, 2, 3]
    expected = [2, 1, 2, 3]
    result = []
    for num in nums:
        result.append(gm.set_num(num, initial))
    assert result == expected, 'test_set_num failed'


def test_fitness_calculation():
    '''Testing the fitness calculation function'''
    results = gm.fitness_calculation(
        POPULATION, SETTINGS, DATA, xt.ensemble_fitnesses)
    for result in results:
        assert len(result) == len(POPULATION), 'test_fitness_calculation failed'


def test_elitism():
    '''Testing the elitism function'''
    initial = [1, 2, 3]
    nums = [0.5, 1, 2, 3]
    expected = [[3, 2], [3], [3, 2], [3, 2, 1]]
    result = []
    pop_data = {'fitnesses': FITNESSES}
    for num in nums:
        result.append(gm.elitism(initial, pop_data, num)[0])
    assert result == expected, 'test_elitism failed'


@pytest.mark.skip(reason='Runs too long')
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
    pop_data = {'fitnesses': FITNESSES}
    result = gm.new_population(
        POPULATION, pop_data, SETTINGS, PARAMETERS)[0]
    assert len(result) == len(POPULATION), \
        'test_new_population failed'


def test_create_subpopulations():
    '''Testing the create_subpopulations function'''
    nums = [1, 2, 3]
    expected = [[3], [2, 1], [1, 1, 1]]
    i = 0
    for num in nums:
        j = 0
        SETTINGS.update({'sub_pops': num})
        result = gm.create_subpopulations(
            SETTINGS, PARAMETERS, xt.prepare_run_params)
        assert len(result) == len(expected[i]), \
            'test_create_subpopulations failed'
        for element in result:
            assert len(element) == expected[i][j], \
                'test_create_subpopulations failed'
            j += 1
        i += 1


@pytest.mark.skip(reason='Runs too long')
def test_sub_evolution():
    '''Testing the sub_evolution function'''
    SETTINGS.update({'culling': 0})
    result = gm.sub_evolution(
        [POPULATION[:2], POPULATION[2:]],
        SETTINGS,
        DATA,
        PARAMETERS,
        xt.prepare_run_params,
        xt.ensemble_fitnesses
    )
    assert len(result[0]) == len(POPULATION), 'test_sub_evolution failed'
    for key in result[1]:
        for i in result[1][key]:
            assert len(result[1][key][i]) == len(result[2][i]), \
                'test_sub_evolution failed'


@pytest.mark.skip(reason='Runs too long')
def test_evolve():
    '''Testing the evolve function'''
    initial = POPULATION[:1]
    SETTINGS.update({'culling': 0})
    result = gm.evolve(
        initial,
        SETTINGS,
        DATA,
        PARAMETERS,
        xt.prepare_run_params,
        xt.ensemble_fitnesses
        )
    assert len(result['population']) == len(initial), 'test_evolve failed'
    for key in result['scores']:
        assert len(result['scores'][key]) == len(result['compactnesses']), \
            'test_evolve failed'
    assert len(result['fitnesses']) == len(result['population']), \
        'test_evolve failed'


def test_score_tracker():
    '''Testing the score tracker function'''
    initial = [
        {
            'g_score': 0.984,
            'f1_score': 0.947,
            'd_score': 0.701,
            'test_auc': 0.690,
            'train_auc': 0.115
        },
        {
            'g_score': 0.582,
            'f1_score': 0.651,
            'd_score': 0.123,
            'test_auc': 0.476,
            'train_auc': 0.540
        },
        {
            'g_score': 0.869,
            'f1_score': 0.240,
            'd_score': 0.851,
            'test_auc': 0.646,
            'train_auc': 0.752
        }
    ]
    result = {}
    result = gm.score_tracker(result, initial, FITNESSES, True)
    expected = {
        'best_g_scores': [0.869],
        'best_f1_scores': [0.240],
        'best_d_scores': [0.851],
        'best_test_aucs': [0.646],
        'best_train_aucs': [0.752],
        'avg_scores': [0.6],
        'best_fitnesses': [0.8]
    }
    assert result == expected, 'test_score_tracker failed'


def test_finalize_results():
    '''Testing the finalize results function'''
    initial = {
        'population': POPULATION,
        'scores': {
            'best_g_scores': [0.869],
            'best_f1_scores': [0.240],
            'best_d_scores': [0.851],
            'best_test_aucs': [0.646],
            'best_train_aucs': [0.752],
            'avg_scores': [0.6],
            'best_fitnesses': [0.8]
        },
        'fitnesses': FITNESSES,
        'compactnesses': [0.953],
        'pred_trains': [[0.990], [0.993], [0.503]],
        'pred_tests': [[0.609], [0.529], [0.882]],
        'feature_importances': [[0.041], [0.269], [0.996]]
    }
    result = gm.finalize_results(initial, DATA)
    expected = {
        'best_parameters': POPULATION[2],
        'best_fitnesses': [0.8],
        'avg_scores': [0.6],
        'compactnesses': [0.953],
        'pred_train': [0.503],
        'pred_test': [0.882],
        'feature_importances': [0.996],
        'data_dict': DATA,
        'best_g_scores': [0.869],
        'best_f1_scores': [0.240],
        'best_d_scores': [0.851],
        'best_test_aucs': [0.646],
        'best_train_aucs': [0.752]
    }
    assert result == expected


@pytest.mark.skip(reason='Runs too long')
def test_evolution():
    '''Testing the evolution function'''
    SETTINGS.update({'culling': 0, 'elitism': 0, 'sub_pops': 1})
    result = gm.evolution(
        SETTINGS,
        DATA,
        PARAMETERS,
        xt.prepare_run_params,
        xt.ensemble_fitnesses
    )
    assert len(result['best_parameters']) == len(PARAMETERS), \
        'test_evolution failed'
    assert len(result['avg_scores']) == len(result['best_fitnesses']), \
        'test_evolution failed'
    assert len(result['compactnesses']) == len(result['avg_scores']), \
        'test_evolution failed'
    assert result['data_dict'] == DATA, \
        'test_evolution failed'


def test_dummy_delete_files():
    '''Delete temporary files'''
    if os.path.exists(resources_dir):
        shutil.rmtree(resources_dir)
