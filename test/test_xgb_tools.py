from tthAnalysis.bdtHyperparameterOptimization.xgb_tools import prepare_run_params
from tthAnalysis.bdtHyperparameterOptimization.xgb_tools import initialize_values
from tthAnalysis.bdtHyperparameterOptimization.xgb_tools import prepare_params_calc
from __future__ import division



def test_prepare_params_calc():
    values = {
        'num_boost_round': 371,
        'learning_rate': 0.07,
        'max_depth': 9,
        'gamma': 1.9,
        'min_child_weight': 18,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'verbosity': 1,
        'objective': 'multi:softprob',
        'num_class': 10,
        'nthread': 2,
        'seed': 1
    }
    expected = {
        'num_boost_round': 371,
        'learning_rate': 0.07,
        'max_depth': 9,
        'gamma': 1.9,
        'min_child_weight': 18,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
    }
    result = prepare_params_calc(values)
    assert result == expected


def test_prepare_params_calc2():
    values = {
        'num_boost_round': 371,
        'learning_rate': 0.07,
        'max_depth': 9,
        'gamma': 1.9,
        'min_child_weight': 18,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
        'verbosity': 1,
        'objective': 'multi:softprob',
        'num_class': 10,
        'nthread': 2,
        'seed': 1
    }
    values_list = [
        values,
        values,
        values
    ]
    expected = {
        'num_boost_round': 371,
        'learning_rate': 0.07,
        'max_depth': 9,
        'gamma': 1.9,
        'min_child_weight': 18,
        'subsample': 0.9,
        'colsample_bytree': 0.8,
    }
    expected_list = [
        expected,
        expected,
        expected
    ]
    result = prepare_params_calc(values_list)
    assert result == expected_list


def test_initialize_values():
    value_dict1 = {
        'p_name': 'test1',
        'range_start': 0,
        'range_end': 10,
        'true_int': 'True'
    }
    value_dict2 = {
        'p_name': 'test2',
        'range_start': 0,
        'range_end': 10,
        'true_int': 'False'
    }
    value_dicts = [value_dict1, value_dict2]
    result = initialize_values(value_dicts)
    assert result['test2'] >= 0 and result['test2'] <= 10
    assert isinstance(result['test1'], int)


def test_prepare_run_params():
    nthread = 28
    value_dict1 = {
        'p_name': 'test1',
        'range_start': 0,
        'range_end': 10,
        'true_int': 'True'
    }
    value_dict2 = {
        'p_name': 'test2',
        'range_start': 0,
        'range_end': 10,
        'true_int': 'False'
    }
    value_dicts = [value_dict1, value_dict2]
    sample_size = 3
    result = prepare_run_params(
        nthread,
        value_dicts,
        sample_size
    )
    sum = 0
    for i in result:
        if isinstance(i['test1'], int):
            sum +=1
    assert len(result) == 3
    assert sum == 3