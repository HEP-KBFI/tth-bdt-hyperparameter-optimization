from __future__ import division
from tthAnalysis.bdtHyperparameterOptimization import xgb_tools as xt


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
    result = xt.initialize_values(value_dicts)
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
    result = xt.prepare_run_params(
        value_dicts,
        sample_size
    )
    sum = 0
    for i in result:
        if isinstance(i['test1'], int):
            sum +=1
    assert len(result) == 3
    assert sum == 3