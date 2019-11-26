from __future__ import division
import numpy as np
import xgboost as xgb
from tthAnalysis.bdtHyperparameterOptimization import universal


def initialize_values(value_dicts):
    sample = {}
    for xgb_params in value_dicts:
        if bool(xgb_params['true_int']):
            sample[xgb_params['p_name']] = np.random.randint(
                low=xgb_params['range_start'],
                high=xgb_params['range_end']
            )
        else:
            sample[xgb_params['p_name']] = np.random.uniform(
                low=xgb_params['range_start'],
                high=xgb_params['range_end']
            )
    return sample


def prepare_run_params(nthread, value_dicts, sample_size):
    run_params = []
    for i in range(sample_size):
        run_param = initialize_values(value_dicts)
        run_params.append(run_param)
    return run_params


# currently not used?
def prepare_params_calc(value_dicts):
    keys_to_remove = [
        'verbosity',
        'objective',
        'num_class',
        'nthread',
        'seed'
    ]
    reduct_value_dicts = []
    try:
        z = value_dicts['num_boost_round']
        reduct_value_dict = value_dicts.copy()
        for key in keys_to_remove:
            try:
                reduct_value_dict.pop(key)
            except KeyError:
                pass
        return reduct_value_dict
    except TypeError:
        for value_dict in value_dicts:
            reduct_value_dict = value_dict.copy()
            for key in keys_to_remove:
                try:
                    reduct_value_dict.pop(key)
                except KeyError:
                    pass
            reduct_value_dicts.append(reduct_value_dict)
        return reduct_value_dicts


def parameter_evaluation(parameter_dict, data_dict, nthread, num_class):
    params = {
        'silent': 1,
        'objective': 'multi:softprob',
        'num_class': num_class,
        'nthread': nthread,
        'seed': 1,
    }
    parameters = parameter_dict.copy()
    num_boost_round = parameters.pop('num_boost_round')
    parameters.update(params)
    model = xgb.train(
        parameters,
        data_dict['dtrain'],
        num_boost_round=int(num_boost_round),
        verbose_eval=False
    )
    # print("Importances: ")
    # print(model.get_score(importance_type='gain'))
    pred_train = model.predict(data_dict['dtrain'])
    pred_test = model.predict(data_dict['dtest'])
    # score = calculate_fitness(pred_train, pred_test, data_dict)
    prob_train, prob_test = universal.get_most_probable(pred_train, pred_test)
    train_confusionMatrix, test_confusionMatrix = universal.calculate_conf_matrix(
        prob_train, prob_test, data_dict)
    score = universal.calculate_f1_score(test_confusionMatrix)[1]
    return score, pred_train, pred_test


# parameter evaluation as argument for the function. Move to universal
def ensemble_fitnesses(parameter_dicts, data_dict, global_settings):
    fitnesses = []
    pred_trains = []
    pred_tests = []
    for parameter_dict in parameter_dicts:
        fitness, pred_train, pred_test = parameter_evaluation(
            parameter_dict, data_dict,
            global_settings['nthread'], global_settings['num_classes'])
        fitnesses.append(fitness)
        pred_trains.append(pred_train)
        pred_tests.append(pred_test)
    return fitnesses, pred_trains, pred_tests
