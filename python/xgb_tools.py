'''Tools necessary for XGBoost based fitness calculation and model creation
'''
from __future__ import division
import numpy as np
import xgboost as xgb
from tthAnalysis.bdtHyperparameterOptimization import universal


def initialize_values(value_dicts):
    '''Initializes the parameters according to the value dict specifications

    Parameters:
    ----------
    value_dicts : list of dicts
        Specifications how each value should be initialized

    Returns:
    -------
    sample : list of dicts
        Parameter-set for a particle
    '''
    sample = {}
    for xgb_params in value_dicts:
        if bool(xgb_params['true_int']):
            sample[str(xgb_params['p_name'])] = np.random.randint(
                low=xgb_params['range_start'],
                high=xgb_params['range_end']
            )
        else:
            sample[str(xgb_params['p_name'])] = np.random.uniform(
                low=xgb_params['range_start'],
                high=xgb_params['range_end']
            )
    return sample


def prepare_run_params(value_dicts, sample_size):
    ''' Creates parameter-sets for all particles (sample_size)

    Parameters:
    ----------
    value_dicts : list of dicts
        Specifications how each value should be initialized
    sample_size : int
        Number of particles to be created

    Returns:
    -------
    run_params : list of dicts
        List of parameter-sets for all particles
    '''
    run_params = []
    for i in range(sample_size):
        run_param = initialize_values(value_dicts)
        run_params.append(run_param)
    return run_params


def parameter_evaluation(parameter_dict, data_dict, nthread, num_class):
    '''Evaluates the parameter_dict for fitness

    Parameters:
    ----------
    parameter_dict :
        Parameter-set to be evaluated
    data_dict : dict
        Dictionary that contains the labels for testing and training. Keys are
        called 'testing_labels' and 'training_labels'
    nthread : int
        Number of threads to be used in the xgboost training
    num_class : int
        Number of classes the event can belong to

    Returns:
    -------
    score : float
        Fitness of the parameter-set
    pred_train : list
        List of numpy arrays containing probabilities for all labels
        for the training sample
    pred_test : list
        List of numpy arrays containing probabilities for all labels
        for the testing sample
    '''
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
    pred_train = model.predict(data_dict['dtrain'])
    pred_test = model.predict(data_dict['dtest'])
    prob_train, prob_test = universal.get_most_probable(pred_train, pred_test)
    train_conf_matrix, test_conf_matrix = universal.calculate_conf_matrix(
        prob_train, prob_test, data_dict)
    g_score_test, f1_score_test = universal.calculate_f1_score(
        test_conf_matrix)
    g_score_train, f1_score_train = universal.calculate_f1_score(
        train_conf_matrix)
    d_score = calculate_d_score(pred_train, pred_test, data_dict)
    train_auc, test_auc = universal.calculate_auc(
        data_dict, pred_train, pred_test)[:2]
    score_dict = {
        'f1_score_test': f1_score_test,
        'g_score_test': g_score_test,
        'test_auc': test_auc,
        'f1_score_train': f1_score_train,
        'g_score_train': g_score_train,
        'train_auc': train_auc,
        'd_score': d_score
    }
    return score_dict, pred_train, pred_test


# parameter evaluation as argument for the function. Move to universal
def ensemble_fitnesses(parameter_dicts, data_dict, global_settings):
    '''Finds the data_dict, pred_train and pred_test for all particles

    Parameters:
    ----------
    parameter_dicts : list of dicts
        Parameter-sets of all particles
    data_dict : dict
        Dictionary that contains the labels for testing and training. Keys are
        called 'testing_labels' and 'training_labels'
    global_settings : dict
        Global settings for the run.

    Returns:
    -------
    score_dicts : list
        List of score_dicts of each parameter-set
    pred_trains : list of lols
        List of pred_trains
    pred_tests : list of lols
        List of pred_tests
    '''
    score_dicts = []
    pred_trains = []
    pred_tests = []
    for parameter_dict in parameter_dicts:
        score_dict, pred_train, pred_test = parameter_evaluation(
            parameter_dict, data_dict,
            global_settings['nthread'], global_settings['num_classes'])
        score_dicts.append(score_dict)
        pred_trains.append(pred_train)
        pred_tests.append(pred_test)
    return score_dicts, pred_trains, pred_tests
