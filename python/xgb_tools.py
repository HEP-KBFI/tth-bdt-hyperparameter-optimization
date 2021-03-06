'''Tools necessary for XGBoost based fitness calculation and model creation
'''
import numpy as np
import xgboost as xgb
from tthAnalysis.bdtHyperparameterOptimization import universal
import os


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
             value = np.random.randint(
                low=xgb_params['range_start'],
                high=xgb_params['range_end']
            )
        else:
            value = np.random.uniform(
                low=xgb_params['range_start'],
                high=xgb_params['range_end']
            )
        if bool(xgb_params['exp']):
            value = np.exp(value)
        sample[str(xgb_params['p_name'])] = value
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
    score_dict : dict
        Dictionary containing different scoring metrics
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
    score_dict = universal.get_scores_dict(pred_train, pred_test, data_dict)
    feature_importance = model.get_score(importance_type='gain')
    return score_dict, pred_train, pred_test, feature_importance


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
    feature_importances = []
    output_dir = os.path.expandvars(global_settings['output_dir'])
    previous_files_dir = os.path.join(output_dir, 'previous_files')
    for param_set_nr, parameter_dict in enumerate(parameter_dicts):
        score_dict, pred_train, pred_test, feature_importance = parameter_evaluation(
            parameter_dict, data_dict,
            global_settings['nthread'], global_settings['num_classes'])
        universal.save_predictions_and_score_dict(
            score_dict,
            pred_train,
            pred_test,
            feature_importance,
            param_set_nr,
            previous_files_dir
        )
        score_dicts.append(score_dict)
        pred_trains.append(pred_train)
        pred_tests.append(pred_test)
        feature_importances.append(feature_importance)
    return score_dicts, pred_trains, pred_tests, feature_importances
