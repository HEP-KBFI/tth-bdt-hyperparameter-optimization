import pandas
from sklearn.model_selection import train_test_split
import os
import numpy as np
import xgboost as xgb
from tthAnalysis.bdtHyperparameterOptimization import universal


def create_atlas_data_dict(path_to_file, global_settings):
    print('::: Loading data from ' + path_to_file + ' :::')
    path_to_file = os.path.expandvars(path_to_file)
    labels_to_drop = ['Kaggle', 'EventId', 'Weight']
    atlas_data_df = pandas.read_csv(path_to_file)
    atlas_data_df['Label'] = atlas_data_df['Label'].replace(
        to_replace='s', value=1)
    atlas_data_df['Label'] = atlas_data_df['Label'].replace(
        to_replace='b', value=0)
    for trainvar in atlas_data_df.columns:
        for label_to_drop in labels_to_drop:
            if label_to_drop in trainvar:
                try:
                    atlas_data_df = atlas_data_df.drop(trainvar, axis=1)
                except:
                    continue
    trainvars = list(atlas_data_df.columns)
    trainvars.remove('Label')
    output_dir = os.path.expandvars(global_settings['output_dir'])
    info_dir = os.path.join(output_dir, 'previous_files', 'data_dict')
    if not os.path.exists(info_dir):
        os.makedirs(info_dir)
    print(':::::::::::: Creating datasets ::::::::::::::::')
    train, test = train_test_split(
        atlas_data_df, test_size=0.2, random_state=1)
    training_labels = train['Label'].astype(int)
    testing_labels = test['Label'].astype(int)
    traindataset = np.array(train[trainvars].values)
    testdataset = np.array(test[trainvars].values)
    dtrain = xgb.DMatrix(
        traindataset,
        label=training_labels,
        nthread=global_settings['nthread'],
        feature_names=trainvars
    )
    dtest = xgb.DMatrix(
        testdataset,
        label=testing_labels,
        nthread=global_settings['nthread'],
        feature_names=trainvars
    )
    data_dict = {
        'dtrain': dtrain,
        'dtest': dtest,
        'training_labels': training_labels,
        'testing_labels': testing_labels,
        'trainvars': trainvars
    }
    universal.write_data_dict_info(info_dir, data_dict)
    return data_dict


def AMS(s, b):
    """ Approximate Median Significance defined as:
        AMS = sqrt(
                2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s}
              )        
    where b_r = 10, b = background, s = signal, log is natural logarithm
    """
    br = 10.0
    radicand = 2 *( (s+b+br) * np.log (1.0 + s/(b+br)) -s)
    if radicand < 0:
        print 'radicand is negative. Exiting'
        exit()
    else:
        return np.sqrt(radicand)


def try_different_thresholds(predicted, true_labels):
    thresholds = np.arange(0, 1, 0.001)
    ams_scores = []
    predicted = pandas.Series([i[1] for i in predicted])
    for threshold in thresholds:
        th_prediction = pandas.Series(
            [1 if pred >= threshold else 0 for pred in predicted])
        signal, background = calculate_s_and_b(th_prediction, true_labels)
        ams_score = AMS(signal, background)
        ams_scores.append(ams_score)
    max_score = max(ams_scores)
    return max_score


def calculate_s_and_b(prediction, labels):
    labels = labels.reset_index(drop=True)
    pred_signal_indices = prediction.index[prediction == 1]
    pred_signal_labels = labels[pred_signal_indices]
    signal = sum(pred_signal_labels == 1)
    background = sum(pred_signal_labels == 0)
    return signal, background


def higgs_evaluation(parameter_dict, data_dict, nthread, num_class):
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
    d_ams = calculate_d_ams(pred_train, pred_test, data_dict)
    feature_importance = model.get_score(importance_type='gain')
    score_dict = {'d_ams': d_ams}
    return score_dict, pred_train, pred_test, feature_importance


def calculate_d_ams(pred_train, pred_test, data_dict, kappa=1.5):
    train_score = try_different_thresholds(pred_train, data_dict['training_labels'])
    test_score = try_different_thresholds(pred_test, data_dict['testing_labels'])
    d_ams = universal.calculate_d_roc(train_score, test_score, kappa)
    return d_ams


def ensemble_fitness(
        parameter_dicts,
        data_dict,
        global_settings
):
    scores = []
    pred_tests = []
    pred_trains = []
    feature_importances = []
    for parameter_dict in parameter_dicts:
        score, pred_train, pred_test, feature_importance = higgs_evaluation(
            parameter_dict, data_dict,
            global_settings['nthread'], global_settings['num_class']
        )
        scores.append(score)
        pred_trains.append(pred_train)
        pred_tests.append(pred_test)
        feature_importances.append(feature_importance)
    return scores, pred_trains, pred_tests, feature_importances


def run_pso(
        data_dict,
        value_dicts,
        calculate_fitnesses,
        parameter_dicts,
        output_dir
):
    '''Performs the whole particle swarm optimization

    Parameters:
    ----------
    global_settings : dict
        Global settings for the run.
    pso_settings : dict
        Particle swarm settings for the run
    parameter_dicts : list of dicts
        The parameter-sets of all particles.

    Returns:
    -------
    result_dict : dict
        Dictionary that contains the results like best_parameters,
        best_fitnesses, avg_scores, pred_train, pred_test, data_dict
    '''
    print(':::::::: Initializing :::::::::')
    settings_dir = os.path.join(output_dir, 'run_settings')
    global_settings = universal.read_settings(settings_dir, 'global')
    pso_settings = universal.read_settings(settings_dir, 'pso')
    inertial_weight, inertial_weight_step = pm.get_weight_step(pso_settings)
    iterations = pso_settings['iterations']
    i = 0
    new_parameters = parameter_dicts
    personal_bests = {}
    score_dicts, pred_trains, pred_tests, feature_importances = calculate_result(
        parameter_dicts, data_dict, global_settings)
    fitnesses = universal.fitness_to_list(
        score_dicts, fitness_key='d_ams')
    result_dict = {}
    index = np.argmax(fitnesses)
    compactness = universal.calculate_compactness(parameter_dicts)
    result_dict['best_fitness'] = fitnesses[index]
    result_dict['pred_test'] = pred_tests[index]
    result_dict['pred_train'] = pred_trains[index]
    result_dict['feature_importances'] = feature_importances[index]
    result_dict['best_parameters'] = parameter_dicts[index]
    result_dict['best_fitnesses'] = [fitnesses[index]]
    result_dict['compactnesses'] = [compactness]
    personal_bests = parameter_dicts
    best_fitnesses = fitnesses
    current_speeds = pm.initialize_speeds(parameter_dicts)
    print('::::::::::: Optimizing ::::::::::')
    while i <= iterations and compactness_threshold < compactness:
        print('---- Iteration: ' + str(i) + '----')
        parameter_dicts = new_parameters
        print(' --- Compactness: ' + str(compactness) + ' ---')
        score_dicts, pred_trains, pred_tests, feature_importances = calculate_result(
            parameter_dicts, data_dict, global_settings)
        fitnesses = universal.fitness_to_list(
            score_dicts, fitness_key='d_ams')
        best_fitnesses = pm.find_best_fitness(fitnesses, best_fitnesses)
        personal_bests = pm.calculate_personal_bests(
            fitnesses, best_fitnesses, parameter_dicts, personal_bests)
        weight_dict = {
            'c1': pso_settings['c1'],
            'c2': pso_settings['c2'],
            'w': inertial_weight}
        new_parameters, current_speeds = prepare_new_day(
            personal_bests, parameter_dicts,
            result_dict['best_parameters'],
            current_speeds, value_dicts,
            weight_dict
        )
        if result_dict['best_fitness'] < max(fitnesses):
            index = np.argmax(fitnesses)
            result_dict['best_fitness'] = max(fitnesses)
            result_dict['best_parameters'] = parameter_dicts[index]
            result_dict['pred_test'] = pred_tests[index]
            result_dict['pred_train'] = pred_trains[index]
            result_dict['feature_importances'] = feature_importances[index]
        compactness = universal.calculate_compactness(parameter_dicts)
        result_dict['best_fitnesses'].append(result_dict['best_fitness'])
        result_dict['compactnesses'].append(compactness)
        inertial_weight += inertial_weight_step
        i += 1
    return result_dict


def save_results(result_dict, output_dir):
    universal.save_feature_importances(result_dict, output_dir)
    universal.best_to_file(
        result_dict['best_parameters'], output_dir, result_dict['score_dict'])
    universal.plot_costfunction(
        result_dict['compactnesses'], output_dir, y_label='Compactness (cov)')
    save_extra_results(result_dict, output_dir)
    create_extra_plots(result_dict, output_dir)



def save_extra_results(result_dict, output_dir):
    '''Saves the scoring and stopping criteria values to file.

    Parameters:
    ----------
    result_dicts : dict
        Dictionary containing the results and info that is to be plotted
    output_dir : str
        Path to the directory where the plots are to be saved

    Returns:
    -------
    Nothing
    '''
    file_out1 = os.path.join(output_dir, 'scoring_metrics.json')
    file_out2 = os.path.join(output_dir, 'stopping_criteria.json')
    keys1 = ['d_ams']
    keys2 = ['compactnesses']
    universal.save_single_file(keys1, result_dict, file_out1)
    universal.save_single_file(keys2, result_dict, file_out2)
    universal.save_fitness_improvement(result_dict, keys1, output_dir)


def create_extra_plots(result_dict, output_dir):
    '''Creates some additional figures

    Parameters:
    ----------
    result_dicts : dict
        Dictionary containing the results and info that is to be plotted
    output_dir : str
        Path to the directory where the plots are to be saved

    Returns:
    -------
    Nothing
    '''
    plot_out1 = os.path.join(output_dir, 'scoring_metrics.png')
    plot_out2 = os.path.join(output_dir, 'stopping_criteria.png')
    keys1 = ['best_fitnesses']
    keys2 = ['compactnesses']
    universal.plot_single_evolution(
        keys1, result_dict, 'Scoring metrics', plot_out1)
    universal.plot_single_evolution(
        keys2, result_dict, 'Stopping criteria', plot_out2)