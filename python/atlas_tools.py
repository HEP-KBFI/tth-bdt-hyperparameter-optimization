import pandas
from sklearn.model_selection import train_test_split
import os
import numpy as np
import xgboost as xgb
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import pso_main as pm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

parameter_dict= {'eta':0.1, 'max_depth':8, 'min_child_weight':250, 'colsample_bytree':0.1, 'num_boost_round':450}
path_to_file=os.path.expandvars('$HOME/training.csv')

def create_atlas_data_dict(path_to_file, global_settings, plot=False):
    print('::: Loading data from ' + path_to_file + ' :::')
    path_to_file = os.path.expandvars(path_to_file)
    labels_to_drop = ['Kaggle', 'EventId', 'Weight']
    atlas_data_original = pandas.read_csv(path_to_file)
    atlas_data_df = atlas_data_original.copy()
    atlas_data_original['Label'] = atlas_data_original['Label'].replace(
        to_replace='s', value=1)
    atlas_data_original['Label'] = atlas_data_original['Label'].replace(
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
        atlas_data_original, test_size=0.2, random_state=1)
    training_labels = train['Label'].astype(int)
    testing_labels = test['Label'].astype(int)
    traindataset = np.array(train[trainvars].values)
    testdataset = np.array(test[trainvars].values)
    dtrain = xgb.DMatrix(
        traindataset,
        label=training_labels,
        nthread=8, # global_settings['nthread']
        feature_names=trainvars
    )
    dtest = xgb.DMatrix(
        testdataset,
        label=testing_labels,
        nthread=8, # global_settings['nthread']
        feature_names=trainvars
    )
    data_dict = {
        'dtrain': dtrain,
        'dtest': dtest,
        'training_labels': training_labels,
        'testing_labels': testing_labels,
        'trainvars': trainvars,
        'test_full': test,
        'train_full': train
    }
    if plot:
        output_dir = "$HOME/ams"
        plot_bkg_weight_distrib(data_dict, output_dir)
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


def try_different_thresholds(predicted, data_dict, label_type, plot=False):
    output_dir = '$HOME/ams'
    if label_type == 'train':
        factor = 1.25
    elif label_type == 'test':
        factor = 5
    else:
        print("Error: wrong label_type")
    full_key = label_type + '_full'
    label_key = label_type + 'ing_labels'
    weights = data_dict[full_key]['Weight']*factor
    thresholds = np.arange(0, 1, 0.001)
    ams_scores = []
    signals = []
    backgrounds = []
    predicted = pandas.Series([i[1] for i in predicted])
    for threshold in thresholds:
        th_prediction = pandas.Series(
            [1 if pred >= threshold else 0 for pred in predicted])
        signal, background = calculate_s_and_b(th_prediction, data_dict[label_key], weights)
        signals.append(signal)
        backgrounds.append(background)
        ams_score = AMS(signal, background)
        ams_scores.append(ams_score)
    max_score_index = np.argmax(ams_scores)
    if plot:
        best_threshold = thresholds[max_score_index]
        best_prediction = pandas.Series(
            [1 if pred >= threshold else 0 for pred in predicted])
        plot_wrongly_classified(best_prediction, data_dict, label_type, output_dir)
        plot_signal_bkg_yields(
            signals, backgrounds, thresholds,
            max_score_index, label_type, output_dir
        )
        plot_signal_thrs_vs_ams(
            thresholds, ams_scores, max_score_index, label_type, output_dir)
    max_score = ams_scores[max_score_index]
    return max_score


def calculate_s_and_b(prediction, labels, weights):
    signal = 0
    background = 0
    prediction = np.array(prediction)
    labels = np.array(labels)
    weights = np.array(weights)
    for i in range(len(prediction)):
        if prediction[i] == 1:
            if labels[i] == 1:
                signal += weights[i]
            elif labels[i] == 0:
                background += weights[i]
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
    d_ams, test_ams, train_ams = calculate_d_ams(pred_train, pred_test, data_dict)
    feature_importance = model.get_score(importance_type='gain')
    score_dict = {'d_ams': d_ams, 'test_ams': test_ams, 'train_ams': train_ams}
    return score_dict, pred_train, pred_test, feature_importance


def calculate_d_ams(pred_train, pred_test, data_dict, kappa=0.3):
    train_ams = try_different_thresholds(pred_train, data_dict, 'train')
    test_ams = try_different_thresholds(pred_test, data_dict, 'test')
    d_ams = universal.calculate_d_roc(train_ams, test_ams, kappa)
    return d_ams, test_ams, train_ams


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
        calculate_result,
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
    while i <= iterations and pso_settings['compactness_threshold'] < compactness:
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
        new_parameters, current_speeds = pm.prepare_new_day(
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
            print(' ############### New bests ##################')
            print('Parameters: ')
            print(parameter_dicts[index])
            print('Scores: ')
            print(score_dicts[index])
            print(' #############################################')
        compactness = universal.calculate_compactness(parameter_dicts)
        result_dict['best_fitnesses'].append(result_dict['best_fitness'])
        result_dict['compactnesses'].append(compactness)
        inertial_weight += inertial_weight_step
        i += 1
    return result_dict


def save_results(result_dict, output_dir):
    universal.save_feature_importances(result_dict, output_dir)
    result_dict['d_ams'] = result_dict['best_fitnesses']
    universal.best_to_file(
        result_dict['best_parameters'], output_dir, {'d_ams': result_dict['best_fitness']})
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


def plot_signal_thrs_vs_ams(
        thresholds, ams_scores, index, label_type, output_dir
):
    output_dir = os.path.expandvars(output_dir)
    file_out = os.path.join(output_dir, label_type + '_ams_vs_threshold.png')
    plt.plot(thresholds, ams_scores, label='AMS_score')
    plt.axvline(thresholds[index])
    plt.grid(True)
    plt.xlabel("Threshold")
    plt.ylabel("AMS_score")
    plt.legend()
    plt.yscale('log')
    axis = plt.gca()
    axis.set_aspect('auto', adjustable='box')
    axis.xaxis.set_major_locator(ticker.AutoLocator())
    plt.savefig(file_out, bbox_inches='tight')
    plt.close('all')


def plot_signal_bkg_yields(
        signals, backgrounds, thresholds, index, label_type, output_dir
):
    output_dir = os.path.expandvars(output_dir)
    file_out = os.path.join(output_dir, label_type + '_signal_bkg_yield.png')
    plt.plot(thresholds, signals, label='signal yield')
    plt.plot(thresholds, backgrounds, label='background yield')
    plt.axvline(thresholds[index])
    plt.grid(True)
    plt.xlabel("Threshold")
    plt.ylabel("AMS_score")
    plt.yscale('log')
    plt.legend()
    axis = plt.gca()
    axis.set_aspect('auto', adjustable='box')
    axis.xaxis.set_major_locator(ticker.AutoLocator())
    plt.savefig(file_out, bbox_inches='tight')
    plt.close('all')


def plot_bkg_weight_distrib(data_dict, output_dir):
    output_dir = os.path.expandvars(output_dir)
    file_out = os.path.join(output_dir, 'bkg_weight_distrib.png')
    test_background = data_dict['test_full'].loc[data_dict['test_full']['Label'] == 0]
    test_weights = test_background['Weight']
    train_background = data_dict['train_full'].loc[data_dict['train_full']['Label'] == 0]
    train_weights = train_background['Weight']
    bins = plt.hist(
        test_weights, normed=True,
        histtype='step',
        bins= int(np.ceil(np.sqrt(len(test_weights)))), label='test')[1]
    plt.hist(train_weights, normed=True, histtype='step', label='train', bins=bins)
    plt.xlabel("Weight")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(file_out, bbox_inches='tight')
    plt.close('all')


def plot_wrongly_classified(predicted, data_dict, label_type, output_dir):
    bad_signal_weights = []
    bad_bkg_weights = []
    labels = data_dict[label_type + 'ing_labels']
    weights = data_dict[label_type + '_full']['Weight']
    for pred, label, weight in zip(predicted, labels, weights):
        if label == 1 and pred == 0:
            bad_signal_weights.append(weight)
        elif label == 0 and pred == 1:
            bad_bkg_weights.append(weight)
    print(bad_bkg_weights)
    plot_false_classification_weights(
        bad_signal_weights, bad_bkg_weights, label_type, output_dir)



def plot_false_classification_weights(
        bad_signal_weights,
        bad_bkg_weights,
        label_type,
        output_dir
):
    output_dir = os.path.expandvars(output_dir)
    file_out = os.path.join(
        output_dir, label_type + '_false_classification_weigts.png')
    nr_bins = int(np.ceil(np.sqrt(len(bad_signal_weights))))
    bins = plt.hist(
        bad_signal_weights, histtype='step',
        bins=nr_bins,
        label='Signal classified as background'
    )[1]
    plt.hist(
        bad_bkg_weights,
        histtype='step',
        bins=bins,
        label='Background classified as signal'
    )
    plt.yscale('log')
    plt.xlabel("Weight")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig(file_out, bbox_inches='tight')
    plt.close('all')