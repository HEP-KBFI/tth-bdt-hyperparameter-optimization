'''Universal functions to be used in evolutionary algorithms
'''
import warnings
import itertools
import json
import os
import numpy as np
import glob
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('agg')
import shutil
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
warnings.filterwarnings('ignore', category=RuntimeWarning)


def read_parameters(param_file):
    '''Read values form a '.json' file

    Parameters:
    ----------
    param_file : str
        Path to the '.json' file

    Returns:
    -------
    value_dicts : list containing dicts
        List of parameter dictionaries
    '''
    value_dicts = []
    with open(param_file, 'rt') as file:
        for line in file:
            json_dict = json.loads(line)
            value_dicts.append(json_dict)
    return value_dicts


def best_to_file(best_values, output_dir, assesment):
    '''Saves the best parameters and the scores to a file
    'best_parameters.json'

    Parameters:
    ----------
    best_values : dict
        Best parameters found during the evolutionary algorithm
    output_dir : str
        Directory where best parameters and assessment is saved
    assessment : dict
        Different scores for the best parameters found for both train and test
        dataset.
    '''
    output_path = os.path.join(output_dir, 'best_parameters.json')
    with open(output_path, 'w') as file:
        json.dump(best_values, file)
        file.write('\n')
        json.dump(assesment, file)


def calculate_d_score(
        pred_train,
        pred_test,
        data_dict
):
    '''Calculate the fitness using the old method [Currently not used anymore].
    Different return type than the methods currently in use

    Parameters
    ----------
    pred_train : list
        List of numpy arrays containing probabilities for all labels
        for the training sample
    pred_test : list
        List of numpy arrays containing probabilities for all labels
        for the testing sample
    data_dict : dict
        Dictionary that contains the labels for testing and training. Keys are
        called 'testing_labels' and 'training_labels'

    Returns:
    -------
    d_score : float
        Score for the parameters used.
    '''
    train_value = []
    for vector in pred_train:
        train_value.append(np.argmax(vector))
    train_pairs = list(zip(data_dict['training_labels'], train_value))
    train_n = 0
    for pair in train_pairs:
        if pair[0] == pair[1]:
            train_n = train_n + 1
    train_score = float(train_n) / len(data_dict['training_labels'])
    test_value = []
    for vector in pred_test:
        test_value.append(np.argmax(vector))
    test_pairs = list(zip(data_dict['testing_labels'], test_value))
    test_n = 0
    for pair in test_pairs:
        if pair[0] == pair[1]:
            test_n = test_n + 1
    test_score = float(test_n)/len(data_dict['testing_labels'])
    d_score = score(train_score, test_score)
    return d_score


def score(train_score, test_score):
    '''Calculates the final score by minimizing the difference of test and
    train score.

    Parameters:
    ----------
    train_score : float
        Score for the parameters when evaluating them on the train dataset
    test_score : float
        Score for the parameters when evaluating them on the train dataset

    Returns:
    -------
    d_score : float
        Score for the set of parameters.
    '''
    d_score = np.mean([
        (1 - (train_score - test_score)),
        (1 - (train_score - test_score)),
        test_score
    ])
    return d_score


def calculate_d_roc(train_auc, test_auc, kappa):
    ''' Calculates the score from test and train AUC score with the method as
    in the D-score

    Parameters:
    ----------
    train_auc : float
        AUC of the training sample
    test_auc : float
        AUC of the testing sample
    kappa : float
        Weighing factor for the difference between test and train auc

    Returns:
    -------
    d_roc : float
        Score based on D-score and AUC
    '''
    difference = max(0, train_auc - test_auc)
    weighed_difference = kappa * (1 - difference)
    denominator = kappa + 1
    d_roc = (test_auc + weighed_difference) / denominator
    return d_roc


def calculate_conf_matrix(predicted_train, predicted_test, data_dict):
    '''Produces the confusion matrix for train and test sample

    Parameters
    ----------
    predicted_train : list
        List containing the class predictions for each model training event
    predicted_test : list
        List containing the class predictions for each model testing event
    data_dict : dict
        Dictionary that contains the labels for testing and training. Keys are
        called 'testing_labels' and 'training_labels'

    Returns:
    -------
    train_confusionMatrix : array, shape = [n_classes, n_classes]
        Confusion matrix of true and predicted labels for train samples
    test_confusionMatrix : array, shape = [n_classes, n_classes]
        Confusion matrix of true and predicted labels for test samples

    Comments:
    --------
    Since sklearn's confusion matrix returns it in the form of:
    [[TN, FN],[FP, TP]] if given the parameters in the order of
    (true, predicted) and we want to have it in the form [[TP, FP], [FN, TN]],
    then we switch the order of the parameters
    '''
    true_test = data_dict['testing_labels']
    true_train = data_dict['training_labels']
    test_confusionmatrix = confusion_matrix(predicted_test, true_test)
    train_confusionmatrix = confusion_matrix(predicted_train, true_train)
    return train_confusionmatrix, test_confusionmatrix


def calculate_f1_score(confusionmatrix):
    '''Calculates the F1-score and the G-score

    Parameters
    ----------
    confusionmatrix : array, shape = [n_classes, n_classes]
        Confusion matrix of true and predicted labels

    Returns
    -------
    mean_f1 : float
        Mean F1-score of all the different lables
    mean_g : float
        Mean G-score of all the different labels
    '''
    nr_labels = len(confusionmatrix)
    labels = np.arange(0, nr_labels)
    g_scores = []
    f1_scores = []
    if nr_labels > 2:
        for label in labels:
            false_positives = 0
            false_negatives = 0
            new_labels = np.delete(labels, label)
            true_positives = confusionmatrix[label, label]
            for new_label in new_labels:
                false_negatives += confusionmatrix[label, new_label]
                false_positives += confusionmatrix[new_label, label]
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_score = 2 * (precision * recall) / (precision + recall)
        g_score = np.sqrt(precision * recall)
        f1_scores.append(f1_score)
        g_scores.append(g_score)
        mean_f1 = np.mean(f1_scores)
        mean_g = np.mean(g_scores)
    elif nr_labels == 1:
        true_positives = confusionmatrix[0][0]
        false_positives = 0
        false_negatives = 0
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        mean_f1 = 2 * (precision * recall) / (precision + recall)
        mean_g = np.sqrt(precision * recall)
    elif nr_labels == 2:
        true_positives = confusionmatrix[0][0]
        false_negatives = confusionmatrix[1][0]
        false_positives = confusionmatrix[0][1]
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        mean_f1 = 2 * (precision * recall) / (precision + recall)
        mean_g = np.sqrt(precision * recall)
    else:
        raise ValueError('The passed confusionmatrix is not correct')
    return mean_f1, mean_g


def get_most_probable(pred_train, pred_test):
    '''Finds the predicted lable given the probabilities for each

    Parameters
    ----------
    pred_train : list
        List of numpy arrays containing probabilities for all labels
        for the training sample
    pred_test : list
        List of numpy arrays containing probabilities for all labels
        for the testing sample

    Returns
    -------
    predicted_train : list
        List of predicted values for each train instance
    predicted_test : list
        List of predicted values for each test instance
    '''
    predicted_test = []
    predicted_train = []
    for vector in pred_test:
        predicted_test.append(np.argmax(vector))
    for vector in pred_train:
        predicted_train.append(np.argmax(vector))
    return predicted_train, predicted_test


def main_f1_calculate(pred_train, pred_test, data_dict):
    '''Main function that calculates the F1-score and G-score for train and
    test samples

    Parameters
    ----------
    pred_train : list
        List of numpy arrays containing probabilities for all labels
        for the training sample
    pred_test : list
        List of numpy arrays containing probabilities for all labels
        for the testing sample
    data_dict : dict
        Dictionary that contains the labels for testing and training. Keys are
        called 'testing_labels' and 'training_labels'

    Returns
    -------
    assessment : dict
        Dictionary that contains the F1 and G-score for test and train samples
        keys = ['Train_F1', 'Train_G', 'Test_F1', 'Test_G']
    '''
    predicted_train, predicted_test = get_most_probable(pred_train, pred_test)
    train_confusionmatrix, test_confusionmatrix = calculate_conf_matrix(
        predicted_train, predicted_test, data_dict)
    train_f1_score, train_g_score = calculate_f1_score(train_confusionmatrix)
    test_f1_score, test_g_score = calculate_f1_score(test_confusionmatrix)
    assessment = {
        'Train_F1': train_f1_score,
        'Train_G': train_g_score,
        'Test_F1': test_f1_score,
        'Test_G': test_g_score
    }
    return assessment


def get_scores_dict(pred_train, pred_test, data_dict):
    '''Calculates different scoring metrics

    Parameters:
    ----------
    pred_train : list
        List of numpy arrays containing probabilities for all labels
        for the training sample
    pred_test : list
        List of numpy arrays containing probabilities for all labels
        for the testing sample
    data_dict : dict
        Dictionary that contains the labels for testing and training. Keys are
        called 'testing_labels' and 'training_labels'

    Returns:
    -------
    score_dict : dict
        Dictionary containing different scoring metrics
    '''
    prob_train, prob_test = get_most_probable(pred_train, pred_test)
    train_conf_matrix, test_conf_matrix = calculate_conf_matrix(
        prob_train, prob_test, data_dict)
    g_score_test, f1_score_test = calculate_f1_score(
        test_conf_matrix)
    g_score_train, f1_score_train = calculate_f1_score(
        train_conf_matrix)
    d_score = calculate_d_score(pred_train, pred_test, data_dict)
    train_auc, test_auc = calculate_auc(
        data_dict, pred_train, pred_test)[:2]
    score_dict = {
        'f1_score': f1_score_test,
        'g_score': g_score_test,
        'test_auc': test_auc,
        'f1_score_train': f1_score_train,
        'g_score_train': g_score_train,
        'train_auc': train_auc,
        'd_score': d_score
    }
    kappas = [0.0, 0.5, 1.0, 1.5, 2.0]
    for kappa in kappas:
        d_roc = calculate_d_roc(train_auc, test_auc, kappa)
        key_name = 'd_roc_' + str(kappa)
        score_dict[key_name] = d_roc
    return score_dict


def save_results(result_dict, output_dir, plot_roc=True, plot_extras=False):
    '''Saves the results from the result_dict to files. Optionally produces
    also plots for ROC and average

    Parameters:
    -----------
    results_dict : dict
        Dictionary that contains keys ['pred_train', 'pred_test', 'data_dict'
        and 'best_parameters']
        If plotROC=True then also key 'avg_scores' is needed
    output_dir : str
        Path to the dictionary where results are saved. If the directory does
        not exist, one will be created
    plotROC=True : bool
        [optional] Whether to plot ROC curve and average scores. Optional
    plot_extras=False : bool
        [optional] Whether to plot extra features

    Returns:
    -------
    Nothing
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    keys = [
        'best_test_auc', 'best_train_auc',
        'best_g_score', 'best_f1_score',
        'best_d_score'
    ]
    assessment = dict(
        (key, result_dict[key]) for key in keys if key in result_dict)
    auc_info = calculate_auc(
        result_dict['data_dict'],
        result_dict['pred_train'],
        result_dict['pred_test']
    )[-1]
    if plot_roc:
        plotting(output_dir, auc_info, result_dict['avg_scores'])
    if plot_extras:
        create_extra_plots(result_dict, output_dir)
        save_extra_results(result_dict, output_dir)
        save_feature_importances(result_dict, output_dir)
    best_to_file(
        result_dict['best_parameters'], output_dir, assessment)


def save_feature_importances(result_dict, output_dir):
    '''Saves the feature importances into a feature_importances.json file

    Parameters:
    ----------
    result_dict : dict
        Dicotionay containing the results of the optimization
    output_dir : str
        Path to the output_dir

    Returns:
    -------
    Nothing
    '''
    output_path = os.path.join(output_dir, 'feature_importances.json')
    with open(output_path, 'w') as file:
        json.dump(result_dict['feature_importances'], file)


def save_run_settings(output_dir):
    '''Saves the run settings for future reference

    Parameters:
    ----------
    output_dir : str
        Path to the output directory

    Returns:
    -------
    Nothing
    '''
    settings_dir = os.path.join(output_dir, 'run_settings')
    if not os.path.exists(settings_dir):
        os.makedirs(settings_dir)
    cmssw_base = os.path.expandvars('$CMSSW_BASE')
    wild_card_path = os.path.join(
        cmssw_base,
        'src',
        'tthAnalysis',
        'bdtHyperparameterOptimization',
        'data',
        '*.json')
    for path in glob.glob(wild_card_path):
        shutil.copy(path, settings_dir)


def calculate_auc(data_dict, pred_train, pred_test):
    '''Calculates the area under curve for training and testing dataset using
    the predicted labels

    Parameters:
    ----------
    data_dict : dict
        Dictionary that contains the labels for testing and training. Keys are
        called 'testing_labels' and 'training_labels'
    pred_train : list of lists
        Predicted labels of the training dataset
    pred_test : list of lists
        Predicted labels of the testing dataset

    Returns:
    -------
    train_auc : float
        Area under curve for training dataset
    test_aud : float
        Area under curve for testing dataset
    '''
    x_train, y_train = roc(
        data_dict['training_labels'], pred_train)
    x_test, y_test = roc(
        data_dict['testing_labels'], pred_test)
    test_auc = (-1) * np.trapz(y_test, x_test)
    train_auc = (-1) * np.trapz(y_train, x_train)
    info = {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test
    }
    return train_auc, test_auc, info



def calculate_improvement_wAVG(avg_scores, improvements, threshold):
    '''Calculates the improvement based on the average scores. Purpose:
    stopping criteria. Currently used only in GA algorithm.

    Parameters:
    -----------
    avg_scores : list
        Average scores of each iteration in the evolutionary algorithm
    improvements : list
        List of improvements of previous iterations
    threshold : float
        Stopping criteria.

    Returns:
    --------
    improvements : list
        List of improvements
    imporvement : float
        Improvement for comparing

    Comments:
    ---------
    Elif clause used in order to have last 2 iterations less than the threshold
    '''
    if len(avg_scores) > 1:
        improvements.append((avg_scores[-1]-avg_scores[-2])/avg_scores[-2])
        improvement = improvements[-1]
    if len(improvements) < 2:
        improvement = 1
    elif improvement <= threshold:
        improvement = improvements[-2]
    return improvements, improvement


def calculate_compactness(parameter_dicts):
    '''Calculates the improvement based on how similar are different sets of
    parameters

    Parameters:
    ----------
    parameter_dicts : list of dicts
        List of dictionaries to be compared for compactness.

    Returns:
    -------
    mean_cov : float
        Coefficient of variation of different sets of parameters.
    '''
    keys = parameter_dicts[0].keys()
    list_dict = values_to_list_dict(keys, parameter_dicts)
    mean_cov = calculate_dict_mean_coeff_of_variation(list_dict)
    return mean_cov


def values_to_list_dict(keys, parameter_dicts):
    '''Adds same key values from different dictionaries into a list w

    Parameters:
    ----------
    keys : list
        list of keys for which same key values are added to a list
    parameter_dicts : list of dicts
        List of parameter dictionaries.

    Returns:
    -------
    list_dict: dict
        Dictionary containing lists as valus.
    '''
    list_dict = {}
    for key in keys:
        key = str(key)
        list_dict[key] = []
        for parameter_dict in parameter_dicts:
            list_dict[key].append(parameter_dict[key])
    return list_dict


def calculate_dict_mean_coeff_of_variation(list_dict):
    '''Calculate the mean coefficient of variation for a given dict filled
    with lists as values

    Parameters:
    ----------
    list_dict : dict
        Dictionary containing lists as values

    Returns:
    -------
    mean_coeff_of_variation : float
        Mean coefficient of variation for a given dictionary haveing lists as
        values
    '''
    coeff_of_variations = []
    for key in list_dict:
        values = list_dict[key]
        coeff_of_variation = np.std(values)/np.mean(values)
        coeff_of_variations.append(coeff_of_variation)
    mean_coeff_of_variation = np.mean(coeff_of_variations)
    return mean_coeff_of_variation


def roc(labels, pred_vectors):
    '''Calculate the ROC values using the method used in Dianas thesis.

    Parameters:
    ----------
    labels : list
        List of true labels
    pred_vectors: list of lists
        List of lists that contain the probabilities for an event to belong to
        a certain label

    Returns:
    -------
    false_positive_rate : list
        List of false positives for given thresholds
    true_positive_rate : list
        List of true positives for given thresholds
    '''
    thresholds = np.arange(0, 1, 0.01)
    number_bg = len(pred_vectors[0]) - 1
    true_positive_rate = []
    false_positive_rate = []
    for threshold in thresholds:
        signal = []
        for vector in pred_vectors:
            sig_vector = np.array(vector) > threshold
            sig_vector = sig_vector.tolist()
            result = []
            for i, element in enumerate(sig_vector):
                if element:
                    result.append(i)
            signal.append(result)
        pairs = list(zip(labels, signal))
        sig_score = 0
        bg_score = 0
        for pair in pairs:
            for i in pair[1]:
                if pair[0] == i:
                    sig_score += 1
                else:
                    bg_score += 1
        true_positive_rate.append(float(sig_score)/len(labels))
        false_positive_rate.append(float(bg_score)/(number_bg*len(labels)))
    return false_positive_rate, true_positive_rate


def plotting(
        output_dir,
        auc_info,
        avg_scores
):
    '''Main function for plotting costfunction and the ROC.

    Parameters:
    ----------
    output_dir : str
        Path to the directory where figures will be saved
    auc_info : dict
        Dictionary containing x and y values for test and train sample from
        ROC calculation
    avg_scores : list
        List of average scores of all iterations

    Returns:
    -------
    Nothing
    '''

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plot_roc_curve(output_dir, auc_info)
    plot_costfunction(avg_scores, output_dir)


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
    keys1 = [
        'best_test_aucs', 'best_train_aucs',
        'best_g_scores', 'best_f1_scores',
        'best_d_scores']
    keys2 = ['compactnesses', 'avg_scores']
    plot_single_evolution(keys1, result_dict, 'Scoring metrics', plot_out1)
    plot_single_evolution(keys2, result_dict, 'Stopping criteria', plot_out2)


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
    keys1 = [
        'best_test_aucs', 'best_train_aucs',
        'best_g_scores', 'best_fitnesses',
        'best_f1_scores', 'best_d_scores'
    ]
    keys2 = ['compactnesses', 'avg_scores']
    save_single_file(keys1, result_dict, file_out1)
    save_single_file(keys2, result_dict, file_out2)
    save_fitness_improvement(result_dict, keys1, output_dir)


def save_fitness_improvement(result_dict, keys, output_dir):
    '''Finds how much the performance increased based on different scoring
    metrics

    Parameters:
    ----------
    result_dict : dict
        Dictionary containing the results and info that is to be plotted

    Returns:
    -------
    Nothing
    '''
    output_path = os.path.join(output_dir, 'fitness_improvement.json')
    relative_improvement = {}
    for key in keys:
        try:
            improvement = result_dict[key][-1] - result_dict[key][0]
            relative_improvement[key] = improvement/result_dict[key][0]
        except:
            improvements = []
            for i in result_dict[key].keys():
                improvements.append(result_dict[key][i][-1] - result_dict[key][i][0])
            improvement = max(improvements)
            relative_improvement[key] = improvement/result_dict[key]['final'][0]
    with open(output_path, 'w') as file:
        json.dump(relative_improvement, file)


def save_single_file(keys, result_dict, file_out):
    '''Saves a single file with the results

    Parameters:
    ----------
    keys : list
        List of keys of values to be plotted
    result_dict : dict
        Dictionary containing the results and info that is to be plotted
    file_out : str
        Location where the file is to be saved

    Returns:
    -------
    Nothing
    '''
    dict_list = []
    for key in keys:
        key_dict = {}
        key_dict[key] = result_dict[key]
        dict_list.append(key_dict)
    with open(file_out, 'wt') as file:
        for single_dict in dict_list:
            json.dump(single_dict, file)
            file.write('\n')


def plot_single_evolution(keys, result_dict, title, plot_out):
    '''Plots the chosen key_value evolution over iterations

    Parameters:
    ----------
    keys : list
        List of keys of values to be plotted
    result_dict : dict
        Dictionary containing the results and info that is to be plotted
    title : str
        Name of the figure
    plot_out : str
        Location where the figure is to be saved

    Returns:
    -------
    Nothing
    '''
    try:
        n_gens = len(result_dict[keys[0]])
        iteration_nr = np.arange(n_gens)
        for key in keys:
            plt.plot(iteration_nr, result_dict[key], label=key)
        plt.xlim(0, n_gens - 1)
        plt.xticks(np.arange(n_gens - 1))
    except: # in case of a genetic algorithm with multiple subpopulations
        for i in result_dict[keys[0]].keys():
            n_gens = len(result_dict[keys[0]][i])
            if i != 'final':
                iteration_nr = np.arange(n_gens)
            if i == 'final':
                n_gens_final = 2 * n_gens - 1
                iteration_nr = np.arange(n_gens - 1, n_gens_final)
            for key in keys:
                plt.plot(iteration_nr, result_dict[key][i], label=key)
        plt.xlim(0, n_gens_final - 1)
        plt.xticks(np.arange(n_gens_final - 1))
    finally:
        plt.xlabel('Iteration number / #')
        plt.ylabel('scoring_metric')
        axis = plt.gca()
        axis.set_aspect('auto', adjustable='box')
        axis.xaxis.set_major_locator(ticker.AutoLocator())
        plt.grid(True)
        plt.legend(loc='lower right')
        plt.title(title)
        plt.tick_params(top=True, right=True, direction='in')
        plt.savefig(plot_out)
        plt.close('all')


def plot_roc_curve(
        output_dir,
        auc_info
):
    '''Creates the ROC plot.

    Parameters:
    ----------
    output_dir : str
        Path to the directory where figures will be saved
    auc_info : dict
        Dictionary containing x and y values for test and train sample from
        ROC calculation

    Returns:
    -------
    Nothing
    '''
    plot_out = os.path.join(output_dir, 'roc.png')
    plt.xlabel('Proportion of false values')
    plt.ylabel('Proportion of true values')
    axis = plt.gca()
    axis.set_aspect('equal')
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    plt.plot(
        auc_info['x_train'], auc_info['y_train'], color='k', linestyle='--',
        label='optimized values, training data', zorder=100
    )
    plt.plot(
        auc_info['x_test'], auc_info['y_test'], color='r', linestyle='-',
        label='optimized values, testing data'
    )
    plt.tick_params(top=True, right=True, direction='in')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(plot_out)
    plt.close('all')


def plot_costfunction(avg_scores, output_dir, y_label='Fitness score'):
    '''Creates a plot of the cost function

    Parameters:
    ----------
    avg_scores : list
        List of average scores of all itereations of the evolutionary algorithm
    output_dir : str
        Path to the directory where the plot is saved

    Returns:
    -------
    Nothing
    '''
    plot_out = os.path.join(output_dir, 'costFunction.png')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        n_gens = len(avg_scores)
        gen_numbers = np.arange(n_gens)
        plt.plot(gen_numbers, avg_scores, color='k')
        plt.xlim(0, n_gens - 1)
        plt.xticks(np.arange(n_gens - 1))
    except:  # in case of a genetic algorithm with multiple subpopulations
        x_max = 0
        for i in avg_scores:
            n_gens = len(avg_scores[i])
            if n_gens > x_max:
                x_max = n_gens
            if i != 'final':
                gen_numbers = np.arange(n_gens)
                plt.plot(gen_numbers, avg_scores[i], color='b')
        for i in avg_scores:
            if len(avg_scores[i]) < x_max and i != 'final':
                line_length = x_max - len(avg_scores[i]) + 1
                y_values = [avg_scores[i][-1] for n in range(line_length)]
                x_values = np.arange(len(avg_scores[i]) - 1, x_max)
                plt.plot(x_values, y_values, color='b', linestyle='--', alpha=0.2)
        n_gens_final = x_max + len(avg_scores['final']) - 1
        gen_numbers = np.arange(x_max - 1, n_gens_final)
        plt.plot(gen_numbers, avg_scores['final'], color='k')
        plt.xlim(0, n_gens_final - 1)
        plt.xticks(np.arange(n_gens_final - 1))
    finally:
        plt.xlabel('Generation')
        plt.ylabel(y_label)
        axis = plt.gca()
        axis.set_aspect('auto', adjustable='box')
        axis.xaxis.set_major_locator(ticker.AutoLocator())
        plt.grid(True)
        plt.tick_params(top=True, right=True, direction='in')
        plt.savefig(plot_out)
        plt.close('all')


def to_one_dict(list_of_dicts):
    '''Puts dictionaries from list into one big dictionary. (can't have same
    keys)

    Parameters:
    ----------
    list_of_dicts : list of dicts
        List filled with dictionaries to be put together into one big dict

    Returns:
    -------
    main_dict : dict
        Dictionary containing all the small dictionary keys.
    '''
    main_dict = {}
    for elem in list_of_dicts:
        key = list(elem.keys())[0]
        main_dict[key] = elem[key]
    return main_dict


def read_settings(settings_dir, group):
    '''Function to read the global settings of the optimization

    Parameters:
    -----------
    group : str
        Group of settings wanted. Either: 'global', 'ga' or 'pso'

    Returns:
    --------
    settings_dict : dict
        Dictionary containing the settings for the optimization
    '''
    settings_path = os.path.join(
        settings_dir,
        group + '_settings.json')
    parameter_list = read_parameters(settings_path)
    settings_dict = to_one_dict(parameter_list)
    return settings_dict


def fitness_to_list(score_dicts, fitness_key='d_roc'):
    '''Puts the fitness values according to the chosen fitness_key from
    the dictionary to a list

    Parameters:
    ----------
    score_dicts : list of dicts
        List containing dictionaries filled with different scores
    [fitness_key='f1_score'] : str
        Name of the key what is used as the fitness score

    Returns:
    -------
    fitnesses : list
        List of fitness scores for each particle
    '''
    fitnesses = []
    for score_dict in score_dicts:
        fitnesses.append(score_dict[fitness_key])
    return fitnesses


def save_predictions_and_score_dict(
        score_dict,
        pred_train,
        pred_test,
        feature_importance,
        param_set_nr,
        previous_files_dir
):
    '''Saves the iteration info
    '''
    iter_nr = find_iter_number(previous_files_dir)
    save_dir = os.path.join(previous_files_dir, str(iter_nr), str(param_set_nr))
    save_info(score_dict, pred_train, pred_test, save_dir, feature_importance)


def find_iter_number(previous_files_dir):
    '''Finds the number iterations done

    Parameters:
    ----------
    previous_files_dir : str
        Path to the directory where old iterations are saved

    Returns:
    -------
    iter_number : int
        Number of the current iteration
    '''
    wild_card_path = os.path.join(previous_files_dir, 'iteration_*')
    iter_number = len(glob.glob(wild_card_path))
    return iter_number


def save_info(
        score_dict,
        pred_train,
        pred_test,
        save_dir,
        feature_importance
):
    '''Saves the score, pred_train, pred_test into appropriate files in
    the wanted directory

    Parameters:
    ----------
    score_dict : float
        Evaluation of the fitness of the parameters
    pred_train : list of lists
        Predicted probabilities for an event to have a certain label
    pred_test : list of lists
        Predicted probabilities for an event to have a certain label
    save_dir : str
        Path to the directory where restults will be saved

    Returns:
    -------
    Nothing
    '''
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    train_path = os.path.join(save_dir, 'pred_train.lst')
    test_path = os.path.join(save_dir, 'pred_test.lst')
    score_path = os.path.join(save_dir, 'score.json')
    importance_path = os.path.join(save_dir, 'feature_importances.json')
    with open(train_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(pred_train)
    with open(test_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerows(pred_test)
    with open(score_path, 'w') as file:
        json.dump(score_dict, file)
    with open(importance_path, 'w') as file:
        json.dump(feature_importance, file)


def write_data_dict_info(info_dir, data_dict):
    test_labels_path = os.path.join(info_dir, 'testing_labels.txt')
    train_labels_path = os.path.join(info_dir, 'training_labels.txt')
    with open(train_labels_path, 'wt') as out_file:
        for elem in list(data_dict['training_labels']):
            out_file.write(str(elem) + '\n')
    with open(test_labels_path, 'wt') as out_file:
        for elem in list(data_dict['testing_labels']):
            out_file.write(str(elem) + '\n')
    if "training_processes" in data_dict:
        train_process_path = os.path.join(info_dir, 'training_process.txt')
        test_process_path = os.path.join(info_dir, 'testing_process.txt')
        with open(train_process_path, 'wt') as out_file:
            for elem in list(data_dict['training_processes']):
                out_file.write(str(elem) + '\n')
        with open(test_process_path, 'wt') as out_file:
            for elem in list(data_dict['testing_processes']):
                out_file.write(str(elem) + '\n')


# One-vs-All ROC


def create_pairs(matrix):
    '''Creates all the possible pairs from given numbers

    Parameters:
    ----------
    matrix: list of lists
        Matrix that contains lists as rows. One row represents the
        possibilities for an event to belong to a certain class

    Returns:
    -------
    pairs : list
        List of all possible pairs of labels (from 0 to number of matrix
        columns)
    '''
    row = np.arange(len(matrix[0]))
    combinations = itertools.combinations(row, 2)
    pairs = []
    for pair in combinations:
        pairs.append(pair)
    return pairs


def get_values(matrix, pair):
    '''Normalizes the probability values for pairs
    Parameters:
    ----------
    matrix : list of lists
        Matrix that contains lists as rows. One row represents the
        possibilities for an event to belong to a certain class
    pair : tuple(2)
        Pair of labels
    Returns:
    -------
    normed_pairs : list
        List of pairs (tuples(2))
    '''
    first = pair[0]
    second = pair[1]
    first_elems = [row[first] for row in matrix]
    second_elems = [row[second] for row in matrix]
    normed_pairs = []
    for elems in zip(first_elems, second_elems):
        normed = normalization(elems)
        normed_pairs.append(normed)
    return normed_pairs


def normalization(elems):
    '''Normalizes the pair
    Parameters:
    ----------
    elems : tuple(2)
        Pair of elements
    Returns:
    -------
    normed_1 : float
        Normed first element of the tuple
    normed_2 : float
        Normed second element of the tuple
    '''
    elem_sum = elems[0] + elems[1]
    normed_1 = elems[0] / elem_sum
    normed_2 = elems[1] / elem_sum
    return normed_1, normed_2


def create_mask(true_labels, pair):
    '''Creates mask with values in pair as the True values
    Parameters:
    ----------
    true_labels : list
        List of labels
    pair : tuple(2)
        The pair according to which the mask will be created
    Returns:
    -------
    bool_row : list
        The mask created with the values in 'pair' parameter as the True values
    '''
    bool_row = []
    for label in true_labels:
        if label in pair:
            bool_row.append(True)
        else:
            bool_row.append(False)
    return bool_row


def choose_values(matrix, pair, true_labels):
    '''Chooses values and creates appropriate normed pairs and masked labels
    Parameters:
    ----------
    matrix : list of lists
        Matrix that contains lists as rows. One row represents the
        possibilities for an event to belong to a certain class
    pair : tuple(2)
        Pair of labels
    true_labels : list
        List of labels

    Returns:
    -------
    masked_labels : numpy array
        numpy array of true labels with appropriate mask
    normed_pairs : list of tuples(2)
        List of normed tuples(2) for an appropriate pair
    '''
    mask = create_mask(true_labels, pair)
    masked_labels = np.array(true_labels)[mask]
    masked_matrix = np.array(matrix)[mask]
    normed_pairs = get_values(masked_matrix, pair)
    return masked_labels, normed_pairs


# def calculate_ROC(labels, normed_matrix):
#     thresholds = np.arange(0, 1, 0.05)
#     for threshold in thresholds:
