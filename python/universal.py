'''Universal functions to be used in evolutionary algorithms
'''
from __future__ import division
import warnings
import itertools
import json
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
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


def calculate_fitness(
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
    score : float
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
    evaluation = score(train_score, test_score)
    return evaluation


def score(train_score, test_score):
    '''Calculates the final score by minimizing the difference of test and
    train score. [Currently not used anymore]

    Parameters:
    ----------
    train_score : float
        Score for the parameters when evaluating them on the train dataset
    test_score : float
        Score for the parameters when evaluating them on the train dataset

    Returns:
    -------
    evaluation : float
        Score for the set of parameters.
    '''
    evaluation = np.mean([
        (1 - (train_score - test_score)),
        (1 - (train_score - test_score)),
        test_score
    ])
    return evaluation


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
    '''
    true_test = data_dict['testing_labels']
    true_train = data_dict['training_labels']
    test_confusionmatrix = confusion_matrix(true_test, predicted_test)
    train_confusionmatrix = confusion_matrix(true_train, predicted_train)
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


def save_results(result_dict, output_dir, plot_roc=True):
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
    [plotROC=True] : bool
        Whether to plot ROC curve and average scores. Optional

    Returns:
    -------
    Nothing
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_dict = result_dict['data_dict']
    assessment = main_f1_calculate(
        result_dict['pred_train'],
        result_dict['pred_test'],
        result_dict['data_dict'])
    x_train, y_train = roc(
        data_dict['training_labels'], result_dict['pred_train'])
    x_test, y_test = roc(
        data_dict['testing_labels'], result_dict['pred_test'])
    test_auc = np.trapz(y_test, x_test)
    train_auc = np.trapz(y_train, x_train)
    assessment['train_AUC'] = (-1) * train_auc
    assessment['test_AUC'] = (-1) * test_auc
    if plot_roc:
        plotting(
            output_dir,
            x_train, y_train,
            x_test, y_test,
            result_dict['avg_scores']
        )
    best_to_file(
        result_dict['best_parameters'], output_dir, assessment)


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


def calculate_improvement_wSTDEV(parameter_dicts):
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
        x_train,
        y_train,
        x_test,
        y_test,
        avg_scores
):
    '''Main function for plotting costfunction and the ROC.

    Parameters:
    ----------
    output_dir : str
        Path to the directory where figures will be saved
    x_train : list
        List of false positives for given thresholds of the train sample
    y_train : list
        List of true positives for given thresholds of the train sample
    x_test : list
        List of false positives for given thresholds of the test sample
    y_test : list
        List of true positives for given thresholds of the test sample
    avg_scores : list
        List of average scores of all iterations

    Returns:
    -------
    Nothing
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plot_roc_curve(output_dir, x_train, y_train, x_test, y_test)
    plot_costfunction(avg_scores, output_dir)


def plot_roc_curve(
        output_dir,
        x_train,
        y_train,
        x_test,
        y_test
):
    '''Creates the ROC plot.

    Parameters:
    ----------
    output_dir : str
        Path to the directory where figures will be saved
    x_train : list
        List of false positives for given thresholds of the train sample
    y_train : list
        List of true positives for given thresholds of the train sample
    x_test : list
        List of false positives for given thresholds of the test sample
    y_test : list
        List of true positives for given thresholds of the test sample

    Returns:
    -------
    Nothing
    '''
    plot_out = os.path.join(output_dir, 'roc.png')
    plt.xlabel('Proportion of false values')
    plt.ylabel('Proportion of true values')
    axis = plt.gca()
    axis.set_aspect('equal')
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.plot(
        x_train, y_train, color='k', linestyle='--',
        label='optimized values, training data', zorder=100
    )
    plt.plot(
        x_test, y_test, color='r', linestyle='-',
        label='optimized values, testing data'
    )
    plt.tick_params(top=True, right=True, direction='in')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(plot_out)
    plt.close('all')


def plot_costfunction(avg_scores, output_dir):
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
        gen_numbers = np.arange(0, n_gens)
        plt.plot(gen_numbers, avg_scores, color='k')
        plt.xlim(0, n_gens - 1)
        plt.xticks(np.arange(n_gens - 1))
    except:  # in case of a genetic algorithm with multiple subpopulations
        for i in avg_scores.keys():
            n_gens = len(avg_scores[i])
            if i != 'final':
                gen_numbers = np.arange(0, n_gens)
                plt.plot(gen_numbers, avg_scores[i], color='b')
            if i == 'final':
                n_gens_final = n_gens + len(avg_scores[i]) - 1
                gen_numbers = np.arange(n_gens - 1, n_gens_final)
                plt.plot(gen_numbers, avg_scores[i], color='k')
        plt.xlim(0, n_gens_final - 1)
        plt.xticks(np.arange(n_gens_final - 1))
    finally:
        plt.xlabel('Generation')
        plt.ylabel('Fitness score')
        axis = plt.gca()
        axis.set_aspect('auto', adjustable='box')
        plt.grid(True)
        plt.tick_params(top=True, right=True, direction='in')
        plt.savefig(plot_out)
        plt.close('all')


def to_one_dict(list_of_dicts):
    main_dict = {}
    for elem in list_of_dicts:
        key = list(elem.keys())[0]
        main_dict[key] = elem[key]
    return main_dict


def getParameters(parameters_path):
    paramerList = read_parameters(parameters_path)
    parameter_dict = to_one_dict(paramerList)
    return parameter_dict


def read_settings(group):
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
    cmssw_base_path = os.path.expandvars('$CMSSW_BASE')
    settings_path = os.path.join(
        cmssw_base_path,
        'src'
        'tthAnalysis',
        'bdtHyperparameterOptimization',
        'data',
        group + '_settings.json')
    parameter_list = read_parameters(settings_path)
    settings_dict = to_one_dict(parameter_list)
    return settings_dict


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
