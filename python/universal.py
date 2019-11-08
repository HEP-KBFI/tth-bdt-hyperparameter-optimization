import numpy as np
import json
import os
import xgboost as xgb
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix


def read_parameters(param_file):
    value_dicts = []
    with open(param_file, 'rt') as f:
        for line in f:
            json_dict = json.loads(line)
            value_dicts.append(json_dict)
    return value_dicts


def best_to_file(best_values, outputDir, assesment):
    outputPath = os.path.join(outputDir, "best_parameters.json")
    with open(outputPath, "w") as f:
        json.dump(best_values, f)
        f.write('\n')
        json.dump(assesment, f)


def calculate_fitness(
    pred_train, pred_test, data_dict
):
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
    return score(train_score, test_score)


def score(train_score, test_score):
    score = np.mean([
        (1 - (train_score - test_score)),
        (1 - (train_score - test_score)),
        test_score
    ])
    return score


def calculate_conf_matrix(pred_train, pred_test, data_dict):
    true_test = data_dict['testing_labels']
    true_train = data_dict['training_labels']
    test_confusionMatrix = confusion_matrix(true_test, pred_test)
    train_confusionMatrix = confusion_matrix(true_train, pred_train)
    return train_confusionMatrix, test_confusionMatrix


def calculate_f1_score(confusionMatrix):
    nr_labels = len(confusionMatrix)
    labels = np.arange(0, nr_labels)
    num_elem = confusionMatrix.sum()
    gs = []
    f1s = []
    for label in labels:
        fp = 0
        fn = 0
        new_labels = np.delete(labels, label)
        tp = confusionMatrix[label, label]
        for nl in new_labels:
            fn += confusionMatrix[label, nl]
            fp += confusionMatrix[nl, label]
        tn = num_elem - fp - fn - tp
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        # F1 -> harmonic mean of prec & recall
        f1 = 2 * (precision * recall) / (precision + recall)
        # G -> geometric mean of prec & recall
        g = np.sqrt(precision * recall)
        f1s.append(f1)
        gs.append(g)
    mean_f1 = np.mean(f1s)
    mean_g = np.mean(gs)
    return mean_f1, mean_g


def get_most_probable(pred_train, pred_test):
    new_test = []
    new_train = []
    for vector in pred_test:
        new_test.append(np.argmax(vector))
    for vector in pred_train:
        new_train.append(np.argmax(vector))
    return new_train, new_test


def main_f1_calculate(pred_train, pred_test, data_dict):
    pred_train, pred_test = get_most_probable(pred_train, pred_test)
    train_confusionMatrix, test_confusionMatrix = calculate_conf_matrix(
        pred_train, pred_test, data_dict)
    train_f1_score, train_g_score = calculate_f1_score(train_confusionMatrix)
    test_f1_score, test_g_score = calculate_f1_score(test_confusionMatrix)
    assessment = {
        'Train_F1': train_f1_score,
        'Train_G': train_g_score,
        'Test_F1': test_f1_score,
        'Test_G': test_g_score
    }
    return assessment


def save_results(result_dict, outputDir, roc=True):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    data_dict = result_dict['data_dict']
    assessment = main_f1_calculate(
        result_dict['pred_train'],
        result_dict['pred_test'],
        result_dict['data_dict'])
    x_train, y_train = roc(
        data_dict['training_labels'], result_dict['pred_train'])
    x_test, y_test = roc(
        data_dict['testing_labels'], result_dict['pred_test'])
    test_AUC = np.trapz(y_test, x_test)
    train_AUC = np.trapz(y_train, x_train)
    assessment['train_AUC'] = (-1) * train_AUC
    assessment['test_AUC'] = (-1) * test_AUC
    if roc:
        plotting(
            outputDir,
            x_train, y_train,
            x_test, y_test,
            result_dict['avg_scores']
        )
    best_to_file(
        result_dict['best_parameters'], outputDir, assessment)


def calculate_improvement_wAVG(avg_scores, improvements, threshold):
    if len(avg_scores) > 1:
        improvements.append((avg_scores[-1]-avg_scores[-2])/avg_scores[-2])
        improvement = improvements[-1]
    if len(improvements) < 2:
        improvement = 1
    elif improvement <= threshold:
        improvement = improvements[-2]
    return improvements, improvement


def calculate_improvement_wSTDEV(parameter_dicts):
    keys = parameter_dicts[0].keys()
    list_dict = values_to_list_dict(keys, parameter_dicts)
    mean_COV = calculate_dict_mean_coeff_of_variation(list_dict)
    return mean_COV


def values_to_list_dict(keys, parameter_dicts):
    list_dict = {}
    for key in keys:
        list_dict[key] = []
        for parameter_dict in parameter_dicts:
            list_dict[key].append(parameter_dict[key])
    return list_dict


def calculate_dict_mean_coeff_of_variation(list_dict):
    coeff_of_variations = []
    for key in list_dict:
        values = list_dict[key]
        coeff_of_variation = np.std(values)/np.mean(values)
        coeff_of_variations.append(coeff_of_variation)
    mean_coeff_of_variation = np.mean(coeff_of_variations)
    return mean_coeff_of_variation


def roc(labels, pred_vectors):
    thresholds = np.arange(0, 1, 0.01)
    y = []
    x = []
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
        y.append(float(sig_score)/len(labels))
        x.append(float(bg_score)/(9*len(labels)))
    return x, y


def plotting(
    outputDir,
    x_train,
    y_train,
    x_test,
    y_test,
    avg_scores
):
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    plot_roc(outputDir, x_train, y_train, x_test, y_test)
    plot_costFunction(avg_scores, outputDir)


def plot_roc(
    outputDir,
    x_train,
    y_train,
    x_test,
    y_test
):
    plotOut = os.path.join(outputDir, 'roc.png')
    plt.xlabel('Proportion of false values')
    plt.ylabel('Proportion of true values')
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.xlim(0.0,0.1)
    plt.ylim(0.9,1.0)
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
    plt.savefig(plotOut)
    plt.close('all')


def plot_costFunction(avg_scores, outputDir):
    plotOut = os.path.join(outputDir, 'costFunction.png')
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    try:
        n_gens = len(avg_scores)
        x = np.arange(0, n_gens)
        plt.plot(x, avg_scores, color='k')
        plt.xlim(0, n_gens - 1)
        plt.xticks(np.arange(n_gens - 1))
    except: #in case of a genetic algorithm with multiple subpopulations
        for i in avg_scores.keys():
            n_gens = len(avg_scores[i])
            if i != 'final':
                x = np.arange(0, n_gens)
                plt.plot(x, avg_scores[i], color='b')
            if i == 'final':
                n_gens_final = n_gens + len(avg_scores[i]) - 1
                x = np.arange(n_gens - 1, n_gens_final)
                plt.plot(x, avg_scores[i], color='k')
        plt.xlim(0, n_gens_final - 1)
        plt.xticks(np.arange(n_gens_final - 1))
    finally:
        plt.xlabel('Generation')
        plt.ylabel('Fitness score')
        ax = plt.gca()
        ax.set_aspect('auto', adjustable='box')
        plt.grid(True)
        plt.tick_params(top=True, right=True, direction='in')
        plt.savefig(plotOut)
        plt.close('all')


'''
One-vs-All ROC
'''

def create_pairs(matrix):
    row = np.arange(len(matrix[0]))
    combinations = itertools.combinations(row, 2)
    pairs = []
    for pair in combinations:
        pairs.append(pair)
    return pairs


def get_values(matrix, pair):
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
    elem_sum = elems[0] + elems[1]
    normed_1 = elems[0] / elem_sum
    normed_2 = elems[1] / elem_sum
    return normed_1, normed_2


def create_mask(true_labels, pair):
    bool_row = []
    for label in true_labels:
        if label in pair:
            bool_row.append(True)
        else:
            bool_row.append(False)
    return bool_row


def choose_values(matrix, pair, true_labels):
    mask = create_mask(true_labels, pair)
    masked_labels = np.array(true_labels)[mask]
    masked_matrix = np.array(matrix)[mask]
    normed_matrix = get_values(masked_matrix, pair)
    return masked_labels, normed_matrix


# def calculate_ROC(labels, normed_matrix):
#     thresholds = np.arange(0, 1, 0.05)
#     for threshold in thresholds:

