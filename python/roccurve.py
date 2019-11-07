import numpy as np
import matplotlib.pyplot as plt
from tthAnalysis.bdtHyperparameterOptimization import global_functions
import itertools
import os


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

