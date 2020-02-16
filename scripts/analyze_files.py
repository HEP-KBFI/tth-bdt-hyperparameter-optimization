import glob
import os
from tthAnalysis.bdtHyperparameterOptimization import universal
from tthAnalysis.bdtHyperparameterOptimization import slurm_main as sm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def parse_iterations(main_dir):
    best_particles = []
    compactnesses = []
    previous_files_dir = os.path.join(main_dir, 'previous_files')
    data_dict_dir_path = os.path.join(previous_files_dir, 'data_dict')
    data_dict = create_data_dict(data_dict_dir_path)
    wild_card_iterations = os.path.join(previous_files_dir, 'iteration_*')
    ordered_iteration_nrs = list(range(len(glob.glob(wild_card_iterations))-1))
    for iter_nr in ordered_iteration_nrs:
        iter_dir = os.path.join(previous_files_dir, 'iteration_' + str(iter_nr))
        best_particle, compactness = analyze_iteration(iter_dir)
        compactnesses.append(compactness)
        best_particles.append(best_particle)
    result_dict = create_result_dict(best_particles, data_dict)
    result_dict['compactnesses'] = compactnesses
    return result_dict



def analyze_iteration(iter_dir):
    particle_evaluations = []
    parameters_list = []
    wild_card_particle = os.path.join(iter_dir, '*')
    ordered_particle_nr = list(range(len(glob.glob(wild_card_particle))))
    for particle_nr in ordered_particle_nr:
        particle_dir = os.path.join(iter_dir, str(particle_nr))
        particle_evaluation = get_particle_evaluation(particle_dir)
        parameters = particle_evaluation['parameters']
        parameters_list.append(parameters)
        particle_evaluations.append(particle_evaluation)
    best_particle = get_best_particle(particle_evaluations)
    compactness = universal.calculate_compactness(parameters_list)
    return best_particle, compactness


def get_particle_evaluation(particle_dir):
    pred_train_path = os.path.join(
        particle_dir, 'pred_train.lst')
    pred_test_path = os.path.join(
        particle_dir, 'pred_test.lst')
    score_path = os.path.join(
        particle_dir, 'score.json')
    parameter_path = os.path.join(
        particle_dir, 'parameters.json')
    feature_importances_path = os.path.join(
        particle_dir, 'feature_importances.json')
    pred_train = np.array(sm.lists_from_file(pred_train_path), dtype=float)
    pred_test = np.array(sm.lists_from_file(pred_test_path), dtype=float)
    score_dict = universal.read_parameters(score_path)[0]
    feature_importances = universal.read_parameters(feature_importances_path)[0]
    parameters = universal.read_parameters(parameter_path)[0]
    particle_evaluation = {
        'pred_train': pred_train,
        'pred_test': pred_test,
        'score_dict': score_dict,
        'feature_importances': feature_importances,
        'parameters': parameters
    }
    return particle_evaluation


def get_best_particle(particle_evaluations):
    best_score = 0
    for particle_evaluation in particle_evaluations:
        optimizer = particle_evaluation['score_dict']['d_roc_1.5']
        if optimizer > best_score:
            best_particle = particle_evaluation
    return best_particle


def create_result_dict(best_particles, data_dict):
    last_iter_best = best_particles[-1]
    auc_info = universal.calculate_auc(
        data_dict,
        last_iter_best['pred_train'],
        last_iter_best['pred_test']
    )[-1]
    evolutions = get_evolutions(best_particles)
    result_dict = {
        'score_dict': last_iter_best['score_dict'],
        'feature_importances': last_iter_best['feature_importances'],
        'best_parameters': last_iter_best['parameters'],
        'auc_info': auc_info,
        'evolutions': evolutions
    }
    return result_dict


def create_data_dict(data_dict_dir_path):
    training_labels_path = os.path.join(data_dict_dir_path, 'training_labels.txt')
    testing_labels_path = os.path.join(data_dict_dir_path, 'testing_labels.txt')
    testing_labels = get_list_file(testing_labels_path)
    training_labels = get_list_file(training_labels_path)
    data_dict = {
        'testing_labels': testing_labels,
        'training_labels': training_labels
    }
    return data_dict


def get_list_file(in_path):
    elements = []
    with open(in_path, 'rt') as in_file:
        for element in in_file:
            elements.append(int(element.strip('\n')))
    return elements


def get_evolutions(best_particles):
    keys = best_particles[0]['score_dict'].keys()
    evolutions = {}
    for key in keys:
        evolutions[key] = []
        for best_particle in best_particles:
            score_dict = best_particle['score_dict']
            evolutions[key].append(score_dict[key])
    return evolutions


def plotting_and_save_info(result_dict, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    universal.plot_roc_curve(output_dir, result_dict['auc_info'])
    plot_scoring_metrics(result_dict['evolutions'], output_dir)
    universal.best_to_file(
        result_dict['best_parameters'], output_dir, result_dict['score_dict'])
    universal.plot_costfunction(
        result_dict['compactnesses'], output_dir, y_label='Compactness (cov)')


def plot_scoring_metrics(evolutions, output_dir):
    plot_out = os.path.join(output_dir, 'roc_curve.png')
    keys = evolutions.keys()
    n_gens = len(evolutions[keys[0]])
    iteration_nr = range(n_gens)
    for key in keys:
        plt.plot(iteration_nr, evolutions[key], label=key)
    plt.xlim(0, n_gens - 1)
    plt.xticks(np.arange(n_gens - 1))
    plt.xlabel('Iteration number / #')
    plt.ylabel('scoring_metric')
    axis = plt.gca()
    axis.set_aspect('auto', adjustable='box')
    axis.xaxis.set_major_locator(ticker.AutoLocator())
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.tick_params(top=True, right=True, direction='in')
    plt.savefig(plot_out)
    plt.close('all')


def main(main_dir, output_dir):
    result_dict = parse_iterations(main_dir)
    plotting_and_save_info(result_dict, output_dir)
