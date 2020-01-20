import matplotlib.pyplot as plt
import numpy as np
import os
from tthAnalysis.bdtHyperparameterOptimization import universal


def plot_candlebar_covs(keys, covs, output_dir):
    errors = list(covs/2)
    output_path = os.path.join(output_dir, 'stability_check.png')
    fig, ax = plt.subplots()
    x_values = range(len(keys))
    x_range = x_values + [max(x_values)+1]
    x_range.insert(0, -1)
    new_labels = keys + [""]
    new_labels.insert(0, "")
    ax.errorbar(
        x_values, np.zeros(len(keys)), xerr=0, yerr=errors, linestyle='')
    ax.set_xticks(x_range)
    ax.set_xticklabels(new_labels)
    plt.xticks(rotation=-45)
    plt.title('Stability of parameters using COV')
    plt.savefig(output_path)
    plt.close('all')


def plot_individual(key, mean, stdev, parameter_values, output_dir):
    singles_dir = os.path.join(output_dir, 'single_variables')
    if not os.path.exists(singles_dir):
        os.makedirs(singles_dir)
    out_path = os.path.join(singles_dir, key + '_variation.png')
    parameter_count = len(parameter_values)
    x_values = range(parameter_count)
    ymax = 1.5 * max(parameter_values)
    ymin = 0.5 * min(parameter_values)
    plt.plot(x_values, parameter_values, color='r')
    plt.hlines(mean, 0, parameter_count-1, label='mean')
    plt.xlim(0, parameter_count-1)
    plt.ylim(ymin, ymax)
    plt.fill_between(
        [0,parameter_count],
        mean + stdev,
        mean - stdev,
        color='b',
        label='values',
        alpha=0.1)
    plt.title(key + '_variation')
    plt.xticks(x_values)
    ax = plt.gca()
    textstr = '\n'.join([
        r'$\mu=%.3f$' %(mean,),
        r'$\sigma=%.3f$' %(stdev,)
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=1)
    ax.text(
        0.9*x_values[-1], 1.0*ymax,
        textstr,
        bbox=props
    )
    plt.legend()
    plt.savefig(out_path)
    plt.close('all')


def stability_check_main(dict_of_parameter_lists, output_dir):
    keys = dict_of_parameter_lists.keys()
    covs = []
    means = []
    stdevs = []
    maximum = 0
    for key in keys:
        parameter_values = dict_of_parameter_lists[key]
        stdev = np.std(parameter_values)
        mean = np.mean(parameter_values)
        plot_individual(key, mean, stdev, parameter_values, output_dir)
        means.append(mean)
        stdevs.append(stdev)
        cov = stdev/mean
        covs.append(cov)
    covs = np.array(covs)
    plot_candlebar_covs(keys, covs, output_dir)
    plt.close('all')


def plot_radar_chart(parameter_set, ax):
    normalized_dict = normalize_values(parameter_set)
    labels = normalized_dict.keys()
    values = [normalized_dict[key] for key in labels]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.1)
    ax.set_thetagrids(angles * 180./np.pi, labels)
    gridlines = ax.yaxis.get_gridlines()
    for gl in gridlines:
        gl.get_path()._interpolation_steps = len(parameter_set.keys())


def plot_all_radar_charts(dict_of_parameter_lists, output_dir):
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    for parameter_set in dict_of_parameter_lists:
        plot_radar_chart(parameter_set, ax)
    ax.set_yticklabels([])
    ax.spines['polar'].set_visible(False)
    output_path = os.path.join(output_dir, 'radar_chart.png')
    plt.savefig(output_path)
    plt.close('all')


def normalize_values(parameter_set):
    normalized_dict = {}
    cmssw_base_path = os.path.expandvars('$CMSSW_BASE')
    param_file = os.path.join(
        cmssw_base_path,
        'src',
        'tthAnalysis',
        'bdtHyperparameterOptimization',
        'data',
        'xgb_parameters.json'
    )
    value_dicts = universal.read_parameters(param_file)
    for param_info in value_dicts:
        key = param_info['p_name']
        max_value = param_info['range_end']
        min_value = param_info['range_start']
        measured_value = parameter_set[key]
        normalized_value = float((measured_value - min_value)) / (max_value - min_value)
        normalized_dict[key] = normalized_value
    return normalized_dict


def plot_score_stability(score_dicts, output_dir, key='best_test_auc'):
    score_list = []
    for score_dict in score_dicts:
        value = score_dict[key]
        score_list.append(value)
    stdev = np.std(score_list)
    mean = np.mean(score_list)
    x_values = np.arange(len(score_list))
    maximum_y = max(score_list)
    plt.fill_between(x_values, mean + stdev, mean-stdev, alpha=0.1)
    plt.plot(x_values, score_list, c='k', linestyle='--')
    plt.hlines(mean, x_values[0], x_values[-1], color='r')
    plt.xlim(0, len(score_list)-1)
    ax = plt.gca()
    textstr = '\n'.join([
        r'$\mu=%.3f$' %(mean,),
        r'$\sigma=%.3f$' %(stdev,)
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=1)
    ax.text(
        0.9*x_values[-1], 1.0*maximum_y,
        textstr,
        bbox=props
    )
    output_path = os.path.join(output_dir, 'score_stability.png')
    plt.savefig(output_path)