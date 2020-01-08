import matplotlib.pyplot as plt
import numpy as np
import os


def plot_candlebar_covs(keys, covs, output_dir):
    errors = list(covs/2)
    output_path = os.path.join(output_dir, 'stability_check.png')
    fig, ax = plt.subplots()
    x_values = range(len(keys))
    ax.errorbar(
        x_values, np.zeros(len(keys)), xerr=0.4, yerr=errors, linestyle='')
    ax.set_xticks(x_values)
    ax.set_xticklabels(keys)
    # plt.savefig(output_path)
    plt.title('Stability of parameters using COV')
    plt.show()


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
        label='values')
    plt.title(key + '_variation')
    plt.xticks(x_values)
    plt.legend()
    plt.savefig(out_path)


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
