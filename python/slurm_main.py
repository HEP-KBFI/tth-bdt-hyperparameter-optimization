'''Main functions to be used in slurm based fitness calculation

Note:
----
To get the same output as in the normal case, call 'run_iteration'
'''
from __future__ import division
import warnings
import time
from pathlib import Path
import os
import subprocess
import json
import csv
import glob
from shutil import copyfile
import shutil
import numpy as np
from tthAnalysis.bdtHyperparameterOptimization import universal
warnings.filterwarnings('ignore', category=DeprecationWarning)


def parameters_to_file(output_dir, parameter_dicts):
    '''Saves the parameters to the subdirectory (name=sample number) of the
    output_dir into a parameters.json file

    Parameters:
    ----------
    output_dir : str
        Path to the output directory
    parameter_dicts : list dicts
        Parameter-sets of all particles

    Returns:
    -------
    Nothing
    '''
    samples = os.path.join(output_dir, 'samples')
    if not os.path.exists(samples):
        os.makedirs(samples)
    for number, parameter_dict in enumerate(parameter_dicts):
        nr_sample = os.path.join(samples, str(number))
        if not os.path.exists(nr_sample):
            os.makedirs(nr_sample)
        parameter_file = os.path.join(nr_sample, 'parameters.json')
        with open(parameter_file, 'w') as file:
            json.dump(parameter_dict, file)


def prepare_job_file(
        parameter_file,
        sample_nr,
        global_settings
):
    '''Writes the job file that will be executed by slurm

    Parameters:
    ----------
    parameter_file : str
        Path to the parameter file
    sample_nr : int
        Number of the sample (parameter-set)
    global_settings : dict
        Global settings for the run

    Returns:
    -------
    job_file : str
        Path to the script to be executed by slurm
    '''
    cmssw_base_path = os.path.expandvars('$CMSSW_BASE')
    main_dir = os.path.join(
        cmssw_base_path,
        'src',
        'tthAnalysis',
        'bdtHyperparameterOptimization'
    )
    output_dir = os.path.expandvars(global_settings['output_dir'])
    template_dir = os.path.join(main_dir, 'data')
    job_file = os.path.join(output_dir, 'parameter_' + str(sample_nr) + '.sh')
    template_file = os.path.join(template_dir, 'submit_template.sh')
    error_file = os.path.join(output_dir, 'error')
    output_file = os.path.join(output_dir, 'output')
    file_title = '_'.join([
        'slurm', global_settings['ml_method'], global_settings['sample_type']])
    batch_job_file = file_title + '.py'
    run_script = os.path.join(main_dir, 'scripts', batch_job_file)
    copyfile(template_file, job_file)
    with open(job_file, 'a') as filehandle:
        filehandle.writelines('''
#SBATCH --cpus-per-task=%s
#SBATCH -e %s
#SBATCH -o %s
python %s --parameter_file %s --output_dir %s
        ''' % (global_settings['nthread'], error_file, output_file, run_script,
               parameter_file, output_dir))
    return job_file


def run_iteration(
        parameter_dicts,
        data_dict,
        global_settings,
        sample_size=0
):
    '''The main function call that is the slurm equivalent of ensemble_fitness
    in xgb_tools

    Parameters:
    ----------
    parameter_dicts : list of dicts
        Parameter-sets for all particles
    data_dict : dict
        Dummy-parameter. Added in order to have same interface with quasar
        and slurm version of fitness calculation. But in principle contains
        data and its labels
    global_settings : dict
        Global settings for the hyperparameter optimization
    sample_size: integer
        Sample size in case where it does not correspond to the value given
        in the settings file

    Returns:
    -------
    score_dicts : list of dicts
        List of the score_dicts for each parameter-set
    pred_trains : list of lists
        List of probabilities for an event to belong to a certain label for
        test dataset
    pred_tests : list of lists
        List of probabilities for an event to belong to a certain label for
        test dataset
    '''
    output_dir = os.path.expandvars(global_settings['output_dir'])
    previous_files_dir = os.path.join(output_dir, 'previous_files')
    if not os.path.exists(previous_files_dir):
        os.makedirs(previous_files_dir)
    settings_dir = os.path.join(output_dir, 'run_settings')
    if sample_size == 0:
        opt_settings = universal.read_settings(
            settings_dir, global_settings['optimization_algo'])
        sample_size = opt_settings['sample_size']
    parameters_to_file(output_dir, parameter_dicts)
    wild_card_path = os.path.join(output_dir, 'samples', '*', 'parameters.json')
    zero_sized = 1
    while zero_sized != 0:
        zero_sized = check_parameter_file_sizes(wild_card_path)
        time.sleep(2)
    for parameter_file in glob.glob(wild_card_path):
        sample_nr = get_sample_nr(parameter_file)
        job_file = prepare_job_file(
            parameter_file, sample_nr, global_settings
        )
        subprocess.call(['sbatch', job_file])
    wait_iteration(output_dir, sample_size)
    pred_tests = create_result_lists(output_dir, 'pred_test')
    pred_trains = create_result_lists(output_dir, 'pred_train')
    score_dicts = read_fitness(output_dir)
    feature_importances = read_feature_importances(output_dir)
    move_previous_files(output_dir, previous_files_dir)
    return score_dicts, pred_trains, pred_tests, feature_importances


def check_parameter_file_sizes(wild_card_path):
    '''Checks all files in the wild_card_path for their size. Returns the number
    of files with zero size

    Paramters:
    ---------
    wild_card_path : str
        Wild card path for glob to parse

    Returns:
    -------
    zero_sized : int
        Number of zero sized parameter files
    '''
    zero_sized = 0
    for parameter_file in glob.glob(wild_card_path):
        size = os.stat(parameter_file).st_size
        if size == 0:
            zero_sized += 1
    return zero_sized


def create_result_lists(output_dir, pred_type):
    '''Creates the result list that is ordered by the sample number

    Parameters:
    ----------
    output_dir : str
        Path to the directory of the output
    pred_type : str
        Type of the prediction (pred_test or pred_train)

    Returns:
    -------
    ordering_list : list
        Ordered list of the results
    '''
    samples = os.path.join(output_dir, 'samples')
    wild_card_path = os.path.join(samples, '*', pred_type + '.lst')
    ordering_list = []
    for path in glob.glob(wild_card_path):
        sample_nr = get_sample_nr(path)
        row_res = lists_from_file(path)
        ordering_list.append([sample_nr, row_res])
    ordering_list = sorted(ordering_list, key=lambda x: x[0])
    ordering_list = np.array([i[1] for i in ordering_list], dtype=float)
    return ordering_list


def read_fitness(output_dir):
    '''Creates the list of score dictionaries of each sample. List is ordered
    according to the number of the sample

    Parameters:
    ----------
    output_dir : str
        Path to the directory of output

    Returns:
    -------
    score_dicts : list of dicts
        List of score_dicts of each parameter-set
    '''
    samples = os.path.join(output_dir, 'samples')
    wild_card_path = os.path.join(samples, '*', 'score.json')
    number_samples = len(glob.glob(wild_card_path))
    score_dicts = []
    for number in range(number_samples):
        path = os.path.join(samples, str(number), 'score.json')
        score_dict = universal.read_parameters(path)[0]
        score_dicts.append(score_dict)
    return score_dicts


def read_feature_importances(output_dir):
    '''Reads the importances of all the parameter-sets

    Parameters:
    ----------
    output_dir : str
        Location where results were saved by the slurm script

    Returns:
    -------
    feature_importances : list of dicts
        Feature importances of all the parameter-sets
    '''
    samples = os.path.join(output_dir, 'samples')
    wild_card_path = os.path.join(samples, '*', 'feature_importances.json')
    number_samples = len(glob.glob(wild_card_path))
    feature_importances = []
    for number in range(number_samples):
        path = os.path.join(samples, str(number), 'feature_importances.json')
        feature_importance = universal.read_parameters(path)[0]
        feature_importances.append(feature_importance)
    return feature_importances


def get_sample_nr(path):
    '''Extracts the sample number from a given path

    Parameters:
    ----------
    path : str
        Path to the sample

    Returns : int
        Number of the sample
    '''
    path1 = Path(path)
    parent_path = str(path1.parent)
    sample_nr = int(parent_path.split('/')[-1])
    return sample_nr


def wait_iteration(output_dir, sample_size):
    '''Waits until all batch jobs are finised and in case of and warning
    or error that appears in the error file, stops running the optimization

    Parameters:
    ----------
    output_dir : str
        Path to the directory of output
    sample_size : int
        Number of particles (parameter-sets)

    Returns:
    -------
    Nothing
    '''
    wild_card_path = os.path.join(output_dir, 'samples', '*', '*.lst')
    while len(glob.glob(wild_card_path)) != sample_size*2:
        check_error(output_dir)
        time.sleep(5)


def lists_from_file(path):
    '''Creates a list from a file that contains data on different rows

    Parameters:
    ----------
    path : str
        Path to the file

    Returns:
    -------
    row_res : list
        List with the data
    '''
    with open(path, 'r') as file:
        rows = csv.reader(file)
        row_res = []
        for row in rows:
            row_res.append(row)
    return row_res


def move_previous_files(output_dir, previous_files_dir):
    '''Deletes the files from previous iteration

    Parameters:
    -------
    output_dir : str
        Path to the directory of the output

    Returns:
    -------
    Nothing
    '''
    iter_nr = find_iter_number(previous_files_dir)
    samples_dir = os.path.join(output_dir, 'samples')
    iter_dir = os.path.join(previous_files_dir, 'iteration_' + str(iter_nr))
    shutil.copytree(samples_dir, iter_dir)
    shutil.rmtree(samples_dir)
    wild_card_path = os.path.join(output_dir, 'parameter_*.sh')
    for path in glob.glob(wild_card_path):
        os.remove(path)


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


def check_error(output_dir):
    '''In case of warnings or errors during batch job that is written to the
    error file, raises SystemExit(0)

    Parameters:
    ----------
    output_dir : str
        Path to the directory of the output, where the error file is located

    Returns:
    -------
    Nothing
    '''
    number_errors = 0
    error_list = ['FAILED', 'CANCELLED', 'ERROR', 'Error']
    output_error_list = ['Usage']
    error_file = os.path.join(output_dir, 'error')
    output_file = os.path.join(output_dir, 'output')
    if os.path.exists(error_file):
        with open(error_file, 'rt') as file:
            lines = file.readlines()
            for line in lines:
                for error in error_list:
                    if error in line:
                        number_errors += 1
    if os.path.exists(output_file):
        with open(output_file, 'rt') as file:
            lines = file.readlines()
            for line in lines:
                for error in output_error_list:
                    if error in line:
                        number_errors += 1
    if number_errors > 0:
        print("Found errors: " + str(number_errors))
        raise SystemExit(0)


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
