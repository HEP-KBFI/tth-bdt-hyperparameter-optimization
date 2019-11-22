'''
To get the same output as in the normal case, call 'run_iteration'
'''
from __future__ import division
import numpy as np
import os
import json
import subprocess
import csv
import glob
from pathlib import Path
from shutil import copyfile
import time


def parameters_to_file(output_dir, parameter_dicts):
    samples = os.path.join(output_dir, 'samples')
    if not os.path.exists(samples):
        os.makedirs(samples)
    for nr, parameter_dict in enumerate(parameter_dicts):
        nr_sample = os.path.join(samples, str(nr))
        if not os.path.exists(nr_sample):
            os.makedirs(nr_sample)
        parameterFile = os.path.join(nr_sample, 'parameters.json')
        with open(parameterFile, 'w') as f:
            json.dump(parameter_dict, f)


def prepare_jobFile(
        parameterFile,
        sample_nr,
        global_settings
):
    cmssw_base_path = os.path.expandvars('$CMSSW_BASE')
    main_dir = os.path.join(
        cmssw_base_path,
        'tthAnalysis',
        'bdtHyperparameterOptimization'
    )
    output_dir = global_settings['output_dir']
    templateDir = os.path.join(main_dir, 'data')
    jobFile = os.path.join(output_dir, 'parameter_' + str(sample_nr) + '.sh')
    template_file = os.path.join(templateDir, 'submit_template.sh')
    errorFile = os.path.join(output_dir, 'error')
    outputFile = os.path.join(output_dir, 'output')
    batch_job_file = 'slurm_fitness.py'
    runScript = os.path.join(main_dir, 'scripts', batch_job_file)
    copyfile(template_file, jobFile)
    with open(jobFile, 'a') as fh:
        fh.writelines('''
#SBATCH --cpus-per-task=%s
#SBATCH -e %s
#SBATCH -o %s
python %s --parameterFile %s --sample_dir %s --nthread %s
        ''' % (global_settings['nthread'], errorFile, outputFile, runScript,
               parameterFile, global_settings['sample_dir'],
               global_settings['nthread']))
    return jobFile


def run_iteration(
        parameter_dicts,
        data_dict,
        global_settings
):
    output_dir = global_settings['output_dir']
    pso_settings = universal.read_settings('pso')
    parameters_to_file(output_dir, parameter_dicts)
    wild_card_path = os.path.join(output_dir, 'samples', '*', 'parameters.json')
    for parameterFile in glob.glob(wild_card_path):
        sample_nr = get_sample_nr(parameterFile)
        jobFile = prepare_jobFile(
            parameterFile, sample_nr, global_settings
        )
    subprocess.call(['sbatch', jobFile])
    wait_iteration(output_dir, pso_settings['sample_size'])
    pred_tests = create_result_lists(output_dir, 'pred_test')
    pred_trains = create_result_lists(output_dir, 'pred_train')
    fitnesses = read_fitness(output_dir)
    delete_previous_files(output_dir)
    return fitnesses, pred_trains, pred_tests


def create_result_lists(output_dir, pred_type):
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
    samples = os.path.join(output_dir, 'samples')
    wild_card_path = os.path.join(samples, '*', 'score.txt')
    fitnesses = []
    for path in glob.glob(wild_card_path):
        sample_nr = get_sample_nr(path)
        with open(path, 'r') as f:
            fitness = float(f.read())
        fitnesses.append([sample_nr, fitness])
    fitnesses = sorted(fitnesses, key=lambda x: x[0])
    fitnesses = np.array([i[1] for i in fitnesses], dtype=float)
    return fitnesses


def get_sample_nr(path):
    path1 = Path(path)
    parent_path = str(path1.parent)
    sample_nr = int(parent_path.split('/')[-1])
    return sample_nr


def wait_iteration(output_dir, sample_size):
    wild_card_path = os.path.join(output_dir, 'samples', '*', '*.lst')
    while len(glob.glob(wild_card_path)) != sample_size*2:
        check_error(output_dir)
        time.sleep(5)


def lists_from_file(path):
    with open(path, 'r') as f:
        rows = csv.reader(f)
        row_res = []
        for row in rows:
            row_res.append(row)
    return row_res


def delete_previous_files(output_dir):
    wild_card_path1 = os.path.join(output_dir, 'samples', '*', '*.lst')
    wild_card_path2 = os.path.join(output_dir, 'samples', '*', '*.txt')
    wild_card_path = os.path.join(output_dir, '*.sh')
    for path in glob.glob(wild_card_path1):
        os.remove(path)
    for path in glob.glob(wild_card_path2):
        os.remove(path)


def check_error(output_dir):
    errorFile = os.path.join(output_dir, 'error')
    if os.path.exists(errorFile):
        with open(errorFile, 'r') as f:
            number_errors = len(f.readlines())
        if number_errors > 0:
            raise SystemExit(0)


def save_info(score, pred_train, pred_test, save_dir):
    train_path = os.path.join(save_dir, "pred_train.lst")
    test_path = os.path.join(save_dir, "pred_test.lst")
    score_path = os.path.join(save_dir, "score.txt")
    with open(train_path, "w") as f:
        wr = csv.writer(f)
        wr.writerows(pred_train)
    with open(test_path, "w") as f:
        wr = csv.writer(f)
        wr.writerows(pred_test)
    with open(score_path, "w") as f:
        f.write(str(score))
