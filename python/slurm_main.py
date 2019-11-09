'''
To get the same output as in the normal case, call 'run_iteration'
'''
import numpy as np
import os
import json
import subprocess
import csv
import glob
from pathlib import Path
from shutil import copyfile
import time


def parameters_to_file(outputDir, parameter_dicts):
    samples = os.path.join(outputDir, 'samples')
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
    sample_dir,
    nthread,
    job_nr,
    outputDir,
    templateDir
):
    jobFile = os.path.join(outputDir, 'parameter_' + str(job_nr) + '.sh')
    template_file = os.path.join(templateDir, 'submit_template.sh')
    errorFile = os.path.join(outputDir, 'error')
    outputFile = os.path.join(outputDir, 'output')
    runScript = os.path.join(templateDir, 'slurm_fitness.py')
    copyfile(template_file, jobFile)
    with open(jobFile, 'a') as fh:
        fh.writelines('''
#SBATCH --cpus-per-task=%s
#SBATCH -e %s
#SBATCH -o %s
python %s --parameterFile %s --sample_dir %s --nthread %s
        ''' % (nthread, errorFile, outputFile, runScript,
            parameterFile, sample_dir, nthread))
    return jobFile


def run_job(jobFile):
    subprocess.call(['sbatch', jobFile])


def run_iteration(
    parameter_dicts,
    nthread,
    data_dict,
    outputDir,
    sample_dir,
    templateDir,
    sample_size
):
    parameters_to_file(outputDir, parameter_dicts)
    wild_card_path = os.path.join(outputDir, 'samples', '*', 'parameters.json')
    for parameterFile in glob.glob(wild_card_path):
        sample_nr = get_sample_nr(parameterFile)
        jobFile = prepare_jobFile(
            parameterFile, sample_dir, nthread,
            sample_nr, outputDir, templateDir
        )
        run_job(jobFile)
    wait_iteration(outputDir, sample_size)
    pred_tests = create_result_lists(outputDir, 'pred_test')
    pred_trains = create_result_lists(outputDir, 'pred_train')
    fitnesses = read_fitness(outputDir)
    delete_previous_files(outputDir)
    return fitnesses, pred_trains, pred_tests


def create_result_lists(outputDir, pred_type):
    samples = os.path.join(outputDir, 'samples')
    wild_card_path = os.path.join(samples, '*', pred_type + '.lst')
    ordering_list = []
    for path in glob.glob(wild_card_path):
        sample_nr = get_sample_nr(path)
        row_res = lists_from_file(path)
        ordering_list.append([sample_nr, row_res])
    ordering_list = sorted(ordering_list, key=lambda x: x[0])
    ordering_list = np.array([i[1] for i in ordering_list], dtype=float)
    return ordering_list


def read_fitness(outputDir):
    samples = os.path.join(outputDir, 'samples')
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
    parent_path = path1.parent
    sample_nr = int(parent_path.split('/')[-1])
    return sample_nr


def wait_iteration(outputDir, sample_size):
    wild_card_path = os.path.join(outputDir, 'samples', '*', '*.lst')
    while len(glob.glob(wild_card_path)) != sample_size*2:
        check_error(outputDir)
        time.sleep(5)


def lists_from_file(path):
    with open(path, 'r') as f:
        rows = csv.reader(f)
        row_res = []
        for row in rows:
            row_res.append(row)
    return row_res


def delete_previous_files(outputDir):
    wild_card_path1 = os.path.join(outputDir, 'samples', '*', '*.lst')
    wild_card_path2 = os.path.join(outputDir, 'samples', '*', '*.txt')
    wild_card_path = os.path.join(outputDir, '*.sh')
    for path in glob.glob(wild_card_path1):
        os.remove(path)
    for path in glob.glob(wild_card_path2):
        os.remove(path)


def check_error(outputDir):
    errorFile = os.path.join(outputDir, 'error')
    if os.path.exists(errorFile):
        with open(errorFile, 'r') as f:
            number_errors = len(f.readlines())
        if number_errors > 0:
            raise SystemExit(0)