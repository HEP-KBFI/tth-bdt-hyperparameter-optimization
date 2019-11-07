import numpy as np
import json
import os
import xgboost as xgb
from tthAnalysis.bdtHyperparameterOptimization import filereader as fr
from sklearn.metrics import confusion_matrix
from tthAnalysis.bdtHyperparameterOptimization import roccurve as rc


def initialize_values(value_dicts):
    sample = {}
    for xgb_params in value_dicts:
        if xgb_params['true_int'] == 'True':
            sample[xgb_params['p_name']] = np.random.randint(
                low=xgb_params['range_start'],
                high=xgb_params['range_end']
            )
        else:
            sample[xgb_params['p_name']] = np.random.uniform(
                low=xgb_params['range_start'],
                high=xgb_params['range_end']
            )
    return sample


def read_parameters(param_file):
    value_dicts = []
    with open(param_file, 'rt') as f:
        for line in f:
            json_dict = json.loads(line)
            value_dicts.append(json_dict)
    return value_dicts


def prepare_run_params(nthread, value_dicts, sample_size):
    run_params = []
    params = {
        'verbosity': 1,
        'objective': 'multi:softprob',
        'num_class': 10,
        'nthread': nthread,
        'seed': 1,
    }
    for i in range(sample_size):
        run_param = initialize_values(value_dicts)
        run_param.update(params)
        run_params.append(run_param)
    return run_params


def prepare_params_calc(value_dicts):
    keys_to_remove = [
        'verbosity',
        'objective',
        'num_class',
        'nthread',
        'seed'
    ]
    reduct_value_dicts = []
    try:
        z = value_dicts['num_boost_round']
        reduct_value_dict = value_dicts.copy()
        for key in keys_to_remove:
            try:
                reduct_value_dict.pop(key)
            except KeyError:
                pass
        return reduct_value_dict
    except TypeError:
        for value_dict in value_dicts:
            reduct_value_dict = value_dict.copy()
            for key in keys_to_remove:
                try:
                    reduct_value_dict.pop(key)
                except KeyError:
                    pass
            reduct_value_dicts.append(reduct_value_dict)
        return reduct_value_dicts


def best_to_file(best_values, outputDir, assesment):
    outputPath = os.path.join(outputDir, "best_parameters.json")
    with open(outputPath, "w") as f:
        json.dump(best_values, f)
        f.write('\n')
        json.dump(assesment, f)


def create_datasets(sample_dir, nthread):
    image_file = os.path.join(sample_dir, 'train-images-idx3-ubyte')
    label_file = os.path.join(sample_dir, 'train-labels-idx1-ubyte')
    training_images, training_labels = fr.read_dataset(image_file, label_file)
    image_file = os.path.join(sample_dir, 't10k-images-idx3-ubyte')
    label_file = os.path.join(sample_dir, 't10k-labels-idx1-ubyte')
    testing_images, testing_labels = fr.read_dataset(image_file, label_file)
    dtrain = xgb.DMatrix(
        np.asmatrix(training_images),
        label=training_labels,
        nthread=nthread
    )
    dtest = xgb.DMatrix(
        np.asmatrix(testing_images),
        label=testing_labels,
        nthread=nthread
    )
    data_dict = {
        'dtrain': dtrain,
        'dtest': dtest,
        'training_labels': training_labels,
        'testing_labels': testing_labels}
    return data_dict


def ensemble_fitnesses(parameter_dicts, data_dict):
    fitnesses = []
    pred_trains = []
    pred_tests = []
    for parameter_dict in parameter_dicts:
        fitness, pred_train, pred_test = parameter_evaluation(
            parameter_dict, data_dict)
        fitnesses.append(fitness)
        pred_trains.append(pred_train)
        pred_tests.append(pred_test)
    return fitnesses, pred_trains, pred_tests


def parameter_evaluation(parameter_dict, data_dict):
    parameters = parameter_dict.copy()
    num_boost_round = parameters.pop('num_boost_round')
    model = xgb.train(
        parameters,
        data_dict['dtrain'],
        num_boost_round=int(num_boost_round)
    )
    pred_train = model.predict(data_dict['dtrain'])
    pred_test = model.predict(data_dict['dtest'])
    # score = calculate_fitness(pred_train, pred_test, data_dict)
    prob_train, prob_test = get_most_probable(pred_train, pred_test)
    train_confusionMatrix, test_confusionMatrix = calculate_conf_matrix(
        prob_train, prob_test, data_dict)
    score = calculate_f1_score(test_confusionMatrix)[1]
    return score, pred_train, pred_test


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
    x_train, y_train = rc.roc(
        data_dict['training_labels'], result_dict['pred_train'])
    x_test, y_test = rc.roc(
        data_dict['testing_labels'], result_dict['pred_test'])
    test_AUC = np.trapz(y_test, x_test)
    train_AUC = np.trapz(y_train, x_train)
    assessment['train_AUC'] = (-1) * train_AUC
    assessment['test_AUC'] = (-1) * test_AUC
    if roc:
        rc.plotting(
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
    reduct_param = prepare_params_calc(parameter_dicts)
    keys = reduct_param[0].keys()
    list_dict = values_to_list_dict(keys, reduct_param)
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