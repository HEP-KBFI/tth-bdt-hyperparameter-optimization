import pandas
from sklearn.model_selection import train_test_split
import os
import numpy as np


def create_atlas_data_dict(path_to_file, global_settings):
    print('::: Loading data from ' + path_to_file + ' :::')
    path_to_file = os.path.expandvars(path_to_file)
    labels_to_drop = ['Kaggle', 'EventId', 'Weight']
    atlas_data_df = pandas.read_csv(path_to_file)
    atlas_data_df['Label'] = atlas_data_df['Label'].replace(
        to_replace='s', value=1)
    atlas_data_df['Label'] = atlas_data_df['Label'].replace(
        to_replace='b', value=0)
    for trainvar in atlas_data_df.columns:
        for label_to_drop in labels_to_drop:
            if label_to_drop in trainvar:
                try:
                    atlas_data_df = atlas_data_df.drop(trainvar, axis=1)
                except:
                    continue
    trainvars = list(atlas_data_df.columns)
    trainvars.remove('Label')
    output_dir = os.path.expandvars(global_settings['output_dir'])
    info_dir = os.path.join(output_dir, 'previous_files', 'data_dict')
    if not os.path.exists(info_dir):
        os.makedirs(info_dir)
    print(':::::::::::: Creating datasets ::::::::::::::::')
    train, test = train_test_split(
        atlas_data_df, test_size=0.2, random_state=1)
    training_labels = np.array(train['Label']).astype(int)
    testing_labels = np.array(test['Label']).astype(int)
    traindataset = np.array(train[trainvars].values)
    testdataset = np.array(test[trainvars].values)
    data_dict = {
        'dtrain': traindataset,
        'dtest': testdataset,
        'training_labels': training_labels,
        'testing_labels': testing_labels,
        'trainvars': trainvars
    }
    # universal.write_data_dict_info(info_dir, data_dict)
    return data_dict


def calculate_ams_score(signal, background, b_reg=10):
    the_sum = (signal + background + b_reg)
    log_part = 1 + (signal / (background + b_reg))
    return np.sqrt(2*(the_sum * np.log(log_part)- signal))