from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.activations import elu
from keras.layers import ELU
from keras.optimizers import Nadam
# from keras.layers import Activation
from eli5.sklearn import PermutationImportance
import numpy as np
from keras import backend as K
import tensorflow as tf
import keras
from keras.wrappers.scikit_learn import KerasClassifier
import json
from eli5.formatters.as_dataframe import format_as_dataframe
from tthAnalysis.bdtHyperparameterOptimization import universal


# eli5.show_weights(perm, feature_names = X.columns.tolist())
# perm = PermutationImportance(my_model, random_state).fit(X, y)


def create_nn_model(
        nn_hyperparameters={},
        nr_trainvars=9,
        num_class=3,
        number_samples=5000,
        metrics=['accuracy'],
):
    ''' Creates the neural network model. The normalization used is 
    batch normalization. Kernel is initialized by the Kaiming initializer
    called 'he_uniform'

    Parameters:
    ----------
    nn_hyperparameters : dict
        Dictionary containing the hyperparameters for the neural network. The
        keys contained are ['dropout_rate', 'learning_rate', 'schedule_decay',
        'nr_hidden_layers']
    nr_trainvars : int
        Number of training variables, will define the number of inputs for the
        input layer of the model
    num_class : int
        Default: 3 Number of categories one wants the data to be classified.
    number_samples : int
        Number of samples in the training data
    metrics : ['str']
        What metrics to use for model compilation

    Returns:
    -------
    model : keras.engine.sequential.Sequential
        Sequential keras neural network model created.
    '''
    model = keras.models.Sequential()
    model.add(
        Dense(
            2*nr_trainvars,
            input_dim=nr_trainvars,
            kernel_initializer='he_uniform'
        )
    )
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dropout(nn_hyperparameters['visible_layer_dropout_rate']))
    hidden_layers = create_hidden_net_structure(
        nn_hyperparameters['nr_hidden_layers'],
        num_class,
        nr_trainvars,
        number_samples,
        nn_hyperparameters['alpha']
    )
    for hidden_layer in hidden_layers: # TO BE OPTIMIZED
        model.add(Dense(hidden_layer, kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(ELU())
        model.add(Dropout(nn_hyperparameters['hidden_layer_dropout_rate']))
    model.add(Dense(num_class, activation=elu))
    model.compile(
        loss='sparse_categorical_crossentropy', # TO BE OPTIMIZED, create my own?
        optimizer=Nadam(
            lr=nn_hyperparameters['learning_rate'],
            schedule_decay=nn_hyperparameters['schedule_decay']
        ),
        metrics=metrics, # WHAT METRICS TO USE?
    )
    return model


# Taken from XGBoost version
def parameter_evaluation(
        nn_hyperparameters,
        data_dict,
        nthread,
        num_class,
        return_true_feature_importances=True
):
    K.set_session(
        tf.Session(
            config=tf.ConfigProto(
                intra_op_parallelism_threads=nthread,
                inter_op_parallelism_threads=nthread,
                allow_soft_placement=True,
            )
        )
    )
    nr_trainvars = len(data_dict['train'][0])
    number_samples = len(data_dict['train'])
    nn_model = create_nn_model(
        nn_hyperparameters, nr_trainvars, num_class, number_samples)
    k_model  = KerasClassifier(
        build_fn=create_nn_model,
        epochs=nn_hyperparameters['epochs'],
        batch_size=nn_hyperparameters['batch_size'],
        verbose=2,
        nn_hyperparameters=nn_hyperparameters,
        nr_trainvars=nr_trainvars,
        num_class=num_class,
        number_samples=number_samples
    )
    fit_result = k_model.fit(
        data_dict['train'],
        data_dict['training_labels'],
        validation_data=(
            data_dict['test'],
            data_dict['testing_labels']
        )
    )
    if return_true_feature_importances:
        feature_importance = get_feature_importances(
            k_model, data_dict)
    else:
        feature_importance = {}
    pred_train = k_model.predict_proba(data_dict['train'])
    pred_test = k_model.predict_proba(data_dict['test'])
    score_dict = universal.get_scores_dict(pred_train, pred_test, data_dict)
    return score_dict, pred_train, pred_test, feature_importance


def ensemble_fitnesses(parameter_dicts, data_dict, global_settings):
    '''Finds the data_dict, pred_train and pred_test for all particles

    Parameters:
    ----------
    parameter_dicts : list of dicts
        Parameter-sets of all particles
    data_dict : dict
        Dictionary that contains the labels for testing and training. Keys are
        called 'testing_labels' and 'training_labels'
    global_settings : dict
        Global settings for the run.

    Returns:
    -------
    score_dicts : list
        List of score_dicts of each parameter-set
    pred_trains : list of lols
        List of pred_trains
    pred_tests : list of lols
        List of pred_tests
    '''
    score_dicts = []
    pred_trains = []
    pred_tests = []
    feature_importances = []
    for parameter_dict in parameter_dicts:
        score_dict, pred_train, pred_test, feature_importance = parameter_evaluation(
            parameter_dict, data_dict,
            global_settings['nthread'], global_settings['num_classes'])
        score_dicts.append(score_dict)
        pred_trains.append(pred_train)
        pred_tests.append(pred_test)
        feature_importances.append(feature_importance)
    return score_dicts, pred_trains, pred_tests, feature_importances


def get_feature_importances(model, data_dict):
    perm = PermutationImportance(model).fit(
        data_dict['train'], data_dict['training_labels'])
    weights = eli5.explain_weights(perm, feature_names=data_dict['trainvars'])
    weights_df = format_as_dataframe(weights).sort_values(
        by='weight', ascending=False).rename(columns={'weight': 'score'})
    list_of_dicts = weights_df.to_dict('records')
    feature_importances = {}
    for single_variable_dict in list_of_dicts:
        key = single_variable_dict['feature']
        feature_importances[key] = single_variable_dict['score']
    return feature_importances


def calculate_number_nodes_in_hidden_layer(
    number_classes,
    number_trainvars,
    number_samples,
    alpha
):
    '''Calculates the number of nodes in a hidden layer

    Parameters:
    ----------
    number_classes : int
        Number of classes the data is to be classified to.
    number_trainvars : int
        Number of training variables aka. input nodes for the NN
    number_samples : int
        Number of samples in the data
    alpha : float
        number of non-zero weights for each neuron

    Returns:
    -------
    number_nodes : int
        Number of nodes in each hidden layer

    Comments:
    --------
    Formula used: N_h = N_s / (alpha * (N_i + N_o))
    N_h: number nodes in hidden layer
    N_s: number samples in train data set
    N_i: number of input neurons (trainvars)
    N_o: number of output neurons
    alpha: usually 2, but some reccomend it in the range [5, 10]
    '''
    number_nodes = number_samples / (
        alpha * (number_trainvars + number_classes)
    )
    return number_nodes


def create_hidden_net_structure(
    number_hidden_layers,
    number_classes,
    number_trainvars,
    number_samples,
    alpha=2
):
    '''Creates the hidden net structure for the NN

    Parameters:
    ----------
    number_hidden_layers : int
        Number of hidden layers in our NN
    number_classes : int
        Number of classes the data is to be classified to.
    number_trainvars : int
        Number of training variables aka. input nodes for the NN
    number_samples : int
        Number of samples in the data
    [alpha] : float
        [Default: 2] number of non-zero weights for each neuron

    Returns:
    -------
    hidden_net : list
        List of hidden layers with the number of nodes in each.
    '''
    number_nodes = calculate_number_nodes_in_hidden_layer(
        number_classes,
        number_trainvars,
        number_samples,
        alpha
    )
    number_nodes = int(np.floor(number_nodes/number_hidden_layers))
    hidden_net = [number_nodes] * number_hidden_layers
    return hidden_net



def initialize_values(value_dicts):
    '''Initializes the parameters according to the value dict specifications

    Parameters:
    ----------
    value_dicts : list of dicts
        Specifications how each value should be initialized

    Returns:
    -------
    sample : list of dicts
        Parameter-set for a particle
    '''
    sample = {}
    for hyper_params in value_dicts:
        if bool(hyper_params['true_int']):
            sample[str(hyper_params['p_name'])] = np.random.randint(
                low=hyper_params['range_start'],
                high=hyper_params['range_end']
            )
        else:
            sample[str(hyper_params['p_name'])] = np.random.uniform(
                low=hyper_params['range_start'],
                high=hyper_params['range_end']
            )
    return sample



def prepare_run_params(value_dicts, sample_size):
    ''' Creates parameter-sets for all particles (sample_size)

    Parameters:
    ----------
    value_dicts : list of dicts
        Specifications how each value should be initialized
    sample_size : int
        Number of particles to be created

    Returns:
    -------
    run_params : list of dicts
        List of parameter-sets for all particles
    '''
    run_params = []
    for i in range(sample_size):
        run_param = initialize_values(value_dicts)
        run_params.append(run_param)
    return run_params
