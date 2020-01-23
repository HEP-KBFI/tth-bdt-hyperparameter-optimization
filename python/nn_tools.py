from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.activations import elu
from keras.layers import ELU
from keras.optimizers import Nadam
# from keras.layers import Activation
# from eli5.sklearn import PermutationImportance
import numpy as np
from keras import backend as K
import tensorflow as tf
import keras
from keras.wrappers.scikit_learn import KerasClassifier
import json


# eli5.show_weights(perm, feature_names = X.columns.tolist())
# perm = PermutationImportance(my_model, random_state).fit(X, y)


nn_hyperparameters = {
    'hidden_layer_dropout_rate': 0.5,
    'visible_layer_dropout_rate': 0.8,
    'learning_rate': 0.1,
    'schedule_decay': 0.002,
    'nr_hidden_layers': 2,
    'alpha': 8,
    'batch_size': 246,
    'epochs': 45
}



def create_nn_model(
        nn_hyperparameters,
        nr_trainvars,
        num_class,
        number_samples,
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
def parameter_evaluation(nn_hyperparameters, data_dict, nthread, num_class):
    K.set_session(
        tf.Session(
            config=tf.ConfigProto(
                intra_op_parallelism_threads=nthread,
                inter_op_parallelism_threads=nthread,
                allow_soft_placement=True,
            )
        )
    )
    nr_trainvars = len(data_dict['train'].columns)
    number_samples = len(data_dict['train'])
    nn_model = create_nn_model(
        nn_hyperparameters, nr_trainvars, num_class, number_samples)
    k_model  = KerasClassifier(
        build_fn=nn_model,
        epochs=nn_hyperparameters['epochs'],
        batch_size=nn_hyperparameters['batch_size'],
        verbose=2
    )
    fit_result = k_model.fit(
        data_dict['train'].values,
        data_dict['training_labels'].values,
        validation_data=(
            data_dict['test'].values,
            data_dict['testing_labels'].values
        )
    )



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


def create_data_dict(data, trainvars):
    '''Creates the data_dict to be used by the Neural Network

    Parameters:
    ----------
    data : pandas dataframe
        Dataframe containing the data
    trainvars : list
        List of names of the training variables

    Returns:
    -------
    data_dict : dict
        Dictionary containing training and testing labels
    '''
    print('::::::: Create datasets ::::::::')
    additions = ['target', 'totalWeight', 'process']
    variables = trainvars
    for addition in additions:
        if not addition in variables:
            variables = variables + [addition]
    train, test = train_test_split(
        data[variables],
        test_size=0.2, random_state=1
    )
    training_labels = train['target'].astype(int)
    testing_labels = test['target'].astype(int)
    training_processes = train['process']
    testing_processes = test['process']
    traindataset = np.array(train[trainvars].values)
    testdataset = np.array(test[trainvars].values)
    data_dict = {
        'train': traindataset,
        'test': testdataset,
        'training_labels': training_labels,
        'testing_labels': testing_labels,
        'training_processes': training_processes,
        'testing_processes': testing_processes
    }
    return data_dict



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
