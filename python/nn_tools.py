from keras.layers import Dense
# keras.layers.Dense(units, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
# need to specify the shape!!
# model = Sequential()
# model.add(Dense(32, input_shape=(16,)))

from keras.layers import Activation
# keras.layers.Activation(activation)
# see list of activation functions

from keras.layers import Dropout

# keras.layers.Dropout(rate, noise_shape=None, seed=None)
# Helps with overfitting, the rate is a fraction [0, 1]

from keras.layers import Flatten

# keras.layers.Flatten(data_format=None)

from keras.engine.input_layer import Input
# keras.engine.input_layer.Input()
# used to instantiate a Keras tensor

from keras import optimizers


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import StandardScaler
from keras.layers import Conv2D, Flatten, AlphaDropout, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU, ELU, PReLU # del better with zero activation -> avoid nan


#################################################################################

import tensorflow as tf
import keras
from keras import layers



from keras.layers import InputLayer, Input
from keras.layers import Reshape, MaxPooling2D
from keras.layers import Conv2D, Flatten, Dropout


#### ACTUALLY USED #####

from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.activations import elu
from keras.layers import ELU
from keras.optimizers import Nadam
# from keras.layers import Activation
from eli5.sklearn import PermutationImportance

# eli5.show_weights(perm, feature_names = X.columns.tolist())
# perm = PermutationImportance(my_model, random_state).fit(X, y)




def dataset():
    traindataset, valdataset  = train_test_split(data[varsBDT+["target", "key", "is_2016"]], test_size=0.2, random_state=7)


nn_hyperparameters = {
    'dropout_rate': dropout_rate,
    'learning_rate': learning_rate,
    'schedule_decay': schedule_decay,
    'nr_hidden_layers': nr_hidden_layers,
}



def create_nn_model(
        nn_hyperparameters,
        nr_trainvars,
        metrics=['accuracy'],
        ncat=3
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
    metrics : ['str'] (??)
        What metrics to use for model compilation
    [ncat] : int
        [Default: 3] Number of categories one wants the data to be classified.

    Returns:
    -------
    model : keras.engine.sequential.Sequential
        Sequential keras neural network model created.
    '''
    model = keras.models.Sequential()
    model.add(
        Dense(
            2*len(variablesNN),
            input_dim=len(variablesNN),
            kernel_initializer='he_uniform'
        )
    )
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Dropout(dropout_rate))
    for Nnodes in [16,8,8]: # TO BE OPTIMIZED
        model.add(Dense(Nnodes, kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(ELU())
        model.add(Dropout(dropout_rate))
    model.add(Dense(ncat, activation=elu))
    model.compile(
        loss='sparse_categorical_crossentropy', # TO BE OPTIMIZED, create my own?
        optimizer=Nadam(
            lr=learning_rate,
            schedule_decay=schedule_decay
        ),
        metrics=metrics, # WHAT METRICS TO USE?
    )
    return model


# Taken from XGBoost version
def parameter_evaluation(parameter_dict, data_dict, nthread, num_class):





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


### Which loss_function one should use



#### OTHER


# import os
# from tthAnalysis.bdtHyperparameterOptimization import universal
    # cmssw_base_path = os.path.expandvars('$CMSSW_BASE')
    # main_dir = os.path.join(
    #     cmssw_base_path,
    #     'src',
    #     'tthAnalysis',
    #     'bdtHyperparameterOptimization'
    # )
    # param_file = os.path.join(
    #     main_dir,
    #     'data',
    #     'xgb_parameters.json'
    # )
    # value_dicts = universal.read_parameters(param_file)