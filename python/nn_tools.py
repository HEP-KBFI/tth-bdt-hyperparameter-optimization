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

#SDG
# sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(lostt='mean_squared_error', optimizer=sgd)
# Stochastic gradient descent = SGD
# lr: learning_rate float >= 0
# momentum: float >= 0. Accelerates SGD in the relevant direction and dampens oscillations
# nesterov: bool, whether to apply Nesteov momentum


#RMSprop
# keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
# recommended to only change lr
# rho float >= 0


# Xanda used Nadam optimizer:
# optimizer=Nadam(lr=0.00008, schedule_decay=0.00001), # , beta_1 = 0.95, beta_2 = 0.999
# metrics=['accuracy'], 
# )