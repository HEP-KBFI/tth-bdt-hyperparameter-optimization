'''Tools for a gradient descent algorithm for Rosenbrock function'''
from tthAnalysis.bdtHyperparameterOptimization import rosenbrock_tools as rt


def update_values(curr_values, gradients, learning_rate):
    new_values = {}
    for variable in curr_values:
        new_values[variable] = curr_values[variable] - (learning_rate * gradients[variable])
    return new_values


def calculate_gradient(curr_values, a, b):
    gradients = {}
    gradients['x'] = (-2 * (a - curr_values['x'])) - (4 * b * curr_values['x'] * (curr_values['y'] - (curr_values['x'] ** 2)))
    gradients['y'] = 2 * b * (curr_values['y'] - (curr_values['x'] ** 2))
    return gradients


def gradient_descent(settings, parameters, true_values):
    '''Performs the gradient descent optimization algorithm'''
    iteration = 0
    # Choose random values
    value_set = rt.initialize_values(parameters)
    while iteration <= settings['iterations'] or distance < 1e-8:
        print('Iteration: ' + str(iteration))
        curr_values = value_set
        # Calculate error
        # error = rt.ensemble_fitness([curr_values], true_values)
        # Calculate gradient
        gradients = calculate_gradient(curr_values, true_values['a'], true_values['b'])
        # Adjust values with gradients
        value_set = update_values(curr_values, gradients, settings['learning_rate'])
        distance = rt.check_distance(true_values, value_set)
        iteration += 1
