'''Tools for a gradient descent algorithm for Rosenbrock function'''
from sympy import symbols
from tthAnalysis.bdtHyperparameterOptimization import rosenbrock_tools as rt


def rosenbrock_diff(a, b):
    x, y = symbols('x y')
    z = (a - x) ** 2 + b * (y - x ** 2) ** 2
    dx = z.diff(x)
    dy = z.diff(y)
    return dx, dy


def update_values(curr_values, gradients, learning_rate):
    new_values = {}
    for variable in curr_values:
        new_values[variable] = curr_values[variable] - (learning_rate * gradients[variable])
    return new_values


def calculate_gradient(dx, dy, curr_values):
    gradients = {}
    dx = dx.subs('x', curr_values['x'])
    gradients['x'] = float(dx.subs('y', curr_values['y']))
    dy = dy.subs('x', curr_values['x'])
    gradients['y'] = float(dy.subs('y', curr_values['y']))
    return gradients


def gradient_descent(settings, parameters, true_values):
    '''Performs the gradient descent optimization algorithm'''
    iteration = 0
    # Choose random values
    value_set = rt.initialize_values(parameters)
    # Calculate derivatives for the given function
    dx, dy = rosenbrock_diff(true_values['a'], true_values['b'])
    while iteration <= settings['iterations'] or distance < 1e-8:
        print('Iteration: ' + str(iteration))
        curr_values = value_set
        # Calculate error
        # error = rt.ensemble_fitness([curr_values], true_values)
        # Calculate gradient
        gradients = calculate_gradient(dx, dy, curr_values)
        # Adjust values with gradients
        value_set = update_values(curr_values, gradients, settings['learning_rate'])
        distance = rt.check_distance(true_values, value_set)
        iteration += 1
    return value_set
