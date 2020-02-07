'''Tools for a gradient descent algorithm for Rosenbrock function'''
import matplotlib.pyplot as plt
from sympy import symbols
from tthAnalysis.bdtHyperparameterOptimization import rosenbrock_tools as rt


def rosenbrock(a, b):
    x, y = symbols('x y')
    z = (a - x) ** 2 + b * (y - x ** 2) ** 2
    return z


def prepare_gradient(function):
    x, y = symbols('x y')
    dx = function.diff(x)
    dy = function.diff(y)
    return dx, dy


def calculate_function(function, x, y):
    z = function.subs('x', x).subs('y', y)
    return z


def update_values(curr_values, gradients, learning_rate):
    new_values = {}
    for variable in curr_values:
        new_values[variable] = curr_values[variable] - (learning_rate * gradients[variable])
    return new_values


def calculate_gradient(dx, dy, curr_values):
    gradients = {}
    dx = calculate_function(dx, curr_values['x'], curr_values['y'])
    dy = calculate_function(dy, curr_values['x'], curr_values['y'])
    gradients['x'] = float(dx)
    gradients['y'] = float(dy)
    return gradients


def gradient_descent(settings, parameters, true_values):
    '''Performs the gradient descent optimization algorithm for the Rosenbrock function'''
    iteration = 0
    previous_values = []
    # Choose random values
    value_set = rt.initialize_values(parameters)
    # Calculate derivatives for the given function
    dx, dy = prepare_gradient(rosenbrock(true_values['a'], true_values['b']))
    while iteration <= settings['iterations'] or distance < 1e-8:
        print('Iteration: ' + str(iteration))
        curr_values = value_set
        print('Current values: ' + str(curr_values))
        previous_values.append(curr_values)
        # Calculate gradient
        gradients = calculate_gradient(dx, dy, curr_values)
        # Adjust values with gradients
        value_set = update_values(curr_values, gradients, settings['learning_rate'])
        # Calculate distance
        distance = rt.check_distance(true_values, value_set)
        iteration += 1
    x_range = range(-500, 500)
    y_range = x_range
    rosenbrock_contourplot(true_values, x_range, y_range, previous_values)
    return value_set


def rosenbrock_contourplot(true_values, x_range, y_range, previous_values):
    # Create grid
    x = np.arange(min(x_range), max(x_range) + 1)
    y = np.arange(min(y_range), max(y_range) + 1)
    x, y = np.meshgrid(x, y)
    # Calculate function values for each point on grid
    z = []
    for curr_y in y_range:
        curr_z = []
        for curr_x in x_range:
            curr_z.append((1 - curr_x) ** 2 + 100 * (curr_y - curr_x ** 2) ** 2)
        z.append(curr_z)
    z = np.array(z)
    # Plot the function
    plt.contour(x, y, z)   
    # Plot minimum value
    plt.plot([true_values['a']], [true_values['a'] ** 2], 'mo') 
    # Plot the progress
    progress_x = []
    progress_y = []
    for values in previous_values:
        progress_x.append(values['x'])
        progress_y.append(values['y'])
    plt.plot(progress_x, progress_y)
    plt.show()
