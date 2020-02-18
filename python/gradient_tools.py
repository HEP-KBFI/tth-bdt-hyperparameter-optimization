'''Tools for a gradient descent algorithm'''
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from tthAnalysis.bdtHyperparameterOptimization import rosenbrock_tools as rt


def rosenbrock(a, b=None, x=None, y=None, minimum=False):
    '''Calculates the Rosenbrock function for given variables or
    calculates its global minimum if the minimum option is set to True

    Parameters
    ----------
    a : float
        Value of the parameter a
    b : float
        Value of the parameter b
    x : float
        Value of the variable x
    y : float
        Value of the variable y
    minimum : bool
        If True, calculate the global minimum of the function;
        If False, calculate the function for given variable values

    Returns
    -------
    z : float
        Value of the function
    '''
    if minimum:
        return a, a ** 2
    z = ((a - x) ** 2) + (b * (y - x ** 2) ** 2)
    return z


def set_ranges(parameters):
    '''Set ranges for axis according to given parameters

    Parameters
    ----------
    parameters : list
        List of dictionaries of parameters for variables

    Returns
    -------
    x_range : range
        Range for the x-axis
    y_range : range
        Range for the y-axis
    '''
    for parameter in parameters:
        if parameter['p_name'] == 'x':
            x_range = range(
                parameter['range_start'],
                parameter['range_end']
            )
        if parameter['p_name'] == 'y':
            y_range = range(
                parameter['range_start'],
                parameter['range_end']
            )
    return x_range, y_range


# def update_values(curr_values, gradients, learning_rate):
#     '''Update variable values according to given gradients

#     Parameters
#     ----------
#     curr_values : dict
#         Current variable values
#     gradients : dict
#         Gradients corresponding to current variable values
#     learning_rate : float
#         Step size for updating values

#     Returns
#     -------
#     new_values : dict
#         Updated variable values
#     '''
#     new_values = {}
#     for variable in curr_values:
#         new_values[variable] = (curr_values[variable]
#                                 - (learning_rate * gradients[variable]))
#     return new_values


def find_steps(gradients, step_size):
    angle = math.atan(gradients['y'] / gradients['x'])
    x_step = step_size * math.cos(angle)
    y_step = step_size * math.sin(angle)
    steps = {
        'x': x_step,
        'y': y_step
    }
    return steps


def update_values(curr_values, gradients, step_size):
    new_values = {}
    steps = find_steps(gradients, step_size)
    for variable in curr_values:
        new_values[variable] = curr_values[variable] + steps[variable]
    return new_values


def calculate_gradient(curr_values, true_values, h, evaluate):
    '''Calculate the gradient for given variable values

    Parameters
    ----------
    curr_values : dict
        Given variable values
    true_values : dict
        Parameter values for the function
    h : float
        Step size for for numerically computing derivatives

    Returns
    -------
    gradient : dict
        Calculated gradient
    '''
    gradient = {}
    for variable in curr_values:
        temp_values = curr_values.copy()
        temp_values[variable] += h
        fxph = evaluate(
            temp_values, true_values['a'], true_values['b'])
        temp_values = curr_values.copy()
        temp_values[variable] -= h
        fxmh = evaluate(
            temp_values, true_values['a'], true_values['b'])
        gradient[variable] = (fxph - fxmh) / (2 * h)
    return gradient


def gradient_descent(
        settings,
        parameters,
        true_values,
        initialize=rt.initialize_values,
        evaluate=rt.parameter_evaluation,
        distance=rt.check_distance
):
    '''Performs the gradient descent optimization algorithm

    Parameters
    ----------
    settings : dict
        Settings for the algorithm, such as max interations, gamma
        and h
    parameters : dict
        Parameters of the given function
    true_values : dict
        Parameter values of the given function
    initialize : function
        Function for selecting the initial variable values
    distance : function
        Function for calculating the distance from the true global
        minimum

    Return
    ------
    result : dict
        Results of the algorithm
    '''
    iteration = 0
    result = {}
    # Choose random values
    value_set = initialize(parameters)
    print(value_set)
    while iteration <= settings['iterations'] or dist < 1e-8:
        if iteration % 10000 == 0:
            print('Iteration: ' + str(iteration))
        curr_values = value_set
        # Evaluate current values
        fitness = evaluate(
            curr_values, true_values['a'], true_values['b'])
        # Save data
        if iteration == 0:
            result['list_of_old_bests'] = [curr_values]
            result['list_of_best_fitnesses'] = [fitness]
            result['best_fitness'] = fitness
            result['best_parameters']= curr_values
        else:
            result['list_of_old_bests'].append(curr_values)
            result['list_of_best_fitnesses'].append(fitness)
        if fitness < result['best_fitness']:
            result['best_fitness'] = fitness
            result['best_parameters']= curr_values
        # Calculate gradient
        gradient = calculate_gradient(
            curr_values, true_values, settings['h'], evaluate)
        # Adjust values with gradients
        value_set = update_values(curr_values, gradient, settings['step_size'])
        # Calculate distance
        dist = distance(true_values, value_set)
        iteration += 1
    # Final evaluation
    fitness = evaluate(
            value_set, true_values['a'], true_values['b'])
    # Save data
    result['list_of_old_bests'].append(value_set)
    result['list_of_best_fitnesses'].append(fitness)
    if fitness < result['best_fitness']:
        result['best_fitness'] = fitness
        result['best_parameters'] = value_set
    print(result['best_parameters'])
    return result


def contourplot(
        result_dict,
        true_values,
        parameters,
        output_dir,
        function=rosenbrock
):
    '''Draws a contour plot of the function along with a marker for
    the global minimum and a line showing the progress of the algorithm

    Parameters
    ----------
    result_dict : dict
        Results of the gradient descent algorithm
    true_values : dict
        Parameter values for the function
    parameters : dict
        Parameters for the given function
    output_dir : string
        Path to the directory where to save the plot
    function : function
        Function used
    '''
    # Select ranges
    x_range, y_range = set_ranges(parameters)
    # Create grid
    x = np.arange(min(x_range), max(x_range) + 1)
    y = np.arange(min(y_range), max(y_range) + 1)
    x, y = np.meshgrid(x, y)
    # Calculate function values for each point on grid
    z = []
    for y_curr in y_range:
        z_curr = []
        for x_curr in x_range:
            z_curr.append(function(
                true_values['a'], true_values['b'], x_curr, y_curr))
        z.append(z_curr)
    z = np.array(z)
    # Plot the function
    plt.contour(x, y, z)
    # Plot minimum value
    x_min, y_min = function(true_values['a'], minimum=True)
    plt.plot(x_min, y_min, 'mo')
    # Plot the progress
    x_progress = []
    y_progress = []
    for values in result_dict['list_of_old_bests']:
        x_progress.append(values['x'])
        y_progress.append(values['y'])
    plt.plot(x_progress, y_progress)
    plt.xlim(min(x_range), max(x_range))
    plt.ylim(min(y_range), max(y_range))
    # Save plot
    plot_out = os.path.join(output_dir, 'contourplot.png')
    plt.savefig(plot_out, bbox_inches='tight')
    plt.close('all')
