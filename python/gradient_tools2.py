'''Temporary file with some work in progress ideas'''
from __future__ import division
import os
import json
import math
import warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore', category=FutureWarning)


class Point:
    '''A class used to represent a point of a given function

    Attributes
    ----------
    real_coordinates : dict
        Coordinates of the point on the default coordinate system
    real_value : float
        Value of the function on the given coordinates
    temp_coordinates : dict
        Coordinates of the point on a rotated coordinate system
    matrix : array
        Rotation matrix for convertion into the rotated system
    gradient : dict
        Gradient of the function at the given coordinates
    '''
    def __init__(self, coordinates):
        self.real_coordinates = coordinates
        self.real_value = None
        self.temp_coordinates = None
        self.matrix = None
        self.gradient = None
    def assign_value(self, value):
        self.real_value = value
    def assign_temp_coordinates(self, temp_coordinates):
        self.temp_coordinates = temp_coordinates
    def assign_matrix(self, matrix):
        self.matrix = matrix
    def assign_gradient(self, gradient):
        self.gradient = gradient


def real2temp(coordinates, matrix):
    '''Calculates coordinates in a rotated coordinate system based on coordinates
    in the default coordinate system

    Parameters
    ----------
    coordinates : dict
        Coordinates of the point in the default coordinate system
    matrix : array
        Rotation matrix

    Returns
    -------
    temp_coordinates : dict
        Coordinates of the point in a rotated coordinate system
    '''
    real = np.array([
        [coordinates['x']],
        [coordinates['y']]
    ])
    temp = np.matmul(matrix, real)
    temp_coordinates = {
        'x': temp[0][0],
        'y': temp[1][0]
    }
    return temp_coordinates


def temp2real(coordinates, matrix):
    '''Calculates coordinates in the default coordinate system based on
    coordinates in a rotated coordinate system

    Parameters
    ---------
    coordinates : dict
        Coordinates of the point in a rotated coordinate system
    matrix : array
        Rotation matrix

    Returns
    -------
    real_coordinates : dict
        Coordinates of the point in the default coordinate system
    '''
    temp = np.array([
        [coordinates['x']],
        [coordinates['y']]
    ])
    real = np.matmul(matrix.transpose(), temp)
    real_coordinates = {
        'x': real[0][0],
        'y': real[1][0]
    }
    return real_coordinates


def wiggle(coordinates, true_values, step, evaluate, rotated=False, matrix=None):
    '''Finds a numerical gradient of a function at a given coordinate
    by "wiggling" the coordinate values on each axis by a given step size

    Parameters
    ----------
    coordiantes : dict
        Coordinates of the point
    true_values : dict
        Parameter values of the function
    step : float
        Step size of the "wiggle"
    evaluate : function
        Function for evaluating the function at the given coordinates
    rotated : bool
        Whether the current coordinate system is rotated
    matrix : array
        Rotation matrix in case the coordinate system is rotated

    Returns
    -------
    gradient : dict
        Calculated gradient
    '''
    gradient = {}
    for coordinate in coordinates:
        temp_values = coordinates.copy()
        temp_values[coordinate] += step
        if rotated:
            temp_values = temp2real(temp_values, matrix)
        positive_wiggle = evaluate(
            temp_values, true_values['a'], true_values['b'])
        temp_values = coordinates.copy()
        temp_values[coordinate] -= step
        if rotated:
            temp_values = temp2real(temp_values, matrix)
        negative_wiggle = evaluate(
            temp_values, true_values['a'], true_values['b'])
        gradient[coordinate] = (positive_wiggle - negative_wiggle) / (2 * step)
    return gradient


def numerical_gradient(point, true_values, step, evaluate):
    '''Numerically calculates the gradient of the function at a given coordinate
    with the direction of the x-axis determined by the previous gradient

    Parameters
    ----------
    point : Point
        Object containing the current coordinates and coordinate system
    true_values : dict
        Parameter values of the function
    step : float
        Step size of the "wiggle"
    evaluate : function
        Function for evaluating the function at the given coordinates

    Returns
    -------
    point : Point
        Point updated with the gradient
    '''
    gradient = {}
    if not point.temp_coordinates:
        # print('Default system')
        gradient = wiggle(
            point.real_coordinates,
            true_values,
            step,
            evaluate
        )
    else:
        # print('Rotated system')
        gradient = wiggle(
            point.temp_coordinates,
            true_values,
            step,
            evaluate,
            True,
            point.matrix
        )
    point.assign_gradient(gradient)
    return point


def axes_rotation(point):
    '''Rotates the axes of the coordinate system in such a way that the new x-axis
    is directed in the opposite direction of the gradient

    Parameters
    ----------
    point : Point
        Object containing the current coordinates and coordinate system

    Returns
    -------
    point : Point
        Point updated with information about the rotated coordinate system
    '''
    angle = math.atan2(point.gradient['y'], point.gradient['x'])
    # print('Gradient angle: ' + str(angle))
    if angle < math.pi:
        angle += math.pi
    else:
        angle -= math.pi
    # print('Move angle: ' + str(angle))
    matrix = np.array([
        [math.cos(angle), math.sin(angle)],
        [-1 * math.sin(angle), math.cos(angle)]
    ])
    # print('Rotation matrix:')
    # print(matrix)
    point.assign_matrix(matrix)
    point.assign_temp_coordinates(real2temp(point.real_coordinates, point.matrix))
    # print('Rotated coordinates: ' + str(point.temp_coordinates))
    return point, angle


def update_point(temp_coordinates, old_point):
    '''Generates a new point based on the updated values of the previous point

    Parameters
    ----------
    temp_coordinates : dict
        Coordinates of the point in a rotated coordinate system
    old_point : Point
        Object containing information about the previous point

    Returns
    -------
    point : Point
        New point
    '''
    real_coordinates = temp2real(temp_coordinates, old_point.matrix)
    # print('New real coordinates: ' + str(real_coordinates))
    point = Point(real_coordinates)
    point.assign_matrix(old_point.matrix)
    point.assign_temp_coordinates(temp_coordinates)
    return point


def update_coordinates(point, true_values, step, evaluate):
    '''Moves the coordinates by a given step size in the direction of the x-axis
    in the rotated coordinate system

    Parameters
    ----------
    point : Point
        Object containing information about the current point
    true_values : dict
        Parameter values of the given function
    step : float
        Step size for moving the coordinates
    evaluate : function
        Function for evaluating the given variable values and finding
        the function value

    Returns
    -------
    new_point : Point
        Point with new coordinates
    '''
    new_values = {}
    for variable in point.temp_coordinates:
        if variable == 'x':
            new_values[variable] = point.temp_coordinates[variable] + step
        else:
            new_values[variable] = point.temp_coordinates[variable]
    # print('New rotated coordinates: ' + str(new_values))
    new_point = update_point(new_values, point)
    new_point = gradient_check(point, new_point, true_values, step, evaluate)
    return new_point


def gradient_check(point, new_point, true_values, step, evaluate):
    '''Checks whether the function value corresponds to the expected
    value according to gradient to avoid getting stuck in one
    location

    Parameters
    ----------
    point : Point
        Object containing information about the previous point
    new_point : 
        Object containing information about the new point
    true_values : dict
        Parameter values of the given function
    step : float
        Step size for moving the coordinates
    evaluate : function
        Function for evaluating the given variable values and finding
        the function value

    Returns
    -------
    new_point : Point
        Object containing information about the new point
    '''
    print('Step size: ' + str(step))
    expected_change = 0
    for variable in point.gradient:
        expected_change += point.gradient[variable] ** 2
    expected_change = step * math.sqrt(expected_change)
    # print('Expected change: ' + str(expected_change))
    fitness = evaluate(
        new_point.real_coordinates, true_values['a'], true_values['b'])
    new_point.assign_value(fitness)
    # print('Fitness: ' + str(fitness))
    # new_point.assign_value(evaluate(
    #     new_point.real_coordinates, true_values['a'], true_values['b']))
    actual_change = point.real_value - new_point.real_value
    # print('Actual change: ' + str(actual_change))
    if actual_change < 0.5 * expected_change:
        step /= 2
        new_point = update_coordinates(point, true_values, step, evaluate)
        return new_point
    else:
        return new_point


def collect_history(
        iteration,
        curr_values,
        num_gradient,
        func_value
):
    '''Collects all information about the current iteration into a
    single dictionary

    Parameters
    ----------
    iteration : int
        Number of current iteration
    curr_values : dict
        Variable values
    num_gradient : dict
        Numerically calculated gradient
    func_value : float
        Function value for given variable values

    Returns
    -------
    curr_iteration : dict
        Dictionary with information about the current iteration
    '''
    curr_iteration = {
        'iteration': iteration,
        'x': curr_values['x'],
        'y': curr_values['y'],
        'z': func_value,
        'num_grad_x': num_gradient['x'],
        'num_grad_y': num_gradient['y'],
    }
    return curr_iteration


def gradient_descent(
        settings,
        parameters,
        true_values,
        initialize,
        evaluate,
        distance
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
    evaluate : function
        Function for evaluating the given variable values and finding
        the function value
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
    history = []
    angles = []
    # Choose random initial values
    value_set = initialize(parameters)
    # print('Initial coordinates: ' + str(value_set))
    dist = distance(true_values, value_set)
    # Set chosen values as coordinates
    curr_point = Point(value_set)
    while (iteration <= settings['iterations']
           and dist > settings['step_size']):
        # Calculate funcion value at current coordinates
        curr_values = curr_point.real_coordinates
        # print('Initial coordinates: ' + str(curr_values))
        fitness = evaluate(
            curr_values, true_values['a'], true_values['b'])
        # print('Fitness: ' + str(fitness))
        curr_point.assign_value(fitness)
        # Calculate gradient
        curr_point = numerical_gradient(
            curr_point, true_values, settings['h'], evaluate)
        # print('Coordinates: ' + str(curr_point.real_coordinates))
        # print('Gradient: ' + str(curr_point.gradient))
        # Save data
        if iteration == 0:
            result['list_of_old_bests'] = [curr_values]
            result['list_of_best_fitnesses'] = [fitness]
            result['best_fitness'] = fitness
            result['best_parameters'] = curr_values
        else:
            result['list_of_old_bests'].append(curr_values)
            result['list_of_best_fitnesses'].append(fitness)
        if fitness < result['best_fitness']:
            result['best_fitness'] = fitness
            result['best_parameters'] = curr_values
        history.append(collect_history(
            iteration, curr_values, curr_point.gradient, fitness))
        # Rotation of axes
        curr_point, angle = axes_rotation(curr_point)
        angles.append(angle)
        # print('Rotation matrix: ')
        # print(curr_point.matrix)
        # print('Rotated coordinates: ' + str(curr_point.temp_coordinates))
        # Moving point by a designated step
        curr_point = update_coordinates(
            curr_point, true_values, settings['step_size'], evaluate)
        # Calculate distance to minimum
        dist = distance(true_values, curr_point.real_coordinates)
        iteration += 1
    value_set = curr_point.real_coordinates
    # Final evaluation
    fitness = evaluate(
        value_set, true_values['a'], true_values['b'])
    # Save data
    result['list_of_old_bests'].append(value_set)
    result['list_of_best_fitnesses'].append(fitness)
    if fitness < result['best_fitness']:
        result['best_fitness'] = fitness
        result['best_parameters'] = value_set
    result['history'] = history
    result['angles'] = angles
    return result


def write_history(result_dict, output_dir):
    '''Writes history of the run of the gradient descent algorithm
    into a json file

    Parameters
    ----------
    result_dict : dict
        Results of the algorithm
    output_dir : string
        Path to the directory where to save the plot
    '''
    history = result_dict['history']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'iteration_history.json')
    with open(output_path, 'w') as file:
        for line in history:
            json.dump(line, file)
            file.write('\n')


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


def rosenbrock(a, b=None, x=None, y=None, minimum=False):
    '''Calculates the Rosenbrock function for given variables or
    calculates its global minimum if the minimum option is set
    to True

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


def contourplot(
        result_dict,
        true_values,
        parameters,
        output_dir,
        function=rosenbrock
):
    '''Draws a contour plot of the function along with a marker for
    the global minimum and a line showing the progress of the
    algorithm

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
    plt.plot(x_progress, y_progress, '.', linestyle='')
    plt.xlim(min(x_range), max(x_range))
    plt.ylim(min(y_range), max(y_range))
    # Save plot
    plot_out = os.path.join(output_dir, 'contourplot.png')
    plt.savefig(plot_out, bbox_inches='tight')
    plt.close('all')


def angle_plot(result_dict, output_dir):
    '''Draws a plot of the gradient angles across all iterations

    Parameters
    ----------
    result_dict : dict
        Results of the gradient descent algorithm
    output_dir : string
        Path to the directory where to save the plot
    '''
    plt.plot(result_dict['angles'], '.', linestyle='')
    plt.ylim(0, 2 * math.pi)
    plot_out = os.path.join(output_dir, 'angle_plot.png')
    plt.savefig(plot_out, bbox_inches='tight')
    plt.close('all')
