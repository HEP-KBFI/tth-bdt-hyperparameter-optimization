'''Tools for a gradient descent algorithm'''
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
    coordinates : dict
        Coordinates of the point
    value : float
        Value of the function on the given coordinates
    gradient : dict
        Gradient of the function at the given coordinates
    '''
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.value = None
        self.gradient = None
        self.an_gradient = None
    def assign_value(self, value):
        self.value = value
    def assign_gradient(self, gradient):
        self.gradient = gradient
    def assign_an_gradient(self, gradient):
        self.an_gradient = gradient


def gradient_wiggle(
        coordinates,
        curr_coordinate,
        true_values,
        step,
        evaluate,
        positive=True
):
    '''Evaluates how much does the function value change when changing
    the value of one of the coordinates

    Parameters
    ----------
    coordinates : dict
        Coordinates of the point
    curr_coordinate : string
        Coordinate whose value to change
    true_values : dict
        Parameter values of the function
    step : float
        Step size for changing the value of the coordinate
    evaluate : function
        Function for evaluating the given variable values and finding
        the function value
    positive : bool
       True for changing the coordinate value in the positive
       direction, False for changing the coordinate value in the
       negative direction

    Returns
    -------
    wiggle : float
        Function value at changed coordinates
    '''
    temp_values = coordinates.copy()
    if positive:
        temp_values[curr_coordinate] += step
    else:
        temp_values[curr_coordinate] -= step
    wiggle = evaluate(
        temp_values, true_values['a'], true_values['b'])
    return wiggle


def numerical_gradient(
        point,
        true_values,
        step,
        evaluate
):
    '''Numerically calculates the gradient for given point

    Parameters
    ----------
    point : Point
        Current point and its information
    true_values : dict
        Parameter values for the function
    step : float
        Step size for numerically computing derivatives
    evaluate : function
        Function for evaluating the given variable values and finding
        the function value

    Returns
    -------
    point : Point
        Current point with its gradient added
    '''
    gradient = {}
    for coordinate in point.coordinates:
        positive_wiggle = gradient_wiggle(
            point.coordinates, coordinate, true_values, step, evaluate)
        negative_wiggle = gradient_wiggle(
            point.coordinates,
            coordinate,
            true_values,
            step,
            evaluate,
            False
        )
        gradient[coordinate] = ((positive_wiggle - negative_wiggle)
                                / (2 * step))
    point.assign_gradient(gradient)
    return point


def analytical_gradient(point, true_values):
    '''Analytically calculate the gradient of the Rosenbrock function
    for given variable values

    Parameters
    ----------
    point : Point
        Current point and its information
    true_values : dict
        Parameter values for the Rosenbrock function

    Returns
    -------
    point : Point
        Current point with its analytical gradient added
    '''
    gradient = {}
    a = true_values['a']
    b = true_values['b']
    x = point.coordinates['x']
    y = point.coordinates['y']
    gradient['x'] = -2*a - 4*b*x*(-x**2 + y) + 2*x
    gradient['y'] = b*(-2*x**2 + 2*y)
    point.assign_an_gradient(gradient)
    return point


def calculate_steps(point, step):
    '''Calculates steps for x and y coordinates in case of a
    3-dimensional function

    Parameters
    ----------
    point : Point
        Current point and its information
    step : float
        Step size for updating coordinates

    Returns
    -------
    steps : dict
        Steps for all coordinates
    angle : float
        Direction of the gradient
    '''
    steps = {}
    angle = math.atan2(point.gradient['y'], point.gradient['x'])
    steps['x'] = step * math.cos(angle)
    steps['y'] = step * math.sin(angle)
    return steps, angle


def step_adjustment_check(point, new_point, true_values, step, evaluate):
    '''Checks whether the function value corresponds to the expected
    value according to gradient to avoid getting stuck in one
    location

    Parameters
    ----------
    point : Point
        Current point and its information
    new_point : Point
        New point according to currently chosen step
    true_values : dict
        Parameter values for the function
    step : float
        Current step size
    evaluate : function
        Function for evaluating the given variable values and finding
        the function value

    Returns
    -------
    new_point : Point
        New point according to chosen step size
    step : float
        Chosen step size
    '''
    expected_change = 0
    for variable in point.gradient:
        expected_change += point.gradient[variable] ** 2
    expected_change = step * math.sqrt(expected_change)
    actual_value = evaluate(
        new_point.coordinates, true_values['a'], true_values['b'])
    actual_change = point.value - actual_value
    if actual_change < 0.5 * expected_change:
        step /= 2
        new_point, angle, step = update_coordinates(
            point, true_values, step, evaluate)
        return new_point, step
    new_point.assign_value(actual_value)
    return new_point, step


def update_coordinates(point, true_values, step, evaluate):
    '''Update coordinates with a maximum step size according to
    its gradients

    Parameters
    ----------
    point : Point
        Current point and its information
    true_values : dict
        Parameter values for the function
    step : float
        Maximum step size for updating values
    evaluate : function
        Function for evaluating the given variable values and finding
        the function value

    Returns
    -------
    new_point : Point
        New point and its information
    angle : float
        Direction of the gradient
    step : float
        Size of the step that was taken
    new_values : dict
        Updated variable values
    '''
    new_coordinates = {}
    steps, angle = calculate_steps(point, step)
    for coordinate in point.coordinates:
        new_coordinates[coordinate] = (point.coordinates[coordinate]
                                       - steps[coordinate])
    new_point = Point(new_coordinates)
    new_point, step = step_adjustment_check(
        point, new_point, true_values, step, evaluate)
    return new_point, angle, step


def collect_history(iteration, point):
    '''Collects all information about the current iteration into a
    single dictionary

    Parameters
    ----------
    iteration : int
        Number of current iteration
    point : Point
        Current point and its information

    Returns
    -------
    curr_iteration : dict
        Dictionary with information about the current iteration
    '''
    curr_iteration = {
        'iteration': iteration,
        'x': point.coordinates['x'],
        'y': point.coordinates['y'],
        'z': point.value,
        'num_grad_x': point.gradient['x'],
        'num_grad_y': point.gradient['y'],
        'an_grad_x': point.an_gradient['x'],
        'an_grad_y': point.an_gradient['y'],
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
    steps = []
    # Choose random values
    coordinates = initialize(parameters)
    curr_point = Point(coordinates)
    # Calculate distance from true minimum
    dist = distance(true_values, coordinates)
    # Algorithm loop
    while (iteration <= settings['iterations']
           and dist > settings['step_size']):
        if iteration % 10000 == 0:
            print('Iteration: ' + str(iteration))
        # Calculate function value at current coordinates
        if not curr_point.value:
            fitness = evaluate(
                curr_point.coordinates, true_values['a'], true_values['b'])
            curr_point.assign_value(fitness)
        # Save data
        if iteration == 0:
            result['list_of_old_bests'] = [curr_point.coordinates]
            result['list_of_best_fitnesses'] = [curr_point.value]
            result['best_fitness'] = curr_point.value
            result['best_parameters'] = curr_point.coordinates
        else:
            result['list_of_old_bests'].append(curr_point.coordinates)
            result['list_of_best_fitnesses'].append(curr_point.value)
        if curr_point.value < result['best_fitness']:
            result['best_fitness'] = curr_point.value
            result['best_parameters'] = curr_point.coordinates
        # Calculate numerical gradient
        curr_point = numerical_gradient(
            curr_point, true_values, settings['h'], evaluate)
        # Calculate analytical gradient
        curr_point = analytical_gradient(curr_point, true_values)
        # Move point by step size
        new_point, angle, step_size = update_coordinates(
            curr_point, true_values, settings['step_size'], evaluate)
        # Finalize iteration
        angles.append(angle)
        steps.append(step_size)
        dist = distance(true_values, curr_point.coordinates)
        history.append(collect_history(iteration, curr_point))
        curr_point = new_point
        iteration += 1
    # Final evaluation
    fitness = evaluate(
        curr_point.coordinates, true_values['a'], true_values['b'])
    # Save data
    result['list_of_old_bests'].append(curr_point.coordinates)
    result['list_of_best_fitnesses'].append(fitness)
    if fitness < result['best_fitness']:
        result['best_fitness'] = fitness
        result['best_parameters'] = curr_point.coordinates
    result['angles'] = angles
    result['steps'] = steps
    result['history'] = history
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
    # Label axes
    plt.xlabel('x')
    plt.ylabel('y')
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
    plt.ylim(-math.pi, math.pi)
    plt.xlabel('Iteration number / #')
    plt.ylabel('Gradient angle')
    plot_out = os.path.join(output_dir, 'angle_plot.png')
    plt.savefig(plot_out, bbox_inches='tight')
    plt.close('all')


def step_plot(result_dict, output_dir):
    '''Draws a plot of the step sizes across all iterations

    Parameters
    ----------
    result_dict : dict
        Results of the gradient descent algorithm
    output_dir : string
        Path to the directory where to save the plot
    '''
    plt.plot(result_dict['steps'], '.', linestyle='')
    plt.yscale('log')
    plt.xlabel('Iteration number / #')
    plt.ylabel('Step size')
    plot_out = os.path.join(output_dir, 'step_plot.png')
    plt.savefig(plot_out, bbox_inches='tight')
    plt.close('all')
