#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

"""
Linear Regression with Gradient Descent
Author: dowusu
Date: Aug 24, 2020
"""

def extractData():
    """
    Extracts two-column data from file and returns two arrays.
    """
    data_file = open('ex1data1.txt', 'r')
    data = [line.split(',') for line in data_file.read().strip().split('\n')]
    x = np.array([float(entry[0]) for entry in data])
    y = np.array([float(entry[1]) for entry in data])
    return x, y

def plotLineData(x, y):
    """
    Line plots x and y array data.
    """
    figure = plt.figure()
    plt.plot(x, y, c='r')
    plt.xlabel('x data')
    plt.ylabel('y data')
    plt.title('Title')
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

def plotScatterData(x, y):
    """
    Scatter plots x and y array data.
    """
    figure = plt.figure()
    plt.scatter(x, y, c='r', marker='x')
    plt.xlabel('x data')
    plt.ylabel('y data')
    plt.title('Title')
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

def plotLineFit(x, y, theta):
    """
    Plots parameter line fit with scatter data.
    """
    num_examples = x.shape[0]
    ones = np.ones((num_examples,))
    X = np.vstack((x, ones)).T

    x_fit = X[:,0]
    y_fit = X.dot(theta)

    figure = plt.figure()
    plt.scatter(x, y, c='r', marker='x')
    plt.plot(x_fit, y_fit, c='b')
    plt.xlabel('x data')
    plt.ylabel('y data')
    plt.title('Title')
    manager = plt.get_current_fig_manager()
    manager.window.showMaximized()

    return X

def computeCost(X, y, theta, num_examples):
    """
    Computes cost for gradient descent (represents how close hypothesis is to
        actual output).

    X: design matrix, feature columns and training example rows
    y: output, column vector of training example outputs
    theta: parameters that determine hypothesis (predicted output)
    num_examples: number of training examples
    """
    hypothesis = X.dot(theta)
    return 1 / (2 * num_examples) * sum((hypothesis - y)**2)

def updateTheta(X, y, theta, learning_rate, num_examples):
    """
    Updates parameter values using gradient descent formula.

    X: design matrix, feature columns and training example rows
    y: output, column vector of training example outputs
    theta: parameters that determine hypothesis (predicted output)
    learning_rate: determines step size for computing new theta values
    num_examples: number of training examples
    """
    hypothesis = X.dot(theta)
    theta = theta - (learning_rate / num_examples) \
          * (X.transpose().dot((hypothesis - y)))
    return theta

def gradientDescent(x, y):
    """
    For linear regression, performs gradient descent for hard-coded number of
        iterations.

    x: array, one-variable input data (training data input)
    y: array, one-variable output data (training data output)
    """
    num_examples = x.shape[0]

    ones = np.ones((num_examples,))
    X = np.vstack((x, ones)).T
    theta = np.zeros((2,1))
    y = y.reshape(num_examples, 1)   
 
    num_iterations = 1500
    learning_rate = 0.01

    cost_array = np.array([])
    for i in range(num_iterations):
        theta = updateTheta(X, y, theta, learning_rate, num_examples)
        cost = computeCost(X, y, theta, num_examples)
        cost_array = np.append(cost_array, cost)
        print("Cost: %s" % cost)
    return cost_array, theta 

if __name__ == "__main__":
    x, y = extractData()
    plotScatterData(x, y)
    cost_array, theta = gradientDescent(x, y)
    X = plotLineFit(x, y, theta)
    plotLineData(range(len(cost_array)), cost_array)
