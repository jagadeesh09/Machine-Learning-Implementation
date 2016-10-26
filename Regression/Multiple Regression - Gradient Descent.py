import graphlab
import numpy as np
from math import sqrt 

sales = graphlab.SFrame('kc_house_data.gl/')

""" Converting an SFrame to Numpy array """
def get_numpy_data(data_sframe, features, output):
    data_sframe['constant'] = 1 # this is how you add a constant column to an SFrame
    # add the column 'constant' to the front of the features list so that we can extract it along with the others:
    features = ['constant'] + features # this is how you combine two lists
    # select the columns of data_SFrame given by the features list into the SFrame features_sframe (now including constant):
    feature_sframe = graphlab.SFrame()
    for feature in features:
        feature_sframe[feature] = data_sframe[feature]
    # the following line will convert the features_SFrame into a numpy matrix:
    feature_matrix = feature_sframe.to_numpy()
    # assign the column of data_sframe associated with the output to the SArray output_sarray
    output_sarray = data_sframe[output]
    # the following will convert the SArray into a numpy array by first converting it to a list
    output_array = output_sarray.to_numpy()
    return(feature_matrix, output_array)
    
def predict_output(feature_matrix, weights):
    # assume feature_matrix is a numpy matrix containing the features as columns and weights is a corresponding numpy array
    # create the predictions vector by using np.dot()
    predictions = np.dot(feature_matrix,weights)
    return(predictions)
    
def feature_derivative(errors, feature):
    # Assume that errors and feature are both numpy arrays of the same length (number of data points)
    # compute twice the dot product of these vectors as 'derivative' and return the value
    derivative = 2 * np.dot(errors ,feature)
    return(derivative)
    
def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False 
    weights = np.array(initial_weights) # make sure it's a numpy array
    while not converged:
        # compute the predictions based on feature_matrix and weights using your predict_output() function
        predictions = np.dot(feature_matrix,weights)
        # compute the errors as predictions - output
        errors = predictions-output
        gradient_sum_squares = 0 # initialize the gradient sum of squares
        # while we haven't reached the tolerance yet, update each feature's weight
        for i in range(len(weights)): # loop over each weight
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            # compute the derivative for weight[i]:
            derivative = feature_derivative(errors,feature_matrix[:,i])
            # add the squared value of the derivative to the gradient sum of squares (for assessing convergence)
            gradient_sum_squares += derivative * derivative
            # subtract the step size times the derivative from the current weight
            weights[i] = weights[i] -  step_size * derivative
        # compute the square-root of the gradient sum of squares to get the gradient matnigude:
        gradient_magnitude = sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)
def get_residual_sum_of_squares(model, data, outcome):
    # First get the predictions
    predictions = model.predict(data)
    # Then compute the residuals/errors
    residual = predictions-outcome
    # Then square and add them up
    sq_residuals = residual * residual
    RSS = sq_residuals.sum()
    return(RSS)       
    
train_data,test_data = sales.random_split(.8,seed=0)
model_features = ['sqft_living', 'sqft_living15'] # sqft_living15 is the average squarefeet for the nearest 15 neighbors. 
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
initial_weights = np.array([-100000., 1., 1.])
step_size = 4e-12
tolerance = 1e9
weights = regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance)
(test_simple_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)
predictions = predict_output(test_simple_feature_matrix,weigh)
Residuals = predictions - test_output
Residualsquares = Residuals * Residuals
print (Residualsquares.sum())

