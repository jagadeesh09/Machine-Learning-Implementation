# This code helps in finding slope and intercept of regression line 
# closed form solution

import graphlab

#Loading the data using SFrame

sales = graphlab.SFrame('kc_house_data.gl/')

#splitting the data in to training data and test data

train_data,test_data = sales.random_split(.8,seed=0)

# Calculating slope and intercept using Closed Form Solution

def simple_linear_regression(input_feature, output):
    # compute the sum of input_feature and output
    input_sum = input_feature.sum()
    output_sum = output.sum()
    # compute the product of the output and the input_feature and its sum
    combine_product = input_feature * output
    combine_sum = combine_product.sum()
    # compute the squared value of the input_feature and its sum
    input_square = input_feature * input_feature
    square_sum = input_square.sum()
    num_features = input_feature.size()
    # use the formula for the intercept
    intercept = ((output_sum * square_sum) - (input_sum * combine_sum))/((num_features * square_sum) -(input_sum * input_sum))
    # use the formula for the slope
    slope = ((num_features * combine_sum)-(input_sum * output_sum))/((num_features * square_sum)-(input_sum *input_sum))
    return (intercept, slope)

# Getting predictions from the model using its slope and intercept
def get_regression_predictions(input_feature, intercept, slope):
    # calculating the predicted values:
    predicted_values = input_feature * slope + intercept
    return predicted_values

# Getting residual sum of squares from the target values using slope and intercept of the model by passing 
# the input feature vector
def get_residual_sum_of_squares(input_feature, output, intercept, slope):
    # First get the predictions
    prediction = input_feature * slope + intercept
    # then compute the residuals (since we are squaring it doesn't matter which order you subtract)
    residuals = prediction - output
    # square the residuals and add them up
    RSS = residuals * residuals
    RSS = RSS.sum()
    return(RSS)

# Predicting the feature from the given output
def inverse_regression_predictions(output, intercept, slope):
    # solve output = intercept + slope*input_feature for input_feature. Use this equation to compute the inverse predictions:
    estimated_feature = (output - intercept) / (slope)
    return estimated_feature

