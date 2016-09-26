# This code helps in finding slope and intercept of regression line 
# closed form solution

import graphlab

#Loading the data using SFrame

sales = graphlab.SFrame('kc_house_data.gl/')

#splitting the data in to training data and test data

train_data,test_data = sales.random_split(.8,seed=0)



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
