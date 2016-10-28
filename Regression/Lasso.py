import graphlab
import numpy as np

sales = graphlab.SFrame('kc_house_data.gl/')
# In the dataset, 'floors' was defined with type string, 
# so we'll convert them to int, before using it below
sales['floors'] = sales['floors'].astype(int) 


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
    
def normalize_features(feature_matrix):
    norms = np.linalg.norm(feature_matrix,axis=0)
    return (feature_matrix/norms, norms)
def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    # compute prediction
    prediction = np.dot(feature_matrix,weights)
    # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
    ro_i = np.sum(feature_matrix[:,i] * (output-prediction + weights[i] *  feature_matrix[:,i]))

    if i == 0: # intercept -- do not regularize
        new_weight_i = ro_i 
    elif ro_i < -l1_penalty/2.:
        new_weight_i = ro_i + l1_penalty/2
    elif ro_i > l1_penalty/2.:
        new_weight_i = ro_i - l1_penalty/2
    else:
        new_weight_i = 0.
    
    return new_weight_i

def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    converged = False
    while (converged==False):
        change = [None] * len(initial_weights)
        for i in range(len(initial_weights)):
            old_weights = initial_weights[i]
            initial_weights[i] = lasso_coordinate_descent_step(i, feature_matrix, output, initial_weights, l1_penalty)
            change[i] = abs(initial_weights[i] - old_weights)
        maxchange = np.max(change)
        if(maxchange < tolerance):
            converged = True
    return initial_weights

train_data,test_data = sales.random_split(.8,seed=0)
all_features = ['bedrooms',
                'bathrooms',
                'sqft_living',
                'sqft_lot',
                'floors',
                'waterfront', 
                'view', 
                'condition', 
                'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built', 
                'yr_renovated']
(simple_feature_matrix, output) = get_numpy_data(train_data, all_features, my_output)
(normalized_simple_feature_matrix, simple_norms) = normalize_features(simple_feature_matrix) # normalize features

initial_weights = np.zeros(len(all_features)+1)
l1_penalty = 1e7
tolerance = 1.0
weights1e7 = lasso_cyclical_coordinate_descent(normalized_simple_feature_matrix, output,
                                            initial_weights, l1_penalty, tolerance)
predictions = np.dot(normalized_simple_feature_matrix,weights1e7)
errors = predictions -output
square = errors*errors
RSS = square.sum()
print(RSS)
print(weights1e7)
