import graphlab
import numpy

sales = graphlab.SFrame('kc_house_data.gl/')

def polynomial_sframe(feature, degree):
    # assume that degree >= 1
    # initialize the SFrame:
    poly_sframe = graphlab.SFrame()
    # and set poly_sframe['power_1'] equal to the passed feature
    poly_sframe['power_1'] = feature
    # first check if degree > 1
    if degree > 1:
        # then loop over the remaining degrees:
        # range usually starts at 0 and stops at the endpoint-1. We want it to start at 2 and stop at degree
        for power in range(2, degree+1):
            tmp = feature.apply(lambda x: x**power)
            # first we'll give the column a name:
            name = 'power_' + str(power)
            # then assign poly_sframe[name] to the appropriate power of feature
            poly_sframe[name] = tmp
    return poly_sframe
    
def k_fold_cross_validation(k, l2_penalty, data, output_name, features_list):
    result = 0
    for i in range(k):
        n = len(data)
        start = (n*i)/k
        end = (n*(i+1))/k-1
        validation = data[start:end+1]
        first_two = data[0:start]
        last_two = data[end+1:n]
        train = first_two.append(last_two)
        model = graphlab.linear_regression.create(train,verbose=False,l2_penalty=l2_penalty, target = output_name, features = features_list, validation_set = None)
        predictions = model.predict(validation)
        result = result + graphlab.evaluation.rmse(predictions,validation[output_name])
    return result/k   
    
    
poly2_data = polynomial_sframe(sales['sqft_living'], 15)
my_features = poly2_data.column_names() # get the name of the features
poly2_data['price'] = sales['price'] # add price to the data since it's the target
model2 = graphlab.linear_regression.create(poly2_data,l2_penalty=l2_small_penalty, target = 'price', features = my_features, validation_set = None)
model2.get("coefficients").print_rows(num_rows=16)
plt.plot(poly2_data['power_1'],poly2_data['price'],'.',
        poly2_data['power_1'], model2.predict(poly2_data),'-')
