import pandas as pd
from math import pow
import matplotlib.pyplot as plot
from sklearn.datasets import load_boston
 
def get_headers(dataframe):
    return dataframe.columns.values
 
 
def average(readings):
    readings_total = sum(readings)
    number_of_readings = len(readings)
    average = readings_total / float(number_of_readings)
    return average
 
 
def varianceOf(readings):
    readings_average = average(readings)
    mean_difference_squared_readings = [pow((reading - readings_average), 2) for reading in readings]
    variance = sum(mean_difference_squared_readings)
    return variance / float(len(readings) - 1)
 
 
def covarianceBetween(readings_1, readings_2):
    readings_1_average = average(readings_1)
    readings_2_average = average(readings_2)
    readings_size = len(readings_1)
    covariance = 0.0
    for i in xrange(0, readings_size):
        covariance += (readings_1[i] - readings_1_average) * (readings_2[i] - readings_2_average)
    return covariance / float(readings_size - 1)
 
 
def simpleLinearRegressionCoefficients(x_readings, y_readings):
    b1 = covarianceBetween(x_readings, y_readings) / float(varianceOf(x_readings))
    b0 = average(y_readings) - (b1 * average(x_readings))
    return b0, b1
 
 
def predict_target_value(x, b0, b1):
    return b0 + b1 * x
 
 
def rootMeanSquareError(actual_readings, predicted_readings):
    square_error_total = 0.0
    total_readings = len(actual_readings)
    for i in range(0, total_readings):
        error = predicted_readings[i] - actual_readings[i]
        square_error_total += pow(error, 2)
    rmse = square_error_total / float(total_readings)
    return rmse
 
 
def simple_linear_regression(dataset):
    dataset_headers = get_headers(dataset)

    square_feet_average = average(dataset[dataset_headers[0]])
    price_average = average(dataset[dataset_headers[1]])
 
    square_feet_variance = varianceOf(dataset[dataset_headers[0]])
    price_variance = varianceOf(dataset[dataset_headers[1]])
 
    # Calculating the regression
    covariance_of_price_and_square_feet = dataset.cov()[dataset_headers[0]][dataset_headers[1]]
    w1 = covariance_of_price_and_square_feet / float(square_feet_variance)
 
    w0 = price_average - (w1 * square_feet_average)
 
    # Predictions
    dataset['PredictedPrice'] = w0 + w1 * dataset[dataset_headers[0]]
    print(dataset[['PredictedPrice','Size']])
    print(rootMeanSquareError(dataset[dataset_headers[1]],dataset['PredictedPrice']))
    return dataset
    

 
 
if __name__ == "__main__":
    #dataset downloaded from https://wiki.csc.calpoly.edu/datasets/wiki/Houses
    input_path = 'RealEstate.csv'
    house_price_dataset = pd.read_csv(input_path)
    price_footage_dataset = house_price_dataset[['Size', 'Price']]
    calculatedDataset = simple_linear_regression(price_footage_dataset)
    plot.scatter(calculatedDataset.Price, calculatedDataset.Size, c = 'b', s = 20 , alpha = 0.5)
    plot.scatter(calculatedDataset.PredictedPrice, calculatedDataset.Size, c = 'r', s = 15)
    plot.title('Distribution of prices for houses depending on the size')
    plot.ylabel('Size in Foots')
    plot.xlabel('Price')
    plot.show()