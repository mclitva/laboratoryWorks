import pandas as pd
from math import pow
from sklearn.datasets import load_boston 
 
def get_headers(dataframe):
    return dataframe.columns.values
 
 
def average(readings):
    readings_total = sum(readings)
    number_of_readings = len(readings)
    mean = readings_total / float(number_of_readings)
    return mean
 
 
def cal_variance(readings):
    readings_average = average(readings)
    mean_difference_squared_readings = [pow((reading - readings_average), 2) for reading in readings]
    variance = sum(mean_difference_squared_readings)
    return variance / float(len(readings) - 1)
 
 
def cal_covariance(readings_1, readings_2):
    readings_1_average = average(readings_1)
    readings_2_average = average(readings_2)
    readings_size = len(readings_1)
    covariance = 0.0
    for i in xrange(0, readings_size):
        covariance += (readings_1[i] - readings_1_average) * (readings_2[i] - readings_2_average)
    return covariance / float(readings_size - 1)
 
 
def cal_simple_linear_regression_coefficients(x_readings, y_readings):
    b1 = cal_covariance(x_readings, y_readings) / float(cal_variance(x_readings))
    b0 = average(y_readings) - (b1 * average(x_readings))
    return b0, b1
 
 
def predict_target_value(x, b0, b1):
    return b0 + b1 * x
 
 
def cal_rmse(actual_readings, predicted_readings):
    square_error_total = 0.0
    total_readings = len(actual_readings)
    for i in xrange(0, total_readings):
        error = predicted_readings[i] - actual_readings[i]
        square_error_total += pow(error, 2)
    rmse = square_error_total / float(total_readings)
    return rmse
 
 
def simple_linear_regression(dataset): 
    # Get the dataset header names
    dataset_headers = get_headers(dataset)
    print("Dataset Headers :: ", (dataset_headers))
 
    # Calculating the mean of the square feet and the price readings
    square_feet_mean = average(dataset[dataset_headers[0]])
    price_mean = average(dataset[dataset_headers[1]])
 
    square_feet_variance = cal_variance(dataset[dataset_headers[0]])
    price_variance = cal_variance(dataset[dataset_headers[1]])
 
    # Calculating the regression
    covariance_of_price_and_square_feet = dataset.cov()[dataset_headers[0]][dataset_headers[1]]
    w1 = covariance_of_price_and_square_feet / float(square_feet_variance)
 
    w0 = price_mean - (w1 * square_feet_mean)
 
    # Predictions
    dataset['Predicted_Price'] = w0 + w1 * dataset[dataset_headers[0]]
 
 
if __name__ == "__main__":
 
    input_path = '../Inputs/input_data.csv'
    house_price_dataset = pd.read_csv(input_path)
    simple_linear_regression(house_price_dataset)