

import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import sklearn

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression

boston = load_boston()
boston_frame = pd.DataFrame(boston.data)
boston_frame.columns = boston.feature_names
boston_frame['Price'] = boston.target

X = boston_frame.drop('Price', axis = 1)
lr = LinearRegression()
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, boston_frame.Price, test_size = 0.15, random_state = 5)
lr.fit(X_train, Y_train)
pred_train = lr.predict(X_train)
pred_test = lr.predict(X_test)

plot.scatter(pred_train, pred_train - Y_train, c = 'b', s = 40, alpha=0.5)
plot.scatter(pred_test, pred_test - Y_test, c = 'r', s = 40)
plot.hlines(y = 0, xmin = 0, xmax = 50)
plot.title('Blue points = train data, Red points = test data')
plot.ylabel('Residuals')
plot.show()