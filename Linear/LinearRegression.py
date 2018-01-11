

import numpy as np
import pandas as pd
import scipy.stats as stats
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
lr.fit(X, boston_frame.Price)
plot.scatter(boston_frame.RM, boston_frame.Price)
plot.xlabel("Number of rooms per dwelling")
plot.ylabel('Price')
plot.title('Relationship between count of rooms and Price')
plot.show()


plot.scatter(boston_frame.Price, lr.predict(X))
plot.xlabel("Prices: $Y_i$")
plot.ylabel('Predicted prices: $\hat{Y}_i$')
plot.title('Prices vs predictedPrices: $Y_i$ vs $\hat{Y}_i$')
plot.show()