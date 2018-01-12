from sklearn.datasets import load_iris
from sklearn import preprocessing
import numpy

from MultiLayer import MLP


class Controller:
    # dataset
    data = []
    training = []
    testing = []

    # characteristics
    int_set_size = 0
    int_num_features = 0
    int_training_size = 30
    int_testing_size = 20
    int_num_per_class = 50
    int_num_classes = 0

    # composition
    obj_mlp = MLP

    # initializiation
    def __init__(self):
        iris = load_iris()
        self.prepareDataSet(iris)

    # normalize or scale data
    # divide data to training and testing
    def prepareDataSet(self, iris):
        self.int_num_classes = numpy.unique(iris.target).shape[0]

        self.int_set_size = iris.data.shape[0]
        self.int_num_features = iris.data.shape[1]

        ## normalize data
        self.data = preprocessing.normalize(iris.data)
        self.data = preprocessing.minmax_scale(self.data, (-1, 1))

        # load data in arrays
        for i in range(0, len(self.data)):
            Y = iris.target[i]
            X = numpy.append(self.data[i], 1)  ## bias input = 1
            check_i = i % self.int_num_per_class
            if check_i < self.int_training_size:
                self.training.append([X, Y])
            else:
                self.testing.append([X, Y])

    # play the MLP
    def playMLP(self):
        # allow to test all combinations of settings
        i = 1  ## number hidden layers
        step_epochs = 1000  ## number of epochs
        # To calculate the number of hidden nodes we use a general rule of: (Number of inputs + outputs) x 2/3
        k = 3  ## number of hidden neurons
        l = 0.0001  ## eta learning rate
        s = 0.1  ## step
        for j in range(0, 20):
            self.obj_mlp = MLP(self.int_num_features, self.int_num_classes, i, j * step_epochs, k, l)
            self.obj_mlp.train(self.training)
            self.obj_mlp.plotMSE()