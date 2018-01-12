import numpy
from math import exp
from matplotlib import pyplot


class MLP:
    # properties
    int_num_epochs = 1000
    int_num_input_neurons = 0
    int_num_output_neurons = 0
    int_num_hidden_layers = 0
    int_num_hidden_neurons = 0
    dbl_mse_threshold = 0.001
    dbl_eta = 0.0001
    dbl_bias = 0.0002
    dbl_w0 = 0.0002

    arr_num_neurons_in_hidden_layers = []

    # weights
    wo = []
    wh = []

    # mse
    arr_mse = []

    ## initializations
    def __init__(self, _int_num_input_neurons, _int_num_output_neurons, _int_num_hidden_layers, _int_num_epochs,
                 _int_num_hidden_neurons, _dbl_eta):
        ## initialize properties
        self.int_num_input_neurons = _int_num_input_neurons
        self.int_num_output_neurons = _int_num_output_neurons
        self.int_num_hidden_layers = _int_num_hidden_layers
        self.int_num_epochs = _int_num_epochs
        self.int_num_hidden_neurons = _int_num_hidden_neurons
        self.dbl_eta = _dbl_eta

        ## initialize weights arrays
        self.wo = [[self.dbl_w0 for x in range(self.int_num_hidden_neurons + 1)] for y in
                   range(self.int_num_output_neurons)]  ## bias +1
        self.wh = [[self.dbl_w0 for x in range(self.int_num_input_neurons + 1)] for y in
                   range(self.int_num_hidden_neurons)]  ## bias +1
        return

    ## back-propagation-algorithm
    def train(self, training_set):
        ## loop epochs
        mse = []
        for e in range(0, self.int_num_epochs):
            ## loop training set
            errors = []
            for t in range(0, len(training_set)):
                # inputs
                x = training_set[t][0]

                # response
                d = training_set[t][1]

                # forward path
                actual_hidden_output = self.hyberb(numpy.inner(self.wh, x))

                actual_hidden_output_plus_rshp = numpy.reshape(actual_hidden_output, (self.int_num_hidden_neurons))
                actual_hidden_output_plus_bias = numpy.append(actual_hidden_output_plus_rshp, self.dbl_bias)

                actual_output = self.hyberb(numpy.inner(self.wo, actual_hidden_output_plus_bias))

                ## Question?! Why substract actual_output[3x1] Vector from scalar d = {0, 1, 2}
                error = d - actual_output

                ## erros will be used for mse
                errors = numpy.append(errors, error)

                ## backward path
                error_signal_output = error * self.derivhyberb(numpy.inner(self.wo, actual_hidden_output_plus_bias))

                error_signal_output_rshp = numpy.reshape(error_signal_output, (self.int_num_hidden_neurons))
                ####### note there is no input weights to bias node            #######
                ####### add dumpy column for bias weights to avoid numpy error #######
                error_signal_output_dump_bias = numpy.append(error_signal_output_rshp, self.dbl_bias)
                error_signal_hidden = self.derivhyberb(numpy.inner(self.wh, x)) * numpy.inner(self.wo,
                                                                                              error_signal_output_dump_bias)
                ###dimensions of error_signal_hidden = (number_of_hidden_neurons x 1)
                ## update weights hidden
                tmp_wh = numpy.transpose(self.wh)
                counter = 0
                for x_ele in x:
                    delta_wh = self.dbl_eta * error_signal_hidden * x_ele
                    tmp_wh[counter] = delta_wh + tmp_wh[counter]  ## update weight
                    counter = counter + 1
                self.wh = tmp_wh.transpose()  ## weights updated
                ## end for x

                ## update weights output
                delta_wo = self.dbl_eta * error_signal_output * actual_hidden_output
                counter = 0
                for delta_wo_ele in delta_wo:
                    self.wo[counter] = self.wo[counter] + delta_wo_ele
                    counter = counter + 1
                    ## out weights updated
            ## end loop training set
            self.arr_mse = numpy.append(self.arr_mse, numpy.mean(numpy.sum(numpy.square(errors))))
        ## end loop epochs
        return

    ## hyber-bolic function
    def hyberb(self, V):
        """
        :rtype: VECTOR SAME DIMENSIONS AS V
        """
        # PHI.arange().reshape
        return (numpy.exp(V * 2) - 1) / (numpy.exp(V * 2) + 1)

    ## derivation of hyper-bolic function
    def derivhyberb(self, V):
        """
        :rtype: VECTOR SAME DIMENSIONS AS V
        """
        # PHI.arange().reshape
        return 4 * numpy.exp(V * 2) / numpy.square(1 + numpy.exp(V * 2))

    ## plot mse
    def plotMSE(self):
        pyplot.xlabel('Number of Epochs')
        pyplot.ylabel('MSE (Mean Square Error)')
        pyplot.plot(self.arr_mse)
        pyplot.show()
