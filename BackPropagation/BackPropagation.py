import math
import random
import sys
import time

#sigmoid function
def sigmoid(x):
    return math.tanh(x)

def dsigmoid(y):
    return 1.0 - y**2

def makeMatrix(Y, X, fill=0.0):
    m = []
    for i in range(Y):
        m.append([fill]*X)
    return m

class Nauron:
    def __init__(self):
        pass

class NN:
    def __init__(self,numinput,numhidden,numoutput):
        
        self.numinput=numinput+1 #+1 for bias input node
        self.numhidden=numhidden
        self.numoutput=numoutput
        
        self.inputact=[1.0]*self.numinput
        self.hiddenact=[1.0]*self.numhidden
        self.outputact=[1.0]*self.numoutput
        
        self.inputweights=makeMatrix(self.numinput,self.numhidden)
        self.outpweights=makeMatrix(self.numhidden,self.numoutput)
        
        #randomize weights
        for i in range(self.numinput):
            for j in range(self.numhidden):
                self.inputweights[i][j] = random.uniform(-0.2, 0.2)
        for j in range(self.numhidden):
            for k in range(self.numoutput):
                self.outpweights[j][k] = random.uniform(-2.0, 2.0)
        
        self.inputchange = makeMatrix(self.numinput, self.numhidden)
        self.outputchange = makeMatrix(self.numhidden, self.numoutput)
        #TODO:Random fill matrix of weights
    
    def update(self,inputs):
        """Update network"""
        
        if len(inputs) != self.numinput-1:
            raise ValueError('Wrong number of inputs, should have %i inputs.' % self.numinput)
        
        #ACTIVATE ALL NEURONS INSIDE A NETWORK
        
        #Activate input layers neurons (-1 ignore bias node)
        for i in range(self.numinput-1):
            self.inputact[i] = inputs[i]
        
        #Activate hidden layers neurons
        for h in range(self.numhidden):
            sum = 0.0
            for i in range(self.numinput):
                sum = sum + self.inputact[i] * self.inputweights[i][h]
            self.hiddenact[h] = sigmoid(sum)
        
        #Activate output layers neurons
        for o in range(self.numoutput):
            sum = 0.0
            for h in range(self.numhidden):
                sum = sum + self.hiddenact[h] * self.outpweights[h][o]
            self.outputact[o] = sigmoid(sum)
        
        return self.outputact[:]
    
    def backPropagate(self, targets, learningrate, momentum):
        """Back Propagate """
        
        if len(targets) != self.numoutput:
            raise ValueError('Wrong number of target values.')

        # calculate error for output neurons
        output_deltas = [0.0] * self.numoutput
        for k in range(self.numoutput):
            error = targets[k]-self.outputact[k]
            output_deltas[k] = dsigmoid(self.outputact[k]) * error

        # calculate error for hidden neurons
        hidden_deltas = [0.0] * self.numhidden
        for j in range(self.numhidden):
            error = 0.0
            for k in range(self.numoutput):
                error = error + output_deltas[k]*self.outpweights[j][k]
            hidden_deltas[j] = dsigmoid(self.hiddenact[j]) * error

        # update output weights
        for j in range(self.numhidden):
            for k in range(self.numoutput):
                change = output_deltas[k]*self.hiddenact[j]
                self.outpweights[j][k] += learningrate*change + momentum*self.outputchange[j][k]
                self.outputchange[j][k] = change

        # update input weights
        for i in range(self.numinput):
            for j in range(self.numhidden):
                change = hidden_deltas[j]*self.inputact[i]
                self.inputweights[i][j] += learningrate*change + momentum*self.inputchange[i][j]
                self.inputchange[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.outputact[k])**2
        return error
    
    def train(self, patterns, iterations=10000, learningrate=0.5, momentum=0.1):
        """Train network a patterns"""
        
        for i in range(iterations):
            error = 0.0
            
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, learningrate, momentum)
            if i % 100 == 0:
                print('error %-.5f' % error)
                time.sleep(0.5)

def test(hiddenLayer):
    
    # Teach network XOR function
    patterns = [
        [[0,0], [0]],
        [[1,0], [1]],
        [[0,1], [1]],
        [[1,1], [0]],
    ]

    # create a network with two input, two hidden, and one output nodes
    network = NN(2, hiddenLayer, 1)
    # train it with some patterns
    network.train(patterns)
    # test it
    for pat in patterns:
        print(pat[0], '=', int(network.update(pat[0])[0]+0.5))

if __name__ == '__main__':
    if(len(sys.argv) < 2):
        test()
    elif(sys.argv[1] == '-h'):
        print("You can call current file with parametr: HiddenLayer")
        print("Just call it like that: BackPropagation N - where n is HiddenLayer neurons")
        sys.exit()
    else:
        test(sys.argv[1])