#!/usr/bin/env ../venv.sh


# Back-Propagation Neural Networks
# 
# Written in Python.  See http://www.python.org/
# Placed in the public domain.
# Neil Schemenauer <nas@arctrix.com>







import math
import random
import string
import numpy as np
import dataLoader as dl
random.seed(0)

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a


# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return np.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2

class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = np.ones(self.ni)
        self.ah = np.ones(self.nh)
        self.ao = np.ones(self.no)
        self.count=0

        self.dsigmoid_vec = np.vectorize(dsigmoid)
        self.sigmoid_vec = np.vectorize(sigmoid)

        # create weights
        self.wi = np.random.uniform(-1, 1, size=self.ni*self.nh).reshape(self.ni,self.nh)
        self.wo = np.random.uniform(-1, 1, size=self.nh*self.no).reshape(self.nh,self.no)
        #self.wi = np.random.random((self.ni,self.nh))-.5
        #self.wo = np.random.random((self.nh,self.no))-.5

        #print(self.wo)
        #print(self.wi)


    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        # input activations
        self.ai[:-1] = inputs


        #dot product of active inputs and weight inputs
        sum = np.dot(self.ai,self.wi)
        self.ah = np.tanh(sum)
        
        #dot product of active hidden and weight outputs
        sum = np.dot(self.ah,self.wo)
        self.ao = np.tanh(sum)
        #print(self.ao)
        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        output_deltas = np.zeros(self.no)

        #calculate output delta
        error = targets-self.ao #
        #logging.debug("error=targets-self.ao= %s = %s-%s",error,targets,self.ao)

        #take that error get
        output_deltas =   error *self.dsigmoid_vec(self.ao) 
        #logging.debug("output_deltas:%s\n error:%s",output_deltas,error)


        
        # calc hidden deltas
        hidden_deltas = np.zeros(self.nh)
        error = output_deltas*self.wo[:self.nh]
        

        #error = np.reshape(error,hidden_deltas.shape)
        #hidden_deltas = error * self.dsigmoid_vec(self.ah)  
        #or.....
        #error = np.reshape(error,hidden_deltas.shape)
        hidden_deltas = np.reshape(error,hidden_deltas.shape) * self.dsigmoid_vec(self.ah)

        #logging.debug("output deltas %s \nerror: %s\nhidden_deltas:%s",output_deltas,error,hidden_deltas)

        # update output weights
        change = output_deltas * self.ah
        self.wo += N*np.reshape(change,(self.nh,self.no))

        #update input weights
        change = hidden_deltas * np.reshape(self.ai,(self.ni,1))
        self.wi += N*change
        
        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error


    def test(self, patterns):
        for p in patterns:
            print('->', self.update(p[0]),p[1])

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, N=1.0, M=0.1):

        # A pattern consists of [[x0,x1,...xn],[y]]
        np.random.shuffle(patterns)
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
            if i % 100 == 0:
                print('error[%d] %f' %(self.count,error))
                self.count+=1



def demo():
    train_patterns,valid_patterns,test = dl.load()
    # Teach network XOR function
    # pat = [
    #     [[0.0,0.0,0.0], [0]],
    #     [[0.0,1.0,1.0], [1]],
    #     [[1.0,0.0,1.0], [1]],
    #     [[1.0,1.0,0.0], [0]]
    # ]
    # pat = np.array(pat)
    # train = pat.copy()
    # valid = pat.copy()

    #create a network with 17 input, 10 hidden, and one output node
    n = NN ( 17, 10, 1)

    # train it with some patterns
    n.train(train_patterns,iterations=10000,N=1.0)
    #n.train(train_patterns,iterations=1000,N=1.3)
    #n.train(train_patterns,iterations=1000,N=1.3)
    # test it
    np.random.shuffle(valid_patterns)
    n.test(valid_patterns)




if __name__ == '__main__':
    demo()
