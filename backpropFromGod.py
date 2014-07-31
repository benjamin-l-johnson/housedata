#!/usr/bin/python3

import random
import numpy as np
import pickle

def sigmoid(x):
    return (np.tanh(0.5 * x) + 1) / 2 # use tanh ?

def softmax(x):
    exp = np.exp(x - x.max())
    return exp / exp.sum()

def binary_decision(x):
    return 1 if x > 0.5 else 0

def categorical_decision(x):
    return x.argmax()

def dropout_mask(x, p):
    return (np.random.random(x.shape) < p)

def scale_dropout_net(clf, dropout):
    if not clf.dropout:
        raise TypeError('Scaling only makes sense for ANNs trained with dropout')
    clf_scaled = Classifier(clf.idim, clf.hdim, clf.odim)
    clf_scaled.A = dropout * clf.A
    clf_scaled.B = dropout * clf.B
    clf_scaled.hb = dropout * clf.hb
    clf_scaled.ob = dropout * clf.ob
    clf_scaled.dropout = False
    return clf_scaled

def gaussian_initializer(m, n, sigma=0.1):
    return np.random.normal(0, sigma, (m, n))

class FFANNClassifier(object):
    """Simple feed forward artificial neural network classifier.

    Uses a single hidden layer. Has a few modern bells and whistles.

    Args:
        input_dim (int): Number of input units.
        hidden_dim (int): Number of hidden units.
        output_dim (int): Number of output units.
        learning_rate (float): Learning rate/step size parameter.
        momentum (float or bool): Momentum parameter. If False, don't use momentum.
        lr_reg (float or bool): L2 regularization parameter. If False, don't use L2 regularization.
        dropout (float or bool): Dropout probability. If False, don't use dropout.
        mcmc_steps (int): If dropout is used, the number of MCMC steps to take for predictions.
    """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 learning_rate=0.01,
                 momentum=0.7,
                 l2_reg=False,
                 dropout=0.5,
                 mcmc_steps=10):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # init weights
        self.A = gaussian_initializer(hidden_dim, input_dim)
        self.B = gaussian_initializer(output_dim, hidden_dim)

        # init biases
        self.hb = np.zeros(hidden_dim)
        self.ob = np.zeros(output_dim)

        # init deltas for momentum
        self.dA = np.zeros(self.A.shape)
        self.dB = np.zeros(self.B.shape)
        self.dhb = np.zeros(self.hb.shape)
        self.dob = np.zeros(self.ob.shape)

        # binary of categorical decision function
        if self.output_dim== 1:
            self.output = sigmoid
            self.decision = binary_decision
        else:
            self.output = softmax
            self.decision = categorical_decision

        # training parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.l2_reg = l2_reg
        self.dropout=dropout
        self.mcmc_steps=mcmc_steps

    def ffhid(self, x):
        return np.tanh(np.dot(self.A, x) + self.hb)

    def ffout(self, h):
        return self.output(np.dot(self.B, h) + self.ob)

    def dropout_predict_prob(self, x):
        """Test time predictions for dropout net. A more computationally efficient solution to
        approximate the exponential number of models if to scale the learned parameter before
        test time, see scale_dropout_net.
        """
        h = self.ffhid(x)
        h_masked = h * dropout_mask(h, self.dropout)
        o = self.ffout(h_masked)
        for i in range(1, self.mcmc_steps):
            h_masked = h * dropout_mask(h, self.dropout)
            o += self.ffout(h_masked)
        return o / o.sum()

    def predict_prob(self, x):
        """If binary return a float representing the predicted probability that P(Y=1|X=x).
        If categorical return a vector whose k^th dimension the the predicted probability
        that P(Y=k|X=x), k=0,...,K-1 where K = self.output_dim.
        """
        if self.dropout:
            return self.dropout_predict_prob(x)
        return self.ffout(self.ffhid(x))

    def predict(self, x):
        """Return an integer representing the predicted label/class. 0 or 1 if binary,
        0,1,...K-1 if K way categorical.
        """
        return self.decision(self.predict_prob(x))

    def update(self, x, y):
        """Single SGD step optimizing log-loss.

        Args:
            x (np.ndarray): Input vector.
            y (int): True output label.
        """

        h = self.ffhid(x)
        if self.dropout:
            dmask = dropout_mask(h, self.dropout)
            o = self.ffout(dmask * h)
        else:
            o = self.ffout(h)

        if self.output_dim == 1:
            erro = y - o
        else:
            erro = -1 * o
            erro[y] += 1

        if self.dropout:
            errh = (dmask * (1 - h**2)) * np.dot(self.B.T, erro)
        else:
            errh = (1 - h**2) * np.dot(self.B.T, erro)

        if self.l2_reg:
            dB = self.learning_rate * (np.outer(erro, h) - self.l2_reg * self.B)
            dA = self.learning_rate * (np.outer(errh, x) - self.l2_reg * self.A)
            dob = self.learning_rate * (erro - self.l2_reg * self.ob)
            dhb = self.learning_rate * (errh - self.l2_reg * self.hb)
        else:
            dB = self.learning_rate * np.outer(erro, h)
            dA = self.learning_rate * np.outer(errh, x)
            dob = self.learning_rate * erro
            dhb = self.learning_rate * errh

        if self.momentum:
            dB += self.momentum * self.dB
            dA += self.momentum * self.dA
            dob += self.momentum * self.dob
            dhb += self.momentum * self.dhb
            self.dB, self.dA, self.dob, self.dhb = dB, dA, dob, dhb

        self.B += dB
        self.A += dA
        self.ob += dob
        self.hb += dhb

        y_pred = self.decision(o)
        return 0 if y_pred == y else 1

    def fit_sgd(self, X, Y, shuffle=True):
        """Make a full SGD pass over X, Y.

        Args:
            X (iterable of np.ndarray): Input vectors.
            Y (iterable of ints): True output labels.
        """
        err = 0.
        if shuffle:
            D = zip(X, Y)
            random.shuffle(D)
            for x, y in D:
                err += self.update(x, y)
        else:
            for x, y in zip(X, Y):
                err += self.update(x, y)
        return err / len(X)

class FFANNRegressor(object):
    """Simple feed forward artificial neural network regressor.

    Uses a single hidden layer. Has a few modern bells and whistles. Assumes
    y_i ~ N(f_i(x), 1) where f_i(x) is the i^th component of the output vector.

    Args:
        input_dim (int): Number of input units.
        hidden_dim (int): Number of hidden units.
        output_dim (int): Number of output units.
        learning_rate (float): Learning rate/step size parameter.
        momentum (float or bool): Momentum parameter. If False, don't use momentum.
        lr_reg (float or bool): L2 regularization parameter. If False, don't use L2 regularization.
        dropout (float or bool): Dropout probability. If False, don't use dropout.
        mcmc_steps (int): If dropout is used, the number of MCMC steps to take for predictions.
    """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 learning_rate=0.01,
                 momentum=0.7,
                 l2_reg=False,
                 dropout=0.5,
                 mcmc_steps=10):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # init weights
        self.A = gaussian_initializer(hidden_dim, input_dim)
        self.B = gaussian_initializer(output_dim, hidden_dim)

        # init biases
        self.hb = np.zeros(hidden_dim)
        self.ob = np.zeros(output_dim)

        # init deltas for momentum
        self.dA = np.zeros(self.A.shape)
        self.dB = np.zeros(self.B.shape)
        self.dhb = np.zeros(self.hb.shape)
        self.dob = np.zeros(self.ob.shape)

        # training parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.l2_reg = l2_reg
        self.dropout=dropout
        self.mcmc_steps=mcmc_steps

    def ffhid(self, x):
        return np.tanh(np.dot(self.A, x) + self.hb)

    def ffout(self, h):
        return np.dot(self.B, h) + self.ob

    def dropout_predict(self, x):
        """Test time predictions for dropout net. A more computationally efficient solution to
        approximate the exponential number of models if to scale the learned parameter before
        test time, see scale_dropout_net.
        """
        h = self.ffhid(x)
        h_masked = h * dropout_mask(h, self.dropout)
        o = self.ffout(h_masked)
        for i in range(1, self.mcmc_steps):
            h_masked = h * dropout_mask(h, self.dropout)
            o += self.ffout(h_masked)
        # use mean prediction for regression
        return o / self.mcmc_steps

    def predict(self, x):
        if self.dropout:
            return self.dropout_predict(x)
        return self.ffout(self.ffhid(x))

    def update(self, x, y):
        """Single SGD step optimizing log-loss.

        Args:
            x (np.ndarray): Input vector.
            y (int): True output label.
        """

        h = self.ffhid(x)
        if self.dropout:
            dmask = dropout_mask(h, self.dropout)
            o = self.ffout(dmask * h)
        else:
            o = self.ffout(h)

        erro = y - o

        if self.dropout:
            errh = (dmask * (1 - h**2)) * np.dot(self.B.T, erro)
        else:
            errh = (1 - h**2) * np.dot(self.B.T, erro)

        if self.l2_reg:
            dB = self.learning_rate * (np.outer(erro, h) - self.l2_reg * self.B)
            dA = self.learning_rate * (np.outer(errh, x) - self.l2_reg * self.A)
            dob = self.learning_rate * (erro - self.l2_reg * self.ob)
            dhb = self.learning_rate * (errh - self.l2_reg * self.hb)
        else:
            dB = self.learning_rate * np.outer(erro, h)
            dA = self.learning_rate * np.outer(errh, x)
            dob = self.learning_rate * erro
            dhb = self.learning_rate * errh

        if self.momentum:
            dB += self.momentum * self.dB
            dA += self.momentum * self.dA
            dob += self.momentum * self.dob
            dhb += self.momentum * self.dhb
            self.dB, self.dA, self.dob, self.dhb = dB, dA, dob, dhb

        self.B += dB
        self.A += dA
        self.ob += dob
        self.hb += dhb

        # L_2 norm = squared_error
        return np.linalg.norm(erro, 2)

    def fit_sgd(self, X, Y, shuffle=True):
        """Make a full SGD pass over X, Y.

        Args:
            X (iterable of np.ndarray): Input vectors.
            Y (iterable of ints): True output labels.
        """
        err = 0.0
        if shuffle:
            D = list(zip(X, Y))
            random.shuffle(D)
            for x, y in D:
                err += self.update(x, y)
        else:
            for x, y in zip(X, Y):
                err += self.update(x, y)
        return err / len(X)

def testparity():
    ''' YOU WILL USE BOSTON HOUSING ? '''
    # Test FFANNClassifier by solving a parity task.
    # n, dim = (500, 10)
    # split = int(0.8 * n)
    # X = np.random.randint(0, 2, (n, dim))
    # Y = X.sum(axis = 1) % 2
    # D = np.column_stack([X, Y])
    # trainD = D[:split]
    # testD = D[split:]
    f = open("houseData.pkl","rb")
    (trainDX,trainDY),(validX,validY),(testX,testY) = pickle.load(f)

    testD = list(zip(testX,testY))
    dim = 17

    clf = FFANNRegressor(dim, 90, 1, learning_rate=0.01, l2_reg=False, dropout=False)
    i = 0
    etrain=0.01
    try:
        while etrain > .0001:

            ''' HERE YOU WILL HAVE TO CODE YOUR TRAINING REGIMEN'''
            etrain = clf.fit_sgd(trainDX, trainDY,shuffle=True)
            #print(etrain)
            if not i % 20:
                err = 0.
                random.shuffle(testD)
                for x, y in testD:
                    #p = 1 if clf.predict(x) > 0.5 else 0
                    #if p != y:
                    #    err += 1
                    p = clf.predict(x)
                    err += abs(p - y)
                print( etrain, err / len(testD))
                #print("Went through the data %d tiems"%i)
            i += 1
    except KeyboardInterrupt:
        err = 0.
        for x, y in testD:
            #p = 1 if clf.predict(x) > 0.5 else 0
            #if p != y:
            #    err += 1
            p = clf.predict(x)
            err += abs(p - y)
        print()
        print( etrain, err / len(testD))


if __name__ == '__main__':
    testparity()
