"""perceptron is a binary classifier. label is either -1 or 1. prediction = sign(wTX + b). sign function is the
activation function used,perceptron can not be used to solve the XOR problem """
""" perceptron training
it is a iterative procedure. classification rate should go up overall as we step through each "epoch"
 w = random vector ( b =0). w should be gaussian distributed
for epoch in range (max_epochs):
    make prediction
    get all currently misclassified examples
    if no misclassified examples ---> break
    X, Y = randomly select one misclassified example
    w = w + nYX. n = 1.0 , 0.1, 0.001 e.t.c typically. where n is the learning rate 

"""
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def get_data():
    w = np.array([-0.5,0.5])
    b =0.1
    X = np.random.random((300,2))*2 -1      # -1 to 1, random.random give data ranging from 0 to 1
    Y = np.sign(X.dot(w) + b)
    return X, Y


class perception:
    def fit(self,X,Y,learning_rate =1.0, epochs = 1000):
        D = X.shape[1]     # dimensionality of X
        self.w = np.random.randn(D)   # w is gaussian distributed
        self.b = 0

        N = len(Y)
        costs =[]

        for epoch in range(epochs):
            Yhat = self.predict(X)
            incorrect = np.nonzero( Y!= Yhat)[0]
            if len(incorrect) == 0:
                break

            i = np.random.choice(incorrect)
            self.w += learning_rate*Y[i]*X[i]
            self.b += learning_rate*Y[i]

            c = len(incorrect)/ float(N)
            costs.append(c)
        print("fianl w:",self.w, "final b:", self.b, "epochs:", (epoch +1) )
        plt.plot(costs)
        plt.show()

    def predict(self, X):
        return np.sign( X.dot(self.w) + self.b)

    def score (self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

if __name__ == "__main__":
    X, Y = get_data()
    plt.scatter(X[:,0], X[:,1], c =Y, s = 100, alpha = 0.5)
    plt.show()

    Ntrain = len(Y)/ 2
    Xtrain, Ytrain = X[:150], Y[:150]
    Xtest, Ytest = X[150:], Y[150:]

    model= perception()
    t0 = datetime.now()
    model.fit(Xtrain,Ytrain)
    print('Training time:', datetime.now() - t0)

    t0 = datetime.now()
    print('Train accuracy:', model.score(Xtrain,Ytrain))
    print('Time to compute train accuracy:',datetime.now() - t0, "Train Size:", len(Ytrain) )
  
    t0 = datetime.now()
    print('Test accuracy:', model.score(Xtest,Ytest))
    print('Time to compute test accuracy:',datetime.now() - t0, "Test Size:", len(Ytest) )
  







    