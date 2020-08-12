import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython

if get_ipython():
    print("jupyter envirionment")
    from tqdm import tqdm_notebook as tqdm
else:
    print("command shell envirionment")
    from tqdm import tqdm

class XOR() :
    def _init_weights(self, i, h = 2):
        W1 = np.random.rand(i,h)
        B1 = np.random.rand(h,1)
        W2 = np.random.rand(h,1)
        B2 = np.random.rand(1,1)

        # W1 = np.array([[0.5, 0], [0.5, 0]])
        # B1 = np.array([[0.6], [0.6]])
        # W2 = np.array([[0.7], [0.7]])
        # B2 = np.array([[0.8]])
        
        return W1, B1, W2, B2

    def _affine (self, W, X, B):
        return np.dot(W.T, X) + B

    def _sigmoid (self, o):
        return 1./(1+np.exp(-1*o))

    def _feedforward(self, X, Y, weights) :
        W1, B1, W2, B2 = weights
        
        # Forward: input Layer
        Z1 = self._affine(W1, X, B1)
        H  = self._sigmoid(Z1)

        # Forward: Hidden Layer
        Z2 = self._affine(W2, H, B2)
        Y_hat = self._sigmoid(Z2)

        return Z1, H, Z2, Y_hat

    def _loss (self, Y, Y_hat):
        loss = -1 * 1./X.shape[1] * np.sum((Y * np.log(Y_hat) + (1-Y) * np.log(1-Y_hat)))
        return loss

    def _gradients (self, X, Y, weights, Z1, H, Z2, Y_hat ):       
        W1, B1, W2, B2 = weights
        m = X.shape[1]
        
        # BackPropagate: Hidden Layer
        dW2 = np.dot(H, (Y_hat-Y).T)
        dB2 = 1. / Y.shape[1] * np.sum(Y_hat-Y, axis=1, keepdims=True)
        dH  = np.dot(W2, Y_hat-Y)

        # BackPropagate: Input Layer
        dZ1 = dH * H * (1-H)
        dW1 = np.dot(X, dZ1.T)
        dB1 = 1. / Y.shape[1] * np.sum(dZ1, axis=1, keepdims=True)
        
        return [dW1, dB1, dW2, dB2]
    
    def _step_func(self, y) :
        return 1 if y > 0.5 else 0 

    def _accuracy(self, Y, Y_hat) :
        tmp = [self._step_func(y) for y in Y_hat[0]]
        return (Y == tmp).mean()


    def optimize (self, X, Y, h = 2, learning_rate = 0.1, epoch = 1000):
        W = self._init_weights(X.shape[0], h)
        loss_trace = []

        for i in tqdm(range(epoch), desc="optimize"):
            Z1, H, Z2, Y_hat = self._feedforward(X, Y, W)
            loss = self._loss(Y, Y_hat)
            gradient = self._gradients(X, Y, W, Z1, H, Z2, Y_hat)

            #가중치 갱신
            for w, gradient in zip(W, gradient):
                w += - learning_rate * gradient
            
            if (i % 1000 == 0):
                print("Loss : {}, Accuracy : {}".format(loss, self._accuracy(Y, Y_hat)))
                loss_trace.append(loss)

        _, _, _, Y_hat = self._feedforward(X, Y, W)
        r = [1 if y > 0.5 else 0 for y in Y_hat[0]]
        
        return W,loss_trace, Y_hat, r


if __name__ == "__main__" :
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    Y = np.array([[0, 1, 1, 0]])

    xor = XOR()
    learned_weights, loss_trace, predicts, results = xor.optimize(X, Y, h=2, learning_rate = 0.1, epoch = 10000)
    print("Y hat : {}".format(predicts))
    print("predicts : {}".format(results))

    plt.plot(loss_trace)
    plt.ylabel('loss')
    plt.xlabel('iterations (per hundreds)')
    plt.show()