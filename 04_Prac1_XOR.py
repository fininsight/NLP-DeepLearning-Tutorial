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
    def _init_weights(self, h = 2):
        W1 = np.random.rand(2,h)
        B1 = np.random.rand(h,1)
        W2 = np.random.rand(h,1)
        B2 = np.random.rand(1,1)
        return W1, B1, W2, B2

    def _affine (self, W, X, B):
        return np.dot(W.T, X) + B

    def _sigmoid (self, o):
        return 1./(1+np.exp(-1*o))

    def _eval_loss (self, X, Y, weights):
        W1, B1, W2, B2 = weights
        
        # Forward: input Layer
        Z1 = self._affine(W1, X, B1)
        H  = self._sigmoid(Z1)

        # Forward: Hidden Layer
        Z2 = self._affine(W2, H, B2)
        Y_hat = self._sigmoid(Z2)

        loss = 1./X.shape[1] * np.sum(-1 * (Y * np.log(Y_hat) + (1-Y) * np.log(1-Y_hat)))
        return Z1, H, Z2, Y_hat, loss

    def _gradients (self, X, Y, weights):       
        W1, B1, W2, B2 = weights
        m = X.shape[1]
        
        Z1, H, Z2, Y_hat, loss = self._eval_loss(X, Y, [W1, B1, W2, B2])
        
        # BackPropagate: Hidden Layer
        dW2 = np.dot(H, (Y_hat-Y).T)
        dB2 = 1. / Y.shape[1] * np.sum(Y_hat-Y, axis=1, keepdims=True)    
        dH  = np.dot(W2, Y_hat-Y)

        # BackPropagate: Input Layer
        dZ1 = dH * H * (1-H)
        dW1 = np.dot(X, dZ1.T)
        dB1 = 1. / Y.shape[1] * np.sum(dZ1, axis=1, keepdims=True)
        
        return [dW1, dB1, dW2, dB2], loss

    def optimize (self, X, Y, h = 3, learning_rate = 0.1, epoch = 1000):
        W = self._init_weights(h)
        loss_trace = []

        for i in tqdm(range(epoch), desc="optimize"):
            d, loss = self._gradients(X, Y, W)
            for w, d in zip(W, d):
                w += - learning_rate * d
            
            if (i % 100 == 0):
                loss_trace.append(loss)
            
        _, _, _, Y_hat, _ = self._eval_loss(X, Y, W) 
        
        return W,loss_trace, Y_hat


if __name__ == "__main__" :
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    Y = np.array([[0, 1, 1, 0]])

    xor = XOR()
    learned_weights, loss_trace, predicts = xor.optimize(X, Y, h=2, learning_rate = 0.1, epoch = 100000)
    #print("weights : {}".format(learned_weights))
    print("Y hat : {}".format(predicts))
    print("predicts : {}".format([1 if y > 0.5 else 0 for y in predicts[0]]))

    # plt.plot(loss_trace)
    # plt.ylabel('loss')
    # plt.xlabel('iterations (per hundreds)')
    # plt.show()