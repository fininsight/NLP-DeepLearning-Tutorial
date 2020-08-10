import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython

if get_ipython():
    print("jupyter envirionment")
    from tqdm import tqdm_notebook as tqdm
else:
    print("command shell envirionment")
    from tqdm import tqdm


def init_weights(h = 2):
    W1 = np.random.rand(2,h)
    B1 = np.random.rand(h,1)
    W2 = np.random.rand(h,1)
    B2 = np.random.rand(1,1)
    return W1, B1, W2, B2

def affine (W, X, B):
    return np.dot(W.T, X) + B

def sigmoid (o):
    return 1./(1+np.exp(-1*o))

def loss_eval (X, Y, weights):
    W1, B1, W2, B2 = weights
    
    # Forward: input Layer
    Z1 = affine(W1, X, B1)
    H  = sigmoid(Z1)

    # Forward: Hidden Layer
    Z2 = affine(W2, H, B2)
    Y_hat = sigmoid(Z2)

    loss = 1./X.shape[1] * np.sum(-1 * (Y * np.log(Y_hat) + (1-Y) * np.log(1-Y_hat)))
    return Z1, H, Z2, Y_hat, loss

def gradients (X, Y, weights):       
    W1, B1, W2, B2 = weights
    m = X.shape[1]
    
    Z1, H, Z2, Y_hat, loss = loss_eval(X, Y, [W1, B1, W2, B2])
    
    # BackPropagate: Hidden Layer
    dW2 = np.dot(H, (Y_hat-Y).T)
    dB2 = 1. / 4. * np.sum(Y_hat-Y, axis=1, keepdims=True)    
    dH  = np.dot(W2, Y_hat-Y)

    # BackPropagate: Input Layer
    dZ1 = dH * H * (1-H)
    dW1 = np.dot(X, dZ1.T)
    dB1 = 1. / 4. * np.sum(dZ1, axis=1, keepdims=True)
    
    return [dW1, dB1, dW2, dB2], loss

def optimize (X, Y, weights, learning_rate = 0.1, iteration = 1000, sample_size = 0):
    loss_trace = []

    for epoch in tqdm(range(iteration), desc="optimize"):
        dweights, loss = gradients(X, Y, weights)
        
        for weight, dweights in zip(weights, dweights):
            weight += - learning_rate * dweights
        
        if (epoch % 100 == 0):
            loss_trace.append(loss)
        
    _, _, _, Y_hat, _ = loss_eval(X, Y, weights) 
    
    return weights,loss_trace, Y_hat


if __name__ == "__main__" :
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    Y = np.array([[0, 1, 1, 0]])

    weights = init_weights(3)
    new_params, loss_trace, Y_hat_predict = optimize(X, Y, weights, 0.1, 100000)
    print(Y_hat_predict)
    print([1 if y > 0.5 else 0 for y in Y_hat_predict[0]])

    # plt.plot(loss_trace)
    # plt.ylabel('loss')
    # plt.xlabel('iterations (per hundreds)')
    # plt.show()