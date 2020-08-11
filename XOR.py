import numpy as np

class XOR() :
    def _init_weights(self, h = 2):

    def _affine (self, W, X, B):

    def _sigmoid (self, o):

    def _eval_loss (self, X, Y, weights):

    def _gradients (self, X, Y, weights):  

    def optimize (self, X, Y, h = 3, learning_rate = 0.1, epoch = 1000):


if __name__ == "__main__" :
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    Y = np.array([[0, 1, 1, 0]])

    xor = XOR()
    learned_weights, loss_trace, predicts = xor.optimize(X, Y, h=3, learning_rate = 0.1, epoch = 100000)
    print("Y hat : {}".format(predicts))
    print("predicts : {}".format([1 if y > 0.5 else 0 for y in predicts[0]]))
