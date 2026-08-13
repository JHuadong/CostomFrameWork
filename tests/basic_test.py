import numpy as np
import dezero.layers as L

if __name__ == '__main__':
    rnn = L.RNN(10)  # only set the size of hidden layer
    x = np.random.rand(1, 1)
    h = rnn(x)
    print(h.shape)