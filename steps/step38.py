import numpy as np
from dezero import Variable
import dezero.functions as F

if __name__ == '__main__':
    x = Variable(np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ]
    ))
    y = F.reshape(x, (6,))
    y.backward(retain_grad=True)
    # print(x.grad)

    x1 = Variable(np.random.randn(1, 2, 3))
    y1 = x1.reshape((2, 3))
    y1 = x1.reshape(2, 3)
    print(x1.grad)
    # print(y1)

    x2 = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
    y2 = F.transpose(x2)
    y2.backward()
    print(x2.grad)