import numpy as np
from dezero.core import Function
class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x)
        return gx

def sin(x):
    return Sin()(x)

class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

def cos(x):
    return Cos()(x)

class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y

    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx

def tanh(x):
    return Tanh()(x)

if __name__ == "__main__":
    import numpy as np
    from dezero import Variable
    import dezero.functions as F

    x = Variable(np.array(1.0))
    y = F.sin(x)
    y.backward(create_graph=True)
    for i in range(3):
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)
        print(x.grad)  # n阶导数