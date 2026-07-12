import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator # 1. Get function
        if f is not None:
            x = f.input  # 2. Retrieving the function’s input
            x.grad = f.backward(self.grad) # 3. Call the function’s `backward` method
            x.backward() # 4. Call the `backward` method of the previous variable (recursively)


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)  # Ensure that the output variable retains the creator’s information
        self.input = input
        self.output = output  # Save the output variables as well
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

def main():
    A = Square()
    B = Exp()
    C = Square()
    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)
    # 反向传播
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad)


if __name__ == '__main__':
    main()