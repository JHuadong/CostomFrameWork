import numpy as np

class Variable:
    def __init__(self, data):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop() # Get function
            x, y = f.input, f.output # Retrieving the function’s input
            x.grad = f.backward(y.grad) # backward — calls the backward method

            if x.creator is not None:
                funcs.append(x.creator) # Add the previous function to the list


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
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

def square(x):
    return Square()(x) # Written on a single line

def exp(x):
    return Exp()(x)

def as_array(x):
    if np.isscalar(x):
        return np.array([x])
    return x


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

if __name__ == "__main__":
    main()