from dezero.core import Parameter
from dezero.core import Variable
import numpy as np
import weakref
import dezero.functions as F

class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)):
            self._params.add(name)
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError

    def params(self):
        for name in self._params:
            obj = self.__dict__[name]

            if isinstance(obj, Layer):
                yield from obj.params()
            else:
                yield obj

    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None):
        super().__init__()

        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if self.in_size is not None: # If `in_size` is not specified, processing is deferred.
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def forward(self, x):
        # Initialise weights when propagating data
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()

        y = F.linear(x, self.W, self.b)
        return y

    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data


if __name__ == '__main__':
    import dezero.layers as L
    import dezero.functions as F
    from dezero import Layer

    model = Layer()
    model.l1 = L.Linear(5)  # 只指定输出大小
    model.l2 = L.Linear(3)


    # 进行推理的函数
    def predict(model, x):
        y = model.l1(x)
        y = F.sigmoid(y)
        y = model.l2(y)
        return y


    # 访问所有参数
    for p in model.params():
        print(p)
    # 重置所有参数的梯度
    model.cleargrads()