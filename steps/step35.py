import numpy as np
from dezero import Variable
from dezero.utils import plot_dot_graph
import dezero.functions as F

if __name__ == "__main__":
    x = Variable(np.array(1.0))
    y = F.tanh(x)
    x.name = 'x'
    y.name = 'y'
    y.backward(create_graph=True)
    iters = 6
    for i in range(iters):
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)

    # 绘制计算图
    gx = x.grad
    gx.name = 'gx' + str(iters + 1)
    plot_dot_graph(gx, verbose=False, to_file='my_tanh.png', from_file='my_tanh.dot')