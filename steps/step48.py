import math
import numpy as np
import dezero
from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP

if __name__ == "__main__":
    # 1.Setting hyperparameters
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    lr = 1.0

    # 2.Load data / Create a model and an optimizer
    x, t = dezero.datasets.get_spiral(train=True)
    model = MLP((hidden_size, 3))
    optimizer = optimizers.SGD(lr).setup(model)

    data_size = len(x)
    max_iter = math.ceil(data_size / batch_size)  # Round up to the nearest whole number

    for epoch in range(max_epoch):
        # 3.Reordering the dataset index
        index = np.random.permutation(data_size)
        sum_loss = 0

        for i in range(max_iter):
            # 4.Create small batches of data
            batch_index = index[i * batch_size:(i + 1) * batch_size]
            batch_x = x[batch_index]
            batch_t = t[batch_index]

            # 5.Compute the gradient / Update the parameters
            y = model(batch_x)
            loss = F.softmax_cross_entropy(y, batch_t)
            model.cleargrads()
            loss.backward()
            optimizer.update()

            sum_loss += float(loss.data) * len(batch_t)

        # 6.Output the training status for each round
        avg_loss = sum_loss / data_size
        print('epoch %d, loss %.2f' % (epoch + 1, avg_loss))