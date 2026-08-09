import os
import dezero
import pathlib
from dezero import optimizers
from dezero import DataLoader
from dezero.models import MLP
import dezero.functions as F

if __name__ == '__main__':
    max_epoch = 3
    batch_size = 100
    train_set = dezero.datasets.MNIST(train=True)
    train_loader = DataLoader(train_set, batch_size)
    model = MLP((1000, 10))
    optimizer = optimizers.SGD().setup(model)

    # load parameters
    if os.path.exists(pathlib.Path(__file__).parent.parent.absolute() / 'weights/my_mlp.npz'):
        print('loading weights from weights/my_mlp.npz......')
        model.load_weights(file_name='my_mlp.npz')

    for epoch in range(max_epoch):
        sum_loss = 0
        for x, t in train_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            model.cleargrads()
            loss.backward()
            optimizer.update()
            sum_loss += float(loss.data[0]) * len(t)
        print('epoch: {}, loss: {:.4f}'.format(
            epoch + 1, sum_loss / len(train_set)))

    model.save_weights(file_name='my_mlp.npz')