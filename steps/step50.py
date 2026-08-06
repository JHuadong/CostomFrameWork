import dezero.datasets
from dezero import DataLoader, optimizers
from dezero.models import MLP
import dezero.functions as F

if __name__ == '__main__':
    max_epoch = 300
    batch_size = 30
    hidden_size = 10
    lr = 1.0

    train_set = dezero.datasets.Spiral(train=True)
    test_set = dezero.datasets.Spiral(train=False)
    train_loader = DataLoader(train_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = MLP((hidden_size, 3))
    optimizer = optimizers.SGD(lr).setup(model)

    for epoch in range(max_epoch):  # 1.Small batches of data for training
        sum_loss, sum_acc = 0, 0
        for x, t in train_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)  # 2.Recognition accuracy of the training data
            model.cleargrads()
            loss.backward()
            optimizer.update()

            sum_loss += float(loss.data[0]) * len(t)
            sum_acc += float(acc.data[0]) * len(t)

        print('epoch: {}'.format(epoch + 1))
        print('train_loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss / len(train_set), sum_acc / len(train_set)))

        sum_loss, sum_acc = 0, 0
        with dezero.no_grad():  # 3.no graditude mode
            for x, t in test_loader:    # 4. used to test small batch of data
                y = model(x)
                loss = F.softmax_cross_entropy(y, t)
                acc = F.accuracy(y, t)  # 5.test the recognition accuracy of testing data
                sum_loss += float(loss.data[0]) * len(t)
                sum_acc += float(acc.data[0]) * len(t)

        print('test loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss / len(test_set), sum_acc / len(test_set)))


