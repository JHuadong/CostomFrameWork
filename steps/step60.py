import numpy as np
import dezero
from dezero import Model
from dezero import SeqDataLoader
import dezero.functions as F
import dezero.layers as L

if __name__ == '__main__':
    max_epoch = 100
    batch_size = 30
    hidden_size = 100
    bptt_length = 30

    train_set = dezero.datasets.SinCurve(train=True)
    # 1.Data Loader for Time Series Data
    dataloader = SeqDataLoader(train_set, batch_size=batch_size)
    seqlen = len(train_set)


    class BetterRNN(Model):
        def __init__(self, hidden_size, out_size):
            super().__init__()


            self.rnn = L.LSTM(hidden_size)  # 2 use lstm
            self.fc = L.Linear(out_size)


        def reset_state(self):
            self.rnn.reset_state()


        def forward(self, x):
            y = self.rnn(x)
            y = self.fc(y)
            return y


    model = BetterRNN(hidden_size, 1)
    optimizer = dezero.optimizers.Adam().setup(model)
    for epoch in range(max_epoch):
        model.reset_state()
        loss, count = 0, 0

        for x, t in dataloader:
            y = model(x)
            loss += F.mean_squared_error(y, t)
            count += 1
            if count % bptt_length == 0 or count == seqlen:
                # dezero.utils.plot_dot_graph(loss)  # plot
                model.cleargrads()
                loss.backward()
                loss.unchain_backward()
                optimizer.update()

        avg_loss = float(loss.data[0]) / count
        print('| epoch %d | loss %f' % (epoch + 1, avg_loss))
