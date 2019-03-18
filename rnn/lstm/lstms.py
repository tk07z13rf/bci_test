import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer.backends import cuda

class RNNLSTM(chainer.Chain):

    def __init__(self, in_channel, hidden_channel, out_channel):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Linear(in_channel, hidden_channel)
            self.lstm1 = L.LSTM(hidden_channel, hidden_channel)
            self.l2 = L.Linear(hidden_channel, out_channel)

    def __call__(self, inputs, targets):
        loss = 0
        for t in range(inputs.shape[2]):
            x = inputs[..., t]
            y = inputs[..., t]
            h = self.one_step_forward(x, y)
            loss += F.softmax_cross_entropy(h, y)

        return h, loss

    def one_step_forward(self, inputs, targets):
        h = self.l1(inputs)
        h = F.relu(h)
        h = self.lstm1(h)
        h = self.l2(h)

        return h

    def reset_state(self):
        self.lstm1.reset_state()


class RNNNStepLSTM(chainer.Chain):

    def __init__(self, in_channel, hidden_channel, out_channel):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Linear(in_channel, hidden_channel)
            self.lstm1 = L.NStepLSTM(2, hidden_channel, hidden_channel, 0.5)
            self.l2 = L.Linear(hidden_channel, out_channel)

    def __call__(self, inputs, targets):
        loss = 0
        xs = tuple(self.l1(x) for x in inputs)
        hy, cy, ys = self.lstm1(hx=None, cx=None, xs=xs)
        ys = [self.l2(y) for y in ys]

        return ys

    def reset_state(self):
        self.lstm1.reset_state()


class Convolution2DNStepLSTM(chainer.Chain):

    def __init__(self, in_channels, hidden_channels, lstm_in_size, lstm_out_size, out_size):
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(in_channels, hidden_channels,
                                         ksize=(2, 2), pad=(1, 1))
            self.lstm1 = L.NStepLSTM(2, lstm_in_size, lstm_out_size, 0.5)
            self.l1 = L.Linear(lstm_out_size, out_size)

    def __call__(self, inputs, targets):
        sequence_len, n, c, wx, wy = inputs.shape
        xs = tuple(self.conv1(x)[..., :-1, :-1].reshape(n, -1) for x in inputs)
        hy, cy, ys = self.lstm1(hx=None, cx=None, xs=xs)
        ys = [self.l1(y) for y in ys]
        print(len(ys))
        return ys


def main():
    model = Convolution2DNStepLSTM(3, 3, 5*5*3, 10, 2).to_gpu(0)
    X = np.random.randn(200, 10, 3, 5, 5).astype(np.float32)
    y = np.random.randn(100, 10)
    X = Variable(cuda.to_gpu(X))
    pred = model(X, y)


if __name__ == "__main__":
    main()
