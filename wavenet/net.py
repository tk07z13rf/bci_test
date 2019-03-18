import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers

from wavenet.resnet import ResidualNetwork1D


class WaveNet(chainer.Chain):
    """WaveNet
    wavenetの論文

    https://arxiv.org/pdf/1609.03499.pdf
    Arg:

    Return:

    """

    def __init__(self, in_channels, res_in_channels, hidden_channels, skip_channels, out_channels, dilation=None):
        super(WaveNet, self).__init__()
        self.in_channels = in_channels
        self.res_in_channels = res_in_channels
        self.hidden_channels = hidden_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.dilation = dilation
        if self.dilation is None:
            self.dilation = [2**d for d in range(12)]*2
        initializer = initializers.GlorotUniform()
        with self.init_scope():
            self.causal1 = L.Convolution1D(self.in_channels, self.res_in_channels, ksize=2,
                                           pad=1, initialW=initializer)
            self.bn1 = L.BatchNormalization(self.res_in_channels)
            self.residual = ResidualNetwork1D(self.res_in_channels, self.hidden_channels,
                                              self.skip_channels, ksize=2,
                                              dilation=self.dilation)
            self.causal2 = L.Convolution1D(self.skip_channels, self.out_channels, ksize=1,
                                           pad=0, initialW=initializer)
            self.bn2 = L.BatchNormalization(self.skip_channels)

    def __call__(self, inputs, targets, conditions=None):
        h = self.causal1(inputs)[..., :-1]
        h = self.bn1(h)
        h = F.tanh(h)
        h = self.residual(h, conditions)
        h = self.bn2(h)
        h = F.tanh(h)
        h = self.causal2(h)

        loss = F.softmax_cross_entropy(h, targets)

        return h, loss

    def generate(self, inputs, conditions=None):
        pass


if __name__ == "__main__":
    help(WaveNet)
