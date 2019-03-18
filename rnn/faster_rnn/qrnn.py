import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L


class QuasiRNNCellND(chainer.Chain):
    def __init__(self, ndim, in_channels, out_channels, ksize=2, pad=1, pool='ifo'):
        initializer = chainer.initializers.GlorotUniform()
        self.ndim = ndim
        self.ksize = ksize
        self.pad = pad

        if isinstance(self.ksize, tuple):
            if not isinstance(self.pad, list):
                self.pad = self.ksize[-1] - 1

            else:
                if self.ksize[-1] == self.pad[-1]:
                    self.pad[-1] = self.ksize[-1] - 1
                self.pad = tuple(self.pad)

        if isinstance(self.ksize, int):
            if self.pad >= self.ksize:
                self.pad = self.ksize - 1

        self.num_split = len(pool) + 1
        super().__init__()
        with self.init_scope():
            self.conv = L.ConvolutionND(ndim, in_channels, self.num_split * out_channels, ksize=ksize, stride=1,
                                        pad=self.pad, initialW=initializer)

        self.ct = None
        self.ht = None
        self.h = None

    def __call__(self, inputs, targets=None):

        if isinstance(self.pad, int):
            pad = self.pad
        if isinstance(self.pad, tuple):
            pad = self.pad[-1]

        x = self.conv(inputs)
        x = x if pad == 0 else x[..., :-pad]

        if self.num_split == 2:
            z, f = F.split_axis(x, self.num_split, axis=1)
            z = F.tanh(z)
            f = F.sigmoid(f)

        elif self.num_split == 3:
            z, f, o = F.split_axis(x, self.num_split, axis=1)
            z = F.tanh(z)
            f = F.sigmoid(f)
            o = F.sigmoid(o)

        elif self.num_split == 4:
            z, f, o, i = F.split_axis(x, self.num_split, axis=1)
            z = F.tanh(z)
            f = F.sigmoid(f)
            o = F.sigmoid(o)
            i = F.sigmoid(i)

        len_sequence = z.shape[-1]

        for t in range(len_sequence):
            zt = z[..., t]
            ft = f[..., t]
            ot = 1. if o is None else o[..., t]
            it = 1. - ft if i is None else i[..., t]

            if self.ct is None:
                self.ct = (1. - ft) * zt
            else:
                self.ct = ft * self.ct + it * zt
            self.ht = self.ct if o is None else ot * self.ct

            if self.h is None:
                self.h = F.expand_dims(self.ht, self.ndim + 1)
            else:
                self.h = F.concat((self.h, F.expand_dims(self.ht, self.ndim + 1)), axis=self.ndim + 1)

        return self.h

    def set_state(self, ct, ht, h):
        self.ct = ct
        self.ht = ht
        self.h = h

    def reset_state(self):
        self.set_state(None, None, None)


if __name__ == "__main__":
    model = QuasiRNNCellND(1, 20, 15, ksize=2, pad=1)
    X = np.random.randn(10, 20, 100)
    y = model(X.astype(np.float32))
    print(y.shape)
