import numpy as np

import chainer
import cupy
import chainer.functions as F
import chainer.links as L
from chainer import Function
from chainer.backends import cuda


class SRPooling(Function):

    def __init__(self):
        super().__init__()
        self.c = None
        self.h = None
        self.grad_c = None

    def forward_gpu(self, inputs):
        z, f, r, x = inputs
        shape = z.shape

        z = z.reshape(-1, shape[-1])
        r = f.reshape(-1, shape[-1])
        f = f.reshape(-1, shape[-1])
        x = x.reshape(-1, shape[-1])

        self.c = cuda.cupy.zeros((z.shape[0], z.shape[1]), dtype=np.float32)

        self.c = cuda.elementwise(
            in_params='raw float32 z, raw float32 f, int32 seq_length',
            out_params='raw float32 h',
            operation='''
                int ind[] = {i, 0};
                h[ind] = z[ind];
                for(int j = 1; j < seq_length; j++){
                    int ind[] = {i, j};
                    int pred_ind[] = {i, j-1};
                    h[ind] = f[ind] * h[pred_ind] + z[ind];                     
                   }
                ''',
            name='srp')(z, f, z.shape[1], self.c, size=z.shape[0])

        self.h = r * (cuda.cupy.tanh(self.c) - x) + x

        return self.h.reshape(shape),

    def backward_gpu(self, inputs, grad_outputs):
        z, f, r, x = inputs
        grad_out, = grad_outputs
        shape = z.shape

        z = z.reshape(-1, shape[-1])
        r = f.reshape(-1, shape[-1])
        f = f.reshape(-1, shape[-1])
        x = x.reshape(-1, shape[-1])

        grad_out = grad_out.reshape(-1, shape[-1])

        grad_tanh = 1. - cupy.tanh(self.c) ** 2

        grad_r = grad_out * (grad_tanh - x) * (1. - r) * r

        self.grad_c = cuda.cupy.zeros((z.shape[0], z.shape[-1]), dtype=np.float32)
        self.grad_c = cuda.elementwise(
            in_params='raw float32 r, raw float32 grad_h, raw float32 grad_tanh, int32 seq_length',
            out_params='raw float32 grad_c',
            operation='''
                int ind[] = {i, seq_length - 1};
                grad_c[ind] = r[ind] * grad_h[ind] * grad_tanh[ind];
                for(int j = seq_length - 2; j >= 0; j--){
                    int ind[] = {i, j};
                    int last_ind[] = {i, j+1};
                    grad_c[ind] = grad_c[last_ind] + r[ind] * grad_h[ind] * grad_tanh[ind];                     
                }    
                ''',
            name='grad_srp')(r, grad_out, grad_tanh,
                              z.shape[1], self.grad_c,
                              size=z.shape[0])
        grad_x = self.grad_c * (1. - r)
        grad_z = self.grad_c * (1. - f)
        grad_f = self.grad_c * (self.c - x) * (1. - f) * f

        return grad_z.reshape(shape), grad_f.reshape(shape), grad_r.reshape(shape), grad_x.reshape(shape)


class SimpleRecurrentUnitCellND(chainer.Chain):
    def __init__(self, ndim, in_channels, out_channels, ksize=1, pad=0):
        initializer = chainer.initializers.GlorotUniform()
        self.ndim = ndim
        self.ksize = ksize
        self.pad = pad

        if isinstance(self.ksize, tuple):
            if not isinstance(self.pad, list):
                self.pad = 0

            else:
                self.pad[-1] = 0
                self.pad = tuple(self.pad)

        if isinstance(self.ksize, int):
            if self.pad >= self.ksize:
                self.pad = 0

        super().__init__()
        with self.init_scope():
            self.conv = L.ConvolutionND(ndim, in_channels, 3 * out_channels, ksize=ksize, stride=1,
                                        pad=self.pad, initialW=initializer)
            self.bnf = L.BatchNormalization(out_channels)
            self.bnz = L.BatchNormalization(out_channels)
            self.bnr = L.BatchNormalization(out_channels)

    def __call__(self, inputs, targets=None):
        x = self.conv(inputs)
        z, f, r = F.split_axis(x, 3, axis=1)
        z = self.bnz(z)
        r = F.sigmoid(self.bnz(r))
        f = F.sigmoid(self.bnz(f))

        shape = z.shape

        self.h = SRPooling()(z, f, r, inputs)

        return self.h


if __name__ == "__main__":
    model = SimpleRecurrentUnitCellND(1, 10, 10, ksize=1, pad=0).to_gpu(0)
    X = cuda.to_gpu(np.random.randn(10, 10, 100).astype(np.float32))
    y = model(X)
    y.grad = y.data
    y.backward()
    print(y.shape)
