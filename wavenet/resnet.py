import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers


class ResidualBlock1D(chainer.Chain):

    """ResidualBlock1D
    Wavenet 論文 https://arxiv.org/pdf/1609.03499.pdf

    """

    def __init__(self, in_channels, hidden_channels, out_channels, condition_channels=None,
                 ksize=2, dilate=2, pad=0, initializer=initializers.GlorotUniform()):

        """Dilation Convolutionのコンストラクタ

        Args:
            in_channels: 入力データのチャネル数
            hidden_channels: 中間層のチャネル数
            out_channels: skip connectのチャネル数
            condition_channels: 補助特徴量のチャネル数
            ksize: フィルタのサイズ(デフォルト: 2)
            dilate: dilateの数(デフォルト: 2)
            pad: paddingの数(デフォルト: 2)
            initializer: パラメタの初期値

        Returns:
            None

        """

        super(ResidualBlock1D, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.condition_channels = condition_channels
        self.ksize = ksize
        self.dilate = dilate
        self.pad = pad
        self.queue = None

        with self.init_scope():
            # dilation畳み込み
            self.convolution = L.Convolution1D(self.in_channels, 2 * self.hidden_channels,
                                               ksize=self.ksize, dilate=self.dilate,
                                               pad=self.pad, initialW=initializer)

            # residual結合の1x1畳み込み
            self.res = L.Convolution1D(self.hidden_channels, self.in_channels, ksize=1, initialW=initializer)
            # skip結合の1x1畳み込み
            self.skip = L.Convolution1D(self.hidden_channels, self.out_channels, ksize=1, initialW=initializer)
            # バッチ正則化
            self.bn_tanh = L.BatchNormalization(self.hidden_channels)
            self.bn_sigmoid = L.BatchNormalization(self.hidden_channels)

            # 補助特徴量のための畳み込みおよびバッチ正則化
            if self.condition_channels is not None:
                self.condition_convolution = L.Convolution1D(self.condition_channels, 2 * self.hidden_channels,
                                                             ksize=2, initialW=initializer)
                self.bn_conditions_tanh = L.BatchNormalization(self.hidden_channels)
                self.bn_conditions_sigmoid = L.BatchNormalization(self.hidden_channels)

    def __call__(self, inputs, conditions=None):
        """forward

        Args:
            inputs: 入力データ(Variable) Shape (データ数，入力チャネル数, 時系列長)
            conditions: 補助特徴量(Variable) Shape (データ数，補助特徴量のチャネル数, 時系列長)

        Returns:
            res_output: residual結合後の出力 Shape (データ数，出力(=入力)チャネル数, 時系列長)
            skip_output: skip結合用の出力 Shape (データ数，skip結合のチャネル数, 時系列長)

        """

        # Gated結合のtanhおよびsigmoidの計算
        h = self.convolution(inputs)[..., :-self.pad]
        h_tanh, h_sigmoid = F.split_axis(h, 2, axis=1)
        h_tanh = F.tanh(self.bn_tanh(h_tanh))
        h_sigmoid = F.sigmoid(self.bn_sigmoid(h_sigmoid))

        # 補助特徴量のがある場合
        if self.condition_channels is not None and conditions is not None:
            h_conditions = self.condition_convolution(conditions)
            h_conditions_tanh, h_conditions_sigmoid = F.split_axis(h_conditions, 2, axis=1)
            h_tanh += F.tanh(self.bn_conditions_tanh(h_conditions_tanh))
            h_sigmoid += F.tanh(self.bn_conditions_sigmoid(h_conditions_sigmoid))

        # Gated結合
        h = h_tanh * h_sigmoid

        # residual結合の出力
        res_outputs = self.res(h) + inputs
        # skip結合の出力
        skip_outputs = self.skip(h)

        return res_outputs, skip_outputs

    def pop(self, condition=None):
        """
        Queueを入力として順伝搬

        Arg:
            condition: 補助特徴量

        Return:
            None

        """

        return self(self.queue, condition)

    def push(self, x):
        """
        Queueの末尾に以前の出力をconcat，Queueの先頭を削除

        Arg:
            x: １時刻前の出力

        Return:
            None

        """

        self.queue = F.concat((self.queue[:, :, 1:], x), axis=2)

    def initialize_queue(self, n):
        """
        queueの初期化

        Arg:
           n: （データ数）

        return:
           None

        """
        self.queue = chainer.Variable(self.xp.zeros((
            n, self.in_channels,
            self.dilate * (self.ksize - 1) + 1, 1),
            dtype=self.xp.float32))


class ResidualNetwork1D(chainer.ChainList):
    """ResidualNetwork1D

    Arg:

    Return:

    """

    def __init__(self, in_channels, hidden_channels, out_channels, condition_channels=None,
                 ksize=2, dilation=None):
        super(ResidualNetwork1D, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.condition_channels = condition_channels
        self.dilation = dilation
        self.ksize = ksize
        for dilate in self.dilation:
            pad = dilate * (self.ksize - 1)
            self.add_link(ResidualBlock1D(in_channels=self.in_channels,
                                          hidden_channels=self.hidden_channels,
                                          out_channels=self.out_channels,
                                          condition_channels=self.condition_channels,
                                          ksize=self.ksize, dilate=dilate, pad=pad))

    def __call__(self, inputs, conditions=None):
        """forward

        Args:
            inputs: 入力データ(Variable) Shape (データ数，入力チャネル数, 時系列長)
            conditions: 補助特徴量(Variable) Shape (データ数，補助特徴量のチャネル数, 時系列長)

        Returns:
            sum_skip_outputs: skip結合の出力 Shape (データ数，skip結合のチャネル数, 時系列長)

        """
        r = inputs
        sum_skip_outputs = None
        for i, func in enumerate(self.children()):
            r, s = func(r, conditions)
            if i == 0:
                sum_skip_outputs = s
            else:
                sum_skip_outputs += s

        return sum_skip_outputs

    def initialize_queue(self, n):
        """initialize_queue
        queueの初期化

        Arg:
            n: （データ数）

        return:
            None

        """
        for func in self.children():
            func.initialize_queue(n)

    def generate(self, x, conditions=None):
        """generate
        自己回帰により波形生成
        Arg:
            x: 入力(１時刻前の出力) Shepe (データ数, 入力のチャネル数, 1)
            conditions 補助特徴量
        Return:
            sum_skip_outputs Shape(データ数，skip出力のチャネル数, 1)
        """

        sum_skip_outputs = None

        if conditions is None:
            conditions = [None for _ in range(len(self.children()))]

        for i, (func, condition) in enumerate(zip(self.children(), conditions)):
            func.push(x)
            r, s = self.func.pop(condition)
            if i == 0:
                sum_skip_outputs = s
            else:
                sum_skip_outputs += s
        return sum_skip_outputs


if __name__ == "__main__":
    help(ResidualBlock1D.pop)
    help(ResidualNetwork1D.generate)
