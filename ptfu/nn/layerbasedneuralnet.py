''' layerbesedneuralnet.py - レイヤーベースのニューラルネットを表現するモジュール '''

from .neuralnet import NeuralNet
from .nnlayer import NNLayer

class LayerBasedNeuralNet(NeuralNet):
    ''' レイヤー構造から構成されるNeuralNet: NeuralNetクラスを継承する '''
        
    # クラス変数
    import os
    br = os.linesep

    ncols = 4 # 画面表示用のカラムの数
    col_width = 18 # カラムの'-'の個数
    seprow = '+'
    for _ in range(ncols):
        for __ in range(col_width):
            seprow += '-'
        seprow += '+'
    seprow += br
    # seprow = '+------------------+-----------------+------------------+---------------+' + br

    def __init__(self, input_tensors, network_name=None):
        ''' LayerBasedNeuralNetのイニシャライザ '''
        super(LayerBasedNeuralNet, self).__init__(input_tensors, network_name)

        self.layers = []
        return

    def get_layers(self):
        ''' このNeuralNetのlayerのリストを取得する '''
        return self.layers

    def last_layer(self):
        ''' このNeuralNetの最後のlayerを取得する '''
        return self.layers[-1]

    def add_input_layer(self, input_tensor):
        ''' このNeuralNetにinput layerを設定する '''
        self.layers.append(NNLayer(input_tensor, input_mode=True))
        return

    def add_layer(self, layer, **options):
        ''' このNeuralNetにレイヤーを追加する。
        layer: Tensorflowのレイヤー 例えばtf.layers.conv2dやtf.layers.denseなど
        options: layerに与えるパラメータ '''
        self.layers.append(NNLayer(layer, **options))
        return

    def lastout(self):
        ''' 最終レイヤーのoutput tensorを返す '''
        return self.last_layer().output_tensor()

    def print_network(self):
        ''' ネットワーク構造を文字列として表現する。strを返す '''

        br = LayerBasedNeuralNet.br
        seprow = LayerBasedNeuralNet.seprow
        
        # ヘッダ
        s = '##### Network Structure of '
        s += self.name + ' #####' + br
        nparam = 0

        # ネットワーク構造
        # name, type, output shape, trainable parameters
        s += seprow
        s += NNLayer.shapeline(['layer name', 'layer type', 'output shape', 'train params'])
        s += br
        s += seprow

        for layer in self.layers:
            s += layer.oneline_string()
            nparam += layer.count_params(layer.trainable_parameters())
            s += br

        s += seprow

        s += NNLayer.shapeline(['', '', 'total', nparam])
        s += br
        s += seprow

        return s

name = 'layerbasedneuralnet'


class NNLayer:
    ''' Layerを表現するクラス。TensorFlowの関数あるいはクラスに対するラッパー
    ライブラリユーザはこのクラスを直接コードする必要はない '''

    # クラス変数
    sep = '|'

    def __init__(self, tflayer, **options):

        self.tflayer = tflayer
        if 'input_mode' in options and options['input_mode'] == True:
            self.outtensor = tflayer
        else:
            self.outtensor = tflayer(**options)

        if 'name' in options:
            self.name = options['name']
        else:
            self.name = self.outtensor.name
            assert self.name is not None
        return

    @staticmethod
    def shapecol(content):
        ''' contentの内容から出力用のカラム文字列を作成する '''
        if content is None:
            content = ''

        collen = LayerBasedNeuralNet.col_width # カラムの文字数
        orig = str(content)
        olen = len(orig)
        prelen = (collen - olen) // 2
        postlen = prelen
        if prelen + olen + postlen < collen:
            postlen += 1

        col = ''
        if olen >= collen:
            col = orig[0:collen] # カラムの長さに合わせてtruncateする
        else:
            # 検算
            assert prelen + olen + postlen == collen

            # 元の文字列の前後にスペースを挿入
            for i in range(prelen):
                col += ' '
            col += orig
            for j in range(postlen):
                col += ' '

        return col
        
    @staticmethod
    def shapeline(content):
        ''' contentの内容から出力用の1行文字列を作成する '''
        sep = NNLayer.sep
        
        s = sep
        for i in range(LayerBasedNeuralNet.ncols):
            if len(content) >= i+1:
                s += NNLayer.shapecol(content[i])
            else:
                s += NNLayer.shapecol(None)
            s += sep
        return s


    def oneline_string(self):
        ''' 出力用の1行文字列を返す '''
        return self.shapeline([self.name, self.gettype(), self.output_shape(),
                               self.count_params(self.trainable_parameters())])

    def gettype(self):
        ''' このlayerのタイプ (conv2d, denseなど)を返す '''
        if callable(self.tflayer): # 関数の場合
            return self.tflayer.__name__
        else:
            return self.tflayer.name

    def output_tensor(self):
        ''' このlayerの出力tensorを返す '''
        return self.outtensor

    def output_shape(self):
        ''' このlayerの出力tensorのshapeを返す '''
        return self.outtensor.shape

    def trainable_parameters(self):
        ''' このlayerに含まれるtrainable parametersのリストを返す '''
        import tensorflow as tf
        params = [x for x in tf.trainable_variables() if self.name in x.name]
        return params

    @staticmethod
    def count_params(trainable_parameters):
        ''' trainable_parametersの出力から訓練可能パラメータの数を計算する '''
        sum = 0
        for variable in trainable_parameters:
            shape = variable.shape
            params = 1
            for dim in shape:
                params *= dim
            sum += params
        return sum

        
