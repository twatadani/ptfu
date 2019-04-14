''' nnlayer.py - LayerBasedNeuralNetのためのレイヤーを記述するモジュール '''

from .neuralnet import NeuralNet

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

name = 'nnlayer'


        
