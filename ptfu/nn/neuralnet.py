''' neuralnet.py - ニューラルネットを表現するモジュール '''

class NeuralNet:
    ''' 一つのニューラルネットを表現する 
    ライブラリユーザはNeuralNet(またはその子クラス)のインスタンスを生成して使用する '''

    def __init__(self, input_tensors, network_name=None):
        ''' コンストラクタ
        input_tensors: このNeuralNetの入力に使われるtensor, placeholder等のTensorFlowパラメータ
        network_name: このNeuralNetインスタンスの名前となる文字列
        '''

        self.inputs = None
        if input_tensors is None: # Noneの場合
            self.inputs = [] # 空リストにする
        elif hasattr(input_tensors, __getitem__): # リスト、タプル等の場合
            self.inputs = input_tensors # そのまま使う
        else: # リストでない場合は
            self.inputs = [ input_tensors ]

        self.name = None
        if network_name is None:
            self.name = 'unnamed_net'
        else:
            self.name = network_name

        self.outputs = []
        return

    def get_input_tensors(self):
        ''' このNeuralNetの入力tensorのリストを得る '''
        return self.inputs

    def get_output_tensors(self):
        ''' このNeuralNetの出力tensorのリストを得る '''
        return self.outputs

    def add_output_tensor(self, tensor):
        ''' NeuralNetの出力tensorのリストにtensorを加える '''
        if hasattr(tensor, __getitem__):
            self.outputs.extend(tensor)
        else:
            self.outputs.append(tensor)
        return

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

    def add_layer(self, layer, **options):
        ''' このNeuralNetにレイヤーを追加する。
        layer: Tensorflowのレイヤー 例えばtf.layers.conv2dやtf.layers.denseなど
        options: layerに与えるパラメータ '''

        self.layers.append(Layer(layer, **options))
        return

    def print_network(self):
        ''' ネットワーク構造を文字列として表現する。strを返す '''

        # ヘッダ
        s = '##### Network Structure of '
        s += self.name + ' #####' + br
        nparam = 0

        # ネットワーク構造
        # name, type, output shape, trainable parameters
        s += seprow
        s += Layer.shapeline(['layer name', 'layer type', 'output shape', 'train params'])
        s += seprow

        for layer in self.layers:
            s += layer.oneline_string()
            nparam += len(layer.trainable_parameters())

        s += seprow

        s += Layer.shapeline(['', '', 'total', nparam])
        s += seprow

        return s

class Layer:
    ''' Layerを表現するクラス。TensorFlowの関数あるいはクラスに対するラッパー
    ライブラリユーザはこのクラスを直接コードする必要はない '''

    # クラス変数
    sep = '|'

    def __init__(self, tflayer, **options):

        self.tflayer = tflayer(**options)

        if 'name' in options:
            self.name = options[name]
        else:
            self.name = self.tflayer.name
            assert self.name is not None
        return

    @staticmethod
    def shapecol(content):
        ''' contentの内容から出力用のカラム文字列を作成する '''
        if content is None:
            content = ''

        collen = LayerBasedNetwork.col_width # カラムの文字数
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

        s = sep
        for i in range(LayerBasedNeuralNet.ncols):
            if len(content) >= i+1:
                s += self.shapecol(content[i])
            else:
                s += sel.shapecol(None)
            s += sep
        return s


    def oneline_string(self):
        ''' 出力用の1行文字列を返す '''
        return self.shapeline([self.name, self.gettype(), self.output_shape(),
                               self.trainable_parameters()])

    def gettype(self):
        ''' このlayerのタイプ (conv2d, denseなど)を返す '''
        return type(self.tflayer)

    def output_shape(self):
        ''' このlayerの出力tensorのshapeを返す '''
        return self.tflayer.shape

    def trainable_parameters(self):
        ''' このlayerに含まれるtrainable parametersのリストを返す '''
        params = [x for x in tf.trainable_variables() if self.tflayer.name in x.name]
        return params

        
