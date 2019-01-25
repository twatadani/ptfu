''' neuralnet.py - ニューラルネットを表現するモジュール '''

class NeuralNet:
    ''' 一つのニューラルネットを表現する 
    ライブラリユーザはNeuralNet(またはその子クラス)のインスタンスを生成して使用する '''

    def __init__(self, input_tensors, network_name=None):
        ''' コンストラクタ
        input_tensors: このNeuralNetの入力に使われるtensor, placeholder等のTensorFlowパラメータ
        辞書形式で、'name': tensorの組み合わせとなる
        network_name: このNeuralNetインスタンスの名前となる文字列
        '''

        self.inputs = None
        if input_tensors is None: # Noneの場合
            self.inputs = {} # 空辞書にする
        elif isinstance(input_tensors, dict): #辞書型の場合
            self.inputs = input_tensors # そのまま使う
        else: # リストでない場合は
            self.inputs = { 'input': input_tensors }

        self._prepare_training_tensor()

        self.name = None
        if network_name is None:
            self.name = 'unnamed_net'
        else:
            self.name = network_name

        self.outputs = {}
        return

    def define_network(self, devices, input_tensors):
        ''' ネットワークを定義する。実際には具象クラスで行う。 
        devicesにはこのネットワークで使用するデバイスのリストが与えられる
        input_tensorsは並列実行などで適切に分割されたinputが与えられる。
        この関数は実際にはModelのインスタンスから呼ばれるため、ライブラリユーザーは通常使用する必要はない '''
        raise NotImplementedError

    def _prepare_training_tensor(self):
        ''' 学習用か検証用かを設定するboolean tensorを設定する '''
        import tensorflow as tf
        self.training_tensor = tf.placeholder(dtype=tf.bool,
                                              shape=(),
                                              name='training')
        self.inputs['training'] = self.training_tensor
        return

    def get_training_tensor(self):
        return self.training_tensor

    def get_input_tensors(self):
        ''' このNeuralNetの入力tensorのリストを得る '''
        return self.inputs

    def get_output_tensors(self):
        ''' このNeuralNetの出力tensorのリストを得る '''
        return self.outputs

    def add_output_tensor(self, tensor, name):
        ''' NeuralNetの出力tensorのリストにtensorを加える '''
        self.outputs[name] = tensor
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

        
