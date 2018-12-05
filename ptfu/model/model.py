''' model.py - 学習モデルを記述するモジュール '''

from ..nn.NeuralNet import NeuralNet

class Model:
    ''' 各種学習モデルの規定となるクラス '''

    def __init__(self):
        return

    def train(self, **options):
        ''' このモデルを訓練する。与えられるパラメータと返り値は具象クラスによって異なる '''
        raise NotImplementedError

    def validate(self, **options):
        ''' 訓練したモデルに新しいデータを与えて検証する。与えられるパラメータと返り値は具象クラスによって異なる '''
        raise NotImplementedError

class SingleNetworkModel(Model):
    ''' 単一のニューラルネットから構成される学習モデル '''
    
    def __init__(self, neural_network, lossfunc, optimizer):
        super(SingleNetworkModel, self).__init__()
        assert isinstance(neural_network, NeuralNet) # 型チェック
        self.nn = neural_network
        self.loss = lossfunc
        self.optimizer = optimizer
        self.training = False
        return

    def train(self, **options):
        ''' このモデルを訓練する。与えるパラメータ:
        dataset: 学習用データセット。DataSetオブジェクト
        endflag: EndFlagオブジェクト。指定しない場合永遠
        minibatchsize: 学習のミニバッチサイズ
        qparallel: ミニバッチ取得キューイングの並列スレッド数
        hooks: 一定回数毎に呼び出されるhookのリスト hookはSmartSessionHookオブジェクト
        '''
        self.training = True

        # データセットの設定
        assert 'dataset' in options
        dataset = options['dataset']
        is_tfrecord = isinstance(dataset, TFRecordDataset)

        # 終了条件の設定
        endflag = options['endflag'] if 'endflag' in options else NoneEndFlag

        # ミニバッチのキューイングを開始
        assert 'minibatchsize' in options
        minibatchsize = options['minibatchsize']
        if not is_tfrecord: # TFRecordではキューイングは行わない
            from concurrent.futures import ThreadPoolExecutor
            from queue import Queue
            qparallel = options['qparallel'] if 'qparallel' in options else 3
            qmax = 200
            self.trainq = Queue(qmax)
            self.qthreashold = 100
            self.qbatchsize = (qmax - self.qthreashold) // qparallel
            for i in range(qparallel):
                self._qloop(dataset, minibatchsize)
        
        train_op = self.optimizer.minimize(self.loss)

        # Sessionを立ち上げて学習を行う
        with SmartSession() as session:

            if 'hooks' in options:
                session.registerHooks(options['hooks'])
            
            while not endflag.should_end():
                session.run(train_op)

        self.training = False
        return


    def _qloop(self, dataset, minibatchsize):
        ''' ミニバッチキューをバックグラウンドで作成するスレッドワーカー関数 '''
        from time import sleep
        import random
        while self.training:
            if self.trainq.qsize() < self.qthreashold:
                for _ in range(self.qbatchsize):
                    minibatch = dataset.obtain_minibatch(minibatchsize)
                    if minibatch is not None:
                        self.trainq.put(minibatch)
            else:
                sleep(random.random())
        return



    def validate(self, **options):
        ''' 訓練したモデルに検証用データを与えて検証を行う。与えるパラメータ:
        dataset: 検証用データセット。DataSetオブジェクト '''
        if self.training:
            # 学習中はvalidationは走らせない
            raise ValueError
        return
        
        
        

    
    
        
