''' model.py - 学習モデルを記述するモジュール '''

class Model:
    ''' 各種学習モデルの規定となるクラス '''

    def __init__(self):
        self.gstep = None
        return

    def train(self, **options):
        ''' このモデルを訓練する。与えられるパラメータと返り値は具象クラスによって異なる '''
        raise NotImplementedError

    def validate(self, **options):
        ''' 訓練したモデルに新しいデータを与えて検証する。与えられるパラメータと返り値は具象クラスによって異なる '''
        raise NotImplementedError

    def global_step_tensor(self):
        ''' global stepを表すtensorを返す '''
        import tensorflow.train as train
        if self.gstep is None:
            self.gstep = train.get_global_step()
        if self.gstep is None:
            self.gstep = train.create_global_step()
        return self.gstep

class SingleNetworkModel(Model):
    ''' 単一のニューラルネットから構成される学習モデル '''
    
    def __init__(self, neural_network, lossfunc, optimizer):
        from ..nn.neuralnet import NeuralNet
        import tensorflow.summary as summary
        super(SingleNetworkModel, self).__init__()
        assert isinstance(neural_network, NeuralNet) # 型チェック
        self.nn = neural_network
        self.loss = lossfunc
        summary.scalar(name = 'Loss', tensor = self.loss)
        self.optimizer = optimizer
        self.training = False
        return

    def train(self, **options):
        ''' このモデルを訓練する。与えるパラメータ:
        dataset: 学習用データセット。DataSetオブジェクト
        tfconfig: TensorFlowのコンフィグ。TFConfigオブジェクト
        fdmapper: feed_dictを作成するための辞書。TFRecord以外のデータセットでは必要
        辞書の内容は
        { feed_dictで使用するkey tensor: obtain_minibatchで得られる辞書のキー,  ...}
        の形式。
        endflag: EndFlagオブジェクト。指定しない場合永遠
        minibatchsize: 学習のミニバッチサイズ
        qparallel: ミニバッチ取得キューイングの並列スレッド数
        hooks: 一定回数毎に呼び出されるhookのリスト hookはSmartSessionHookオブジェクト
        '''
        #import tensorflow as tf
        from .. import SmartSession

        self.training = True

        # データセットの設定
        from ..dataset import TFRecordDataSet
        assert 'dataset' in options
        dataset = options['dataset']
        is_tfrecord = isinstance(dataset, TFRecordDataSet)

        if not is_tfrecord:
            assert 'fdmapper' in options
            fdmapper = options['fdmapper']

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
            executor = ThreadPoolExecutor()
            for i in range(qparallel):
                executor.submit(self._qloop, dataset, minibatchsize)
                #self._qloop(dataset, minibatchsize)
        
        # 計算グラフを定義する
        #self.nn.define_network()
        train_op = self.optimizer.minimize(self.loss, global_step = self.global_step_tensor())

        # Tensorflowのコンフィグ
        if 'tfconfig' in options:
            tfconfig = options['tfconfig']
        else:
            tfconfig = TFConfig

        # Sessionを立ち上げて学習を行う
        with SmartSession(tfconfig) as session:

            # ループ中のhookを設定
            if 'hooks' in options:
                session.registerHooks(options['hooks'])

            # ループ終了条件の設定
            endflag.setSmartSession(session)
            
            while not endflag.should_end():
                minibatch = self._minibatch_fromq()

                if is_tfrecord:
                    session.run(train_op)
                else:
                    fd = self._create_fd(minibatch, fdmapper)
                    session.run(train_op, feed_dict=fd)

        self.training = False
        return

    def _minibatch_fromq(self):
        ''' キューからミニバッチデータを取り出す '''
        import random
        from time import sleep
        while self.trainq.empty():
            sleep(random.random())
        else:
            return self.trainq.get()
            
    @staticmethod
    def _create_fd(minibatch, fdmapper):
        ''' ミニバッチデータからfeed_dictを作成する '''
        fd = {}
        for key in fdmapper:
            fd[key] = minibatch[fdmapper[key]]
        return fd

    def _qloop(self, dataset, minibatchsize):
        ''' ミニバッチキューをバックグラウンドで作成するスレッドワーカー関数 '''
        from time import sleep
        import random
        try:
            while self.training:
                print('_qloop loop: qsize=', self.trainq.qsize())
                if self.trainq.qsize() < self.qthreashold:
                    for _ in range(self.qbatchsize):
                        minibatch = dataset.obtain_minibatch(minibatchsize)
                        if minibatch is not None:
                            self.trainq.put(minibatch)
                else:
                    sleep(random.random())
        except:
            import traceback
            traceback.print_exc()
        return



    def validate(self, **options):
        ''' 訓練したモデルに検証用データを与えて検証を行う。与えるパラメータ:
        dataset: 検証用データセット。DataSetオブジェクト '''
        if self.training:
            # 学習中はvalidationは走らせない
            raise ValueError
        return
        
        
        

    
    
        
