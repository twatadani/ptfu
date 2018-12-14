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
        if self.loss is not None:
            summary.scalar(name = 'Loss', tensor = self.loss)
        self.optimizer = optimizer
        self.training = None # マルチプロセスモデルに従い、trainingはmultiprocessing.Queueに変更。
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
        from .. import SmartSession
        from ..logger import get_default_logger
        from multiprocessing import Manager

        logger = get_default_logger()
        manager = Manager()
        self.training = manager.Queue(2)

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
            from concurrent.futures import ProcessPoolExecutor
            from .. import functions as f
            qparallel = options['qparallel'] if 'qparallel' in options else 3

            qmax = 200
            self.trainq = manager.Queue(qmax)
            self.qthreashold = 100
            self.qbatchsize = (qmax - self.qthreashold) // qparallel
            ncpu = f.cpu_count()
            executor = ProcessPoolExecutor(ncpu)
            datasetinfo = dataset.toDataSetInfo()
            for i in range(qparallel):
                executor.submit(self._qloop, datasetinfo, self.trainq, self.qthreashold,
                                self.qbatchsize, minibatchsize, self.training)
        
        train_op = self.optimizer.minimize(self.loss, global_step = self.global_step_tensor())

        # Tensorflowのコンフィグ
        if 'tfconfig' in options:
            tfconfig = options['tfconfig']
        else:
            tfconfig = TFConfig

        # ネットワーク構造をプリントする
        logger.log(self.nn.print_network())

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

            # 終了条件を表示
            logger.log('学習終了 終了条件: ' + endflag.reason())

        self.training.put(True)
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

    @staticmethod
    def _qloop(datasetinfo, queue, qthreashold, qbatchsize, minibatchsize, signalqueue):
        ''' ミニバッチキューをバックグラウンドで作成するワーカー関数
        processベースの並列処理のため、static methodで実装する '''
        from time import sleep
        import random

        try:
            dataset = datasetinfo.constructDataSet()
            while signalqueue.empty():
                if queue.qsize() < qthreashold:
                    for _ in range(qbatchsize):
                        minibatch = dataset.obtain_minibatch(minibatchsize)
                        if minibatch is not None:
                            queue.put(minibatch)
                        if not signalqueue.empty():
                            continue
                else:
                    sleep(random.random())
        except:
            import traceback
            from ..logger import get_default_logger
            logger = get_default_logger()
            logger.error(traceback.format_exc())
        return

    def forward_wholedataset(self, dataset, tfconfig, fdmapper,
                             session=None, checkpoint_dir=None):
        ''' 学習後のモデルに対して検証用データセットのすべてを与え、forward propagationを行う。
        学習後モデルはsessionが与えられた場合はそのsessionが、checkpoint_dirが与えられた場合は
        checkpoint_dirの最新checkpointが用いられる。
        すでにsessionがある場合は引数のtfconfigは使用されない。
        fdmapperはtrainと同じ形式。
        返り値はこのモデルが保有するneural networkのoutput_tensorsがkey、
        それらに対する値のリストがvalueとなった辞書形式'''
        import tensorflow as tf
        from ptfu import SmartSession

        output_tensors = self.nn.get_output_tensors()

        wholebatch = dataset.obtain_wholedata()
        fd = self._create_fd(wholebatch, fdmapper)

        if session is None:
            last_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            if last_checkpoint is not None:
                with SmartSession(tfconfig) as session:
                    saver = tf.train.Saver()
                    saver.restore(session.session, save_path=last_checkpoint)
                    result = session.run(output_tensors, feed_dict=fd)                    
            else:
                raise ValueError('latest checkpoint cannot be found. search dir=' + checkpoint_dir)
        else:
            result = session.run(output_tensors, feed_dict=fd)


        return result
        
