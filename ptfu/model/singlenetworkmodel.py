''' singlenetworkmodel.py - 単一のネットワークから構成されるモデル '''

from .model import Model

class SingleNetworkModel(Model):
    ''' 単一のニューラルネットから構成される学習モデル '''
    
    def __init__(self, neural_network, optimizer):
        from ..nn.neuralnet import NeuralNet
        #import tensorflow.summary as summary

        super(SingleNetworkModel, self).__init__()
        assert isinstance(neural_network, NeuralNet) # 型チェック
        self.nn = neural_network
        #self.loss = lossfunc
        #if self.loss is not None:
        #    summary.scalar(name = 'Loss', tensor = self.loss)
        self.optimizer = optimizer
        self.training = None # マルチプロセスモデルに従い、trainingはmultiprocessing.Queueに変更。
        self.prepared = False # 学習準備ができたかどうか
        return

    def get_training_tensor(self):
        ''' 学習用か検証用かを示すboolean tensorを取得する。設定されていない場合はNoneが返る '''
        return self.nn.get_training_tensor()

    def prepare_train(self, tfconfig, minibatchsize_per_tower, tf_loss_func, **tf_loss_func_options):
        ''' パラメータを与えてネットワーク定義、損失関数定義などの学習準備を行う
        tfconfig: TFConfigオブジェクト
        minibatchsize_per_tower: ミニバッチサイズ。タワー型並列実行については1タワーあたりの数
        tf_loss_func: TensorFlowの損失関数
        **tf_loss_func_options: tf_loss_funcに与えるオプション '''

        ntowers = tfconfig.ntowers()
        self.minibatchsize = ntowers * minibatchsize_per_tower

        if ntowers == 1:
            self.nn.define_network(tfconfig.towers[0], self.nn.inputs)
        else:
            output_tensors = None
            for i in range(ntowers):
                
                tw_start = i * minibatchsize_per_tower
                tw_end = tw_start + minibatchsize_per_tower

                tower_inputs = {}
                # input tensorsをtower用に分解
                for key in self.nn.inputs:
                    splitted = self.nn.inputs[key][tw_start:tw_end]
                    tower_inputs[key] = splitted

                with tf.variable_scope('tower-' + str(i), reuse = False):
                    self.nn.define_network(tfconfig.towers[i], tower_inputs)
                    # output tensorをtower対応にする
                    if output_tensors is None:
                        output_tensors = self.nn.outputs
                    else:
                        for key in output_tensors:
                            output_tensors[key] = tf.concat([output_tensors[key], self.nn.outputs[key]], axis=0)
            self.nn.outputs = output_tensors

        # 損失関数の定義
        import tensorflow.summary as summary

        ntowers = tfconfig.ntowers()
        towers = tfconfig.towers
        if ntowers == 1:
            self.loss = tf_loss_func(**tf_loss_func_options)
            self.train_op = self.optimizer.minimize(self.loss, global_step = self.global_step_tensor)
        else:
            tower_grads = []
            tower_losses = None

            for i in range(ntowers):
                twname = 'tower-' + str(i)
                with tf.variable_scope(twname, reuse=False):
                    with tf.device(towers[i][-1]):
                        tower_loss = tf_loss_func(**tf_loss_func_options)
                        tower_grad = self.optimizer.compute_gradients(
                            loss = tower_loss,
                            var_list = [x for x in tf.trainable_variables() if twname in x.name])
                        tower_grads.append(tower_grad)
                        if tower_losses is None:
                            tower_losses = tf.reshape(tower_loss, shape=(1,))
                        else:
                            tower_losses = tf.concat([tower_losses, tf.reshape(tower_loss, shape=(1, ))])
            self.loss = tf.reduce_mean(tower_losses, name = 'loss')
            avarage_grad = self.average_gradients(tower_grads)
            self.train_op = self.optimizer.apply_gradients(
                average_grad, global_step = self.global_step_tensor())
        summary.scalar(name = 'Loss', tensor = self.loss)
        self.prepared = True
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
        qparallel: ミニバッチ取得キューイングの並列スレッド数
        hooks: 一定回数毎に呼び出されるhookのリスト hookはSmartSessionHookオブジェクト
        '''
        from .. import SmartSession
        from ..logger import get_default_logger
        from multiprocessing import Manager

        assert self.prepared is True, 'Please call prepare_train before training'

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
        minibatchsize = self.minibatchsize
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
                    session.run(self.train_op)
                else:
                    fd = self._create_fd(minibatch, fdmapper, True)
                    session.run(self.train_op, feed_dict=fd)

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
            
    def _create_fd(self, minibatch, fdmapper, training):
        ''' ミニバッチデータからfeed_dictを作成する '''
        fd = {}
        for key in fdmapper:
            fd[key] = minibatch[fdmapper[key]]
        training_tensor = self.get_training_tensor()
        if training_tensor is not None:
            fd[training_tensor] = training
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
        fd = self._create_fd(wholebatch, fdmapper, False)

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


    @staticmethod
    def average_gradients(tower_grads):
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
        Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            #print('grad_and_vars len=', len(grad_and_vars))
            #for j in grad_and_vars:
            #    print(j)
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)
                
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)
                
            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)
                
            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

        
