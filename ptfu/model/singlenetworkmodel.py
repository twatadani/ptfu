''' singlenetworkmodel.py - 単一のネットワークから構成されるモデル '''

from .model import Model
import tensorflow as tf

class SingleNetworkModel(Model):
    ''' 単一のニューラルネットから構成される学習モデル '''
    
    def __init__(self, neural_network, optimizer, tfconfig, total_minibatchsize):
        ''' SingleNetworkModelの初期化
        neural_network: ニューラルネット。ptfu.nn.NeuralNetのインスタンスを指定する
        optimizer: この学習モデルのオプティマイザ。
        tfconfig: このモデルの学習に用いるTFConfigオブジェクト
        total_minibatchsize: 並列学習のtower分をすべて合わせたミニバッチサイズ '''
        from ..nn.neuralnet import NeuralNet
        from ..tfconfig import TFConfig
        from ..kernel import kernel

        super(SingleNetworkModel, self).__init__()

        # loggerの設定
        logger = kernel.logger()

        # neural_networkのチェック
        assert isinstance(neural_network, NeuralNet) # 型チェック
        self.nn = neural_network

        # optimizerのチェック
        assert isinstance(optimizer, tf.train.Optimizer)
        self.optimizer = optimizer

        # tfconfigのチェック
        assert isinstance(tfconfig, ptfu.TFConfig)
        self.tfconfig = tfconfig

        # initializerで定義する変数
        self.training = None # マルチプロセスモデルに従い、trainingはmultiprocessing.Queueに変更。
        self.defined = False # ネットワーク定義が完了しているかどうか
        self.prepared = False # 学習準備ができたかどうか

        # ネットワークの定義
        self.minibatchsize = total_minibatchsize
        ntowers = tfconfig.ntowers()
        modulo = total_minibatchsize % ntowers
        minibatchsize_per_tower = total_minibatchsize // ntowers
        if modulo != 0:
            logger.warning('total minibatchsizeがタワー数で割り切れません。total_minibatchsize: ' + str(total_minibatchsize) + ', タワー数: ' + str(ntowers))
            logger.warning('タワーあたりのminibatchsize: ' + str(minibatchsize_per_tower) + ', total minibatchsize: ' + str(minibatchsize_per_tower * ntowers) + 'で実行します。')
            self.minibatchsize = minibatchsize_per_tower * ntowers
        
        self.define_network(self.tfconfig, minibatchsize_per_tower)
        self.defined = True
            
        return

    def get_training_tensor(self):
        ''' 学習用か検証用かを示すboolean tensorを取得する。設定されていない場合はNoneが返る '''
        return self.nn.get_training_tensor()

    def define_network(self, tfconfig, minibatchsize_per_tower):
        ''' パラメータを与えてネットワーク定義を行う
        tfconfig: TFConfigオブジェクト
        minibatchsize_per_tower: ミニバッチサイズ。タワー型並列実行については1タワーあたりの数
        通常、この関数は__init__から呼ばれるためライブラリユーザーが明示的に呼び出す必要はない
        '''
        self._define_network_common(tfconfig, minibatchsize_per_tower)
        return

    def _define_network_common(self, tfconfig, minibatchsize_per_tower):
        ''' define_networkの共通部分 子クラスはこれに独自処理を加える '''
        from ..kernel import kernel

        ntowers = tfconfig.ntowers()

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

        training_tensor = kernel.get_training_tensor()
        self.set_training_true_op = tf.assign(training_tensor, True, name='set_training_true_op')
        self.set_training_false_op = tf.assign(training_tensor, False, name='set_training_false_op')
        return

    def prepare_train(self, tfconfig, tf_loss_func, **tf_loss_func_options):
        ''' パラメータを与えて損失関数定義などの学習準備を行う
        tfconfig: TFConfigオブジェクト
        tf_loss_func: TensorFlowの損失関数
        **tf_loss_func_options: tf_loss_funcに与えるオプション '''
        self._prepare_train_common(self, tfconfig, tf_loss_func, **tf_loss_func_options)
        return
        
    def _prepare_train_common(self, tfconfig, tf_loss_func, **tf_loss_func_options):
        ''' prepare_trainの共通実装。子クラスではこれにさらに独自処理を加える '''
        # 損失関数の定義
        import tensorflow.summary as summary
        import tensorflow as tf

        ntowers = tfconfig.ntowers()
        towers = tfconfig.towers
        if ntowers == 1:
            self.loss = tf_loss_func(**tf_loss_func_options)
            self.train_op = self.optimizer.minimize(self.loss, global_step = self.global_step_tensor())
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
                            tower_losses = tf.concat([tower_losses, tf.reshape(tower_loss, shape=(1, ))], axis=0)
            self.loss = tf.reduce_mean(tower_losses, name = 'loss')
            average_grad = self.average_gradients(tower_grads)
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
        from ptfu.kernel import kernel
        from .endflag import NoneEndFlag

        assert self.prepared is True, 'Please call prepare_train before training'

        logger = get_default_logger()
        manager = kernel.manager
        self.training = manager.Queue(2)

        # データセットの設定
        from ..dataset.tfrecorddataset import TFRecordDataSet
        assert 'dataset' in options
        dataset = options['dataset']
        is_tfrecord = isinstance(dataset, TFRecordDataSet)

        if not is_tfrecord:
            assert 'fdmapper' in options
            fdmapper = options['fdmapper']

        # 終了条件の設定
        endflag = options['endflag'] if 'endflag' in options else NoneEndFlag()

        # ミニバッチのキューイングを開始
        minibatchsize = self.minibatchsize
        if not is_tfrecord: # TFRecordではキューイングは行わない
            dataset.start_random_minibatch_queue(minibatchsize)
        
        # Tensorflowのコンフィグ
        if 'tfconfig' in options:
            tfconfig = options['tfconfig']
        else:
            tfconfig = None

        # ネットワーク構造をプリントする
        logger.log(self.nn.print_network())

        import tensorflow as tf

        if is_tfrecord:
            init_ops = [dataset.train_iterator_initializer(),
                        self.set_training_true_op ]
            init_fd = { dataset.srclist_ph: list(dataset.srclist) }
        else:
            init_ops = None
            init_fd = None

        # Sessionを立ち上げて学習を行う
        with SmartSession(tfconfig,
                          session_initialization_ops = init_ops,
                          initialization_feed_dict = init_fd) as self.session:
            
            # ループ中のhookを設定
            if 'hooks' in options:
                self.session.registerHooks(options['hooks'])
            if len(self.trainhooks) > 0:
                self.session.registerHooks(self.trainhooks)

            # ループ終了条件の設定
            endflag.setSmartSession(self.session)

            # initial hookを実行する
            self.session.run_initial_or_final_hooks(True)

            while not endflag.should_end():
                if is_tfrecord:
                    try:
                        self.session.run(self.train_op)
                    except tf.errors.OutOfRangeError:
                        from time import sleep
                        sleep(1)
                else:
                    minibatch = dataset.obtain_random_minibatch(minibatchsize)
                    fd = self._create_fd(minibatch, fdmapper, True)
                    self.session.run(self.train_op, feed_dict=fd)

            # 終了条件を表示
            logger.log('学習終了 終了条件: ' + endflag.reason())

            # final hookを実行する
            self.session.run_initial_or_final_hooks(False)

            # ミニバッチキューイングを止める
            if not is_tfrecord:
                dataset.stop_random_minibatch_queue()
            

        self.training.put(True)
        return

    def _create_fd(self, minibatch, fdmapper, training):
        ''' ミニバッチデータからfeed_dictを作成する '''
        fd = {}
        for key in fdmapper:
            fd[key] = minibatch[fdmapper[key]]
        training_tensor = self.get_training_tensor()
        if training_tensor is not None:
            fd[training_tensor] = training
        return fd

    def validate(self, dataset, minibatchsize, tfconfig, fdmapper, checkpoint_dir, 
                validationhook=None):
        ''' 学習後のモデルに対して検証用データセットのすべてを与え、forward propagationを行う。
        fdmapperはtrain同様。TFRecordの場合はNoneでよい。
        結果に対してどのような処理を行うかはvalidationhook関数に任せる。validationhookはSmartSessionHookの
        子クラスであるValidationHookのインスタンスを指定する '''
        import tensorflow as tf
        from ..dataset import TFRecordDataSet
        from ..smartsession import SmartSession
        from .validationhook import ValidationHook
        from ..logger import get_default_logger

        logger = get_default_logger()

        assert validationhook is not None, 'validationhook must not be None'
        if isinstance(validationhook, list) or isinstance(validationhook, tuple):
            single_hook = False
            for hook in validationhook:
                assert isinstance(hook, ValidationHook), 'validationhook must be an instance of ptfu.model.ValidationHook'
        else:
            single_hook = True
            assert isinstance(validationhook, ValidationHook), 'validationhook must be an instance of ptfu.model.ValidationHook'
        
        # tfconfigの設定を強制的に修正
        if tfconfig.use_checkpoint:
            logger.warning('tfconfig.use_checkpointがTrueですが、validation modeのためFalseに修正します。')
            tfconfig.use_checkpoint = False
        if tfconfig.use_autoreload:
            logger.warning('tfconfig.use_autoreloadがTrueですが、validation modeのためFalseに修正します。')
            tfconfig.use_autoreload = False

        is_tfrecord = isinstance(dataset, TFRecordDataSet)
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint is None:
            raise ValueError('In validation, latest checkpoint was not found. dir=' + checkpoint_dir)
        
        if is_tfrecord:
            init_ops = [dataset.validation_iterator_initializer(),
                        self.set_training_false_op ]
            init_fd = { dataset.srclist_ph: list(dataset.validationsrclist) }
        else:
            init_ops = None
            init_fd = None

        with SmartSession(tfconfig) as session:

            # 評価対象となるtensor
            if single_hook:
                evaluating_tensors = validationhook.tensorlist
            else:
                evaluating_tensors = []
                for hook in validationhook:
                    evaluating_tensors.extend(hook.tensorlist)
                    evaluating_tensors = list(set(evaluating_tensors)) # 重複を削除

            # validation opがあるならば
            if hasattr(self, 'validation_op') and self.validation_op is not None:
                evaluating_tensors.append(self.validation_op)

            # 最新のCheckpointを復元する
            try:
                saver = tf.train.Saver()
                saver.restore(session.session, save_path=latest_checkpoint)
                logger.log('validation用チェックポイントの復元に成功しました。')
            except:
                logger.warning('チェックポイントの復元に失敗しました。終了します。')
                return

            # validationがhookとして行われるよう登録する
            session.registerHooks(validationhook)                        
            if is_tfrecord:
                try:
                    # ロードしてからinitしないと、復元の時点でinitしていないことになる
                    session.session.run(init_ops, feed_dict=init_fd)
                    while True:
                        results = session.run(evaluating_tensors, run_hooks=True)
                except tf.errors.OutOfRangeError:
                    pass
                except:
                    print('validation loopでその他の例外を捕捉しました。')
                    import traceback
                    traceback.print_exc()
            else:
                minibatchq = dataset.obtain_serial_minibatch_queue(minibatchsize)
                while minibatchq.hasnext():
                    minibatch = minibatchq.pop()
                    fd = self._create_fd(minibatch, fdmapper, False)
                    results = session.run(evaluating_tensors, feed_dict=fd, run_hooks=True)
        return


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

name = 'singlenetworkmodel'


        
