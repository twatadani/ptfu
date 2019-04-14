''' classifier.py: 分類タスクを一つのネットワークで行うClassifierを記述する '''

from .singlenetworkmodel import SingleNetworkModel

import tensorflow as tf

class Classifier(SingleNetworkModel):
    ''' 分類タスクを行う単一ネットワークモデル '''

    def __init__(self, neural_network, optimizer, label_tensor_key, prediction_tensor_key, classlist, metrics_duration=1000, in_train_validation_interval=1000):
        '''
        label_tensor_key: nn.get_input_tensors[label_tensor_key]でlabel tensorが取得できるようなキー
        prediction_tensor_key: nn.get_output_tensor[prediction_tensor_key]でprediction tensorが取得できるようなキー
        classlist: ['car', 'signal', 'cat', 'dog'] 
        metrics_duration: accracy
        in_train_validation_interval: in train validation '''
        super(Classifier, self).__init__(neural_network, optimizer)
        self.label_tensor_key = label_tensor_key
        self.prediction_tensor_key = prediction_tensor_key
        self.classlist = classlist
        self.nclasses = len(classlist) # 分類するクラスの数
        self.metrics_duration = metrics_duration
        self.in_train_validation_initialized = False # in_train_validationの初期化
        self.in_train_validation_interval = in_train_validation_interval
        return

    def define_network(self, tfconfig, minibatchsize_per_tower, validation_datasize):
        ''' パラメータを与えてネットワークを定義する '''
        from ..tfconfig import TFConfig

        self._define_network_common(tfconfig, minibatchsize_per_tower)
        
        # 分類タスク用のパラメータ定義
        metric_device = None
        if tfconfig.use_gpu:
            if tfconfig.use_xla:
                metric_device = TFConfig.XLA_GPU
            else:
                metric_device = tfconfig.towers[0][-1]
        else:
            if tfconfig.use_xla:
                metric_device = TFConfig.XLA_CPU
            else:
                metric_device = TFConfig.CPU

        gstep_tensor = self.global_step_tensor()
        label_tensor = tf.to_int64(self.nn.get_input_tensors()[self.label_tensor_key])
        prediction_tensor = self.nn.get_output_tensors()[self.prediction_tensor_key]

        #assert False, metric_device
        
        with tf.name_scope('classifier_metrics'):
            with tf.device(metric_device):
                prediction_hot = tf.argmax(prediction_tensor,
                                           axis = 1,
                                           name = 'prediction_hot')
                prediction_onehottensor = tf.one_hot(indices = prediction_hot,
                                                     depth = self.nclasses,
                                                     dtype = tf.int64,
                                                     name = 'prediction_one_hot')

                minibatchsize = minibatchsize_per_tower * tfconfig.ntowers()
                mod_metrics = self.metrics_duration % minibatchsize
                if mod_metrics == 0:
                    dursize = self.metrics_duration
                else:
                    dursize = self.metrics_duration + (minibatchsize - (mod_metrics))
                zeros_duration = tf.zeros(shape = (dursize, self.nclasses),
                                          dtype = tf.int64,
                                          name = 'zeros_duration')
                labels_duration = tf.Variable(
                    initial_value = zeros_duration,
                    dtype = tf.int64,
                    trainable = False,
                    name = 'labels_duration')
                predictions_duration = tf.Variable(
                    initial_value = zeros_duration,
                    dtype = tf.int64,
                    trainable = False,
                    name = 'predictions_duration')

                zeros_validation = tf.zeros(shape = (validation_datasize, self.nclasses),
                                            dtype = tf.int64,
                                            name = 'zeros_validation')

                labels_validation = tf.Variable(
                    initial_value = zeros_validation,
                    dtype = tf.int64,
                    trainable = False,
                    name = 'labels_validation')
                predictions_validation = tf.Variable(
                    initial_value = zeros_validation,
                    dtype = tf.int64,
                    trainable = False,
                    name = 'predictions_validation')

                validation_step = self.validation_step_tensor()
                self.validation_step_reset_op = tf.assign(
                    ref = validation_step,
                    value = 0,
                    name = 'validation_step_reset_op')
                self.validation_step_increment_op = tf.assign_add(
                    ref = validation_step,
                    value = 1,
                    name = 'validation_step_increment_op')

        # train用
        self._define_metrics(tfconfig, metric_device, gstep_tensor,
                             minibatchsize, dursize,
                             labels_duration, predictions_duration,
                             label_tensor, prediction_onehottensor, train=True)

        # validation用
        self._define_metrics(tfconfig, metric_device, validation_step,
                             minibatchsize, validation_datasize,
                             labels_validation, predictions_validation,
                             label_tensor, prediction_onehottensor, train=False)

        return

    def _define_metrics(self, tfconfig, metrics_device, step_tensor,
                        minibatchsize, metrics_duration, 
                        label_buffer, prediction_buffer, 
                        label_tensor, prediction_tensor, train=True):
        ''' train, validation用それぞれにmetricsを定義する '''
        import tensorflow as tf

        nclasses = label_tensor.shape[-1].value
        
        if train:
            prefix = 'train_'
        else:
            prefix = 'validation_'

        with tf.name_scope('classifier_metrics'):
            with tf.device(metrics_device):
                slice_start = tf.to_int32(tf.mod(step_tensor * minibatchsize, metrics_duration,
                                                 name = prefix + 'metrics_slice_start'))
                assert slice_start.dtype == tf.int32, slice_start.dtype
                actual_batchsize = tf.math.minimum(
                    tf.to_int32(minibatchsize),
                    tf.to_int32(tf.shape(label_tensor)[0]),
                    name = 'actual_batchsize')
                assert actual_batchsize.dtype == tf.int32, actual_batchsize.dtype
                slice_end = tf.math.minimum(
                    tf.to_int32(metrics_duration),
                    tf.add(slice_start, actual_batchsize),
                    name = prefix + 'metrics_slice_end')
                assert slice_end.dtype == tf.int32, slice_end.dtype
                label_sliced = label_buffer[slice_start:slice_end]
                prediction_sliced = prediction_buffer[slice_start:slice_end]

                if train:
                    self.labels_duration_update_op = tf.assign(
                        ref = label_sliced,
                        value = label_tensor,
                        name = 'labels_duration_update_op')
                    self.predictions_duration_update_op = tf.assign(
                        ref = prediction_sliced,
                        value = prediction_tensor,
                        name = 'predictions_duration_update_op')
                else:
                    self.labels_validation_update_op = tf.assign(
                        ref = label_sliced,
                        value = label_tensor[0:tf.shape(label_sliced)[0]], # minibatchが小さいときのため必要
                        name = 'labels_validation_update_op')
                    self.predictions_validation_update_op = tf.assign(
                        ref = prediction_sliced,
                        value = prediction_tensor[0:tf.shape(prediction_sliced)[0]], # minibatchが小さいときのため必要
                        name = 'predictions_validation_update_op')

                # classwise metricsの集計
                classwise_tps = []
                classwise_fps = []
                classwise_tns = []
                classwise_fns = []
                classwise_accuracies = []
                classwise_precisions = []
                classwise_recalls = []
                classwise_specificities = []
                for i, clazz in enumerate(self.classlist):
                    # tf.boolとintのcast -> True: 1, False: 0 (実験にて証明済み)
                    classwise_predictions = tf.squeeze(prediction_buffer[:, i], name= prefix + 'predictions_'+clazz) #[dursize]
                    assert classwise_predictions.shape == (metrics_duration,), print(classwise_predictions.shape)
                    classwise_predictions_bool = tf.cast(classwise_predictions, tf.bool, name=prefix+'predictions_bool_'+clazz) #[dursize], dtype=bool
                    classwise_labels = tf.squeeze(label_buffer[:, i], name=prefix+'classwise_labels_'+clazz) #[dursize]
                    assert classwise_labels.shape == (metrics_duration,), print(classwise_labels.shape)
                    classwise_labels_bool = tf.cast(classwise_labels, dtype=tf.bool, name=prefix+'labels_bool_'+clazz) #[dursize], dtype=bool
                
                    # True Positive:
                    # prediction 1, label 1のときに1, その他は0 -> かけ算で達成できる
                    cls_tp = tf.to_float(tf.reduce_sum(tf.multiply(classwise_predictions, classwise_labels), name=prefix+'tp_'+clazz)) # scalar
                    assert cls_tp.shape == (), print(cls_tp.shape)
                    classwise_tps.append(tf.expand_dims(cls_tp, axis=0))
                
                    # False Positive:
                    # prediction 1, label 0のときに1, その他は0
                    # prediction xor label は一致しないときのみ1
                    # prediction * (prediction_bool xor label_bool)で達成できる
                    # for tensorflow version compatibility
                    try:
                        assert hasattr(tf.math, 'logical_xor')
                    except AssertionError:
                        tf.math.logical_xor = tf.logical_xor
                    xor = tf.math.logical_xor(classwise_predictions_bool, classwise_labels_bool, name=prefix+'xor_'+clazz)
                    xor_recast = tf.cast(xor, dtype=tf.int64, name=prefix+'xor_recasted_'+clazz)
                    cls_fp = tf.to_float(tf.reduce_sum(tf.multiply(classwise_predictions, xor_recast), name=prefix+'fp_'+clazz)) # scalar
                    assert cls_fp.shape == (), print(cls_fp.shape)
                    classwise_fps.append(tf.expand_dims(cls_fp, axis=0))

                    # True Negative
                    # prediction 0, label 0のときに1, その他は0
                    # ORの否定
                    tn_bool = tf.math.logical_not(tf.math.logical_or(classwise_predictions_bool, classwise_labels_bool))
                    cls_tn = tf.to_float(tf.reduce_sum(tf.cast(tn_bool, dtype=tf.int64), name=prefix+'tn_'+clazz)) #scalar
                    assert cls_tn.shape == (), print(cls_tn.shape)
                    classwise_tns.append(tf.expand_dims(cls_tn, axis=0))

                    # False Negative
                    # prediction 0, label 1のときに1, その他は0
                    # label * (prediction xor label)
                    cls_fn = tf.to_float(tf.reduce_sum(tf.multiply(classwise_labels, xor_recast), name=prefix+'fn_'+clazz)) # scalar
                    assert cls_fn.shape == (), print(cls_fn.shape)
                    classwise_fns.append(tf.expand_dims(cls_fn, axis=0))

                    # Accuracy: (TP + TN) / (TP + TN + FP + FN)
                    cls_accuracy = tf.div(cls_tp + cls_tn, tf.maximum(1.0, cls_tp + cls_tn + cls_fp + cls_fn),
                                          name = prefix+'accuracy_' + clazz)
                    assert cls_accuracy.shape == (), print(cls_accuracy.shape)
                    classwise_accuracies.append(tf.expand_dims(cls_accuracy, axis=0))

                    # Precision = PPV: TP / (TP + FP)
                    cls_precision = tf.div(cls_tp, tf.maximum(1.0, cls_tp + cls_fp),
                                           name = prefix+'precision_' + clazz)
                    assert cls_precision.shape == (), print(cls_precision.shape)
                    classwise_precisions.append(tf.expand_dims(cls_precision, axis=0))

                    # Recall = Sensitivity: TP / (TP + FN)
                    cls_recall = tf.div(cls_tp, tf.maximum(1.0, cls_tp + cls_fn),
                                        name = prefix+'recall_' + clazz)
                    assert cls_recall.shape == (), print(cls_recall.shape)
                    classwise_recalls.append(tf.expand_dims(cls_recall, axis=0))

                    # Specificity: TN / (FP + TN)
                    # specificityは2クラス以外の時はあまり意味をなさないため、2クラスのときのみ定義する 
                    if nclasses == 2:
                        cls_specificity = tf.div(cls_tn, tf.maximum(1.0, cls_fp + cls_tn),
                                                 name = prefix+'specificity_' + clazz)
                        assert cls_specificity.shape == (), print(cls_specificity.shape)
                        classwise_specificities.append(tf.expand_dims(cls_specificity, axis=0))

                    # F mesures: 2 * (recall * precision) / ( recall + precision)
                    cls_fmeasures = tf.div(2 * cls_recall * cls_precision, cls_recall + cls_precision,
                                           name = prefix+'Fmeasures_' + clazz)
        
                    # Tensorboardへの登録
                    if tfconfig.use_summary:
                        fname = prefix+'metrics_' + clazz
                        tf.summary.scalar(name='accuracy_'+clazz, tensor=cls_accuracy, family=fname)
                        tf.summary.scalar(name='precision(PPV)_'+clazz, tensor=cls_precision, family=fname)
                        tf.summary.scalar(name='recall(Sensitivity)_'+clazz, tensor=cls_recall, family=fname)
                        if nclasses == 2:
                            tf.summary.scalar(name='specificity_'+clazz, tensor=cls_specificity, family=fname)
                        tf.summary.scalar(name='Fmeasures_'+clazz, tensor=cls_fmeasures, family=fname)
                        tf.summary.scalar(name='tp_'+clazz, tensor=cls_tp, family=fname)
                        tf.summary.scalar(name='fp_'+clazz, tensor=cls_fp, family=fname)
                        tf.summary.scalar(name='tn_'+clazz, tensor=cls_tn, family=fname)
                        tf.summary.scalar(name='fn_'+clazz, tensor=cls_fn, family=fname)

                # 全体のmetrics評価

                # macro precision/recall/specificity/fmeasures
                # classwiseに計算したmetricsの平均をとる
                # accuracyについてはaverage accuracyという名になる
                macro_precision = tf.reduce_mean(tf.concat(classwise_precisions, axis=0), name=prefix+'macro_precision')
                macro_recall = tf.reduce_mean(tf.concat(classwise_recalls, axis=0), name=prefix+'macro_recall')
                #macro_specificity = tf.reduce_mean(tf.concat(classwise_specificities, axis=0), name=prefix+'macro_specificity')
                macro_fmeasures = tf.div(2 * macro_recall * macro_precision, macro_recall + macro_precision,
                                    name = prefix+'macro_fmeasures')
                average_accuracy = tf.reduce_mean(tf.concat(classwise_accuracies, axis=0), name=prefix+'average_accuracy')
                if train:
                    self.average_accuracy = average_accuracy

                # TensorBoardへの登録
                if tfconfig.use_summary:
                    fname = prefix+'macro_metrics'
                    tf.summary.scalar(name='macro_precision(PPV)', tensor=macro_precision, family=fname)
                    tf.summary.scalar(name='macro_recall(Sensitivity)', tensor=macro_recall, family=fname)
                    #tf.summary.scalar(name='macro_specificity', tensor=macro_specificity, family=fname)
                    tf.summary.scalar(name='macro_fmeasures', tensor=macro_fmeasures, family=fname)
                    tf.summary.scalar(name='average_accuracy', tensor=average_accuracy, family=fname)

                # micro precision/recall/specificity/fmeasures
                # accuracyについてはOverall accuracyという名になる
                # micro precision
                # Precision = PPV: TP / (TP + FP)
                tpsum = tf.reduce_sum(tf.concat(classwise_tps, axis=0))
                fpsum = tf.reduce_sum(tf.concat(classwise_fps, axis=0))
                micro_precision = tf.div(tpsum, tf.maximum(1.0, tpsum + fpsum), name=prefix+'micro_precision')
                # micro recall
                # Recall = Sensitivity: TP / (TP + FN)
                fnsum = tf.reduce_sum(tf.concat(classwise_fns, axis=0))
                micro_recall = tf.div(tpsum, tf.maximum(1.0, tpsum + fnsum), name=prefix+'micro_recall')
                # micro specificity
                # # Specificity: TN / (FP + TN)
                #tnsum = tf.reduce_sum(tf.concat(classwise_tns, axis=0))
                #micro_specificity = tf.div(tnsum, tf.maximum(1.0, fpsum + tnsum), name=prefix+'micro_specificity')
                # micro fmeasures
                micro_fmeasures = tf.div(2 * micro_recall * micro_precision, micro_recall + micro_precision, name=prefix+'micro_fmeasures')

                # Overall Accuracy
                # 検討した結果、TP+TNではなくTPsumが分子なのが正しい
                # 分母はFNsum
                overall_accuracy = tf.div(tpsum, tf.maximum(1.0, tpsum + fnsum), name=prefix+'overall_accuracy')
                if train:
                    self.overall_accuracy = overall_accuracy
                else:
                    self.validation_overall_accuracy = overall_accuracy
                    self.validation_micro_precision = micro_precision
                    self.validation_micro_recall = micro_recall
                    self.validation_micro_fmeasures = micro_fmeasures

                # debug metrics
                if tfconfig.use_summary:
                    fname = 'micro_metrics_debug'
                    tf.summary.scalar(name='micro_tpsum', tensor=tpsum, family=fname)
                    tf.summary.scalar(name='micro_fpsum', tensor=fpsum, family=fname)
                    tf.summary.scalar(name='micro_fnsum', tensor=fnsum, family=fname)

                # TensorBoardへの登録
                if tfconfig.use_summary:
                    fname = prefix+'micro_metrics'
                    tf.summary.scalar(name='micro_precision(PPV)', tensor=micro_precision, family=fname)
                    tf.summary.scalar(name='micro_recall(Sensitivity)', tensor=micro_recall, family=fname)
                    #tf.summary.scalar(name='micro_specificity', tensor=micro_specificity, family=fname)
                    tf.summary.scalar(name='micro_fmeasures', tensor=micro_fmeasures, family=fname)
                    tf.summary.scalar(name='overall_accuracy', tensor=overall_accuracy, family=fname)
        return


    def prepare_train(self, tfconfig, validation_dataset = None,
                      tf_loss_func=None,
                      fdmapper = None, **tf_loss_func_options):
        ''' パラメータを与えて損失関数定義などの学習準備を行う
        tfconfig: TFConfigオブジェクト
        tf_loss_func: TensorFlowの損失関数
        Classifierではさらに集計のためのオペレーションを加える '''
        from ..loopsmartsessionhook import LoopSmartSessionHook
        print('Classifier prepare_trainが呼び出されました。')
        self._prepare_train_common(tfconfig, tf_loss_func, **tf_loss_func_options)
        self.train_op = tf.group(self.train_op,
                                 self.labels_duration_update_op,
                                 self.predictions_duration_update_op,
                                 name = 'classifier_train_op')
        self.validation_op = tf.group(self.labels_validation_update_op,
                                      self.predictions_validation_update_op,
                                      self.validation_step_increment_op,
                                      name = 'classifier_validation_op')
        

        if validation_dataset is not None:
            # in_train_validationのhook登録
            itvalidationhook = LoopSmartSessionHook(
                hook_func = self.in_train_validation,
                hook_step = self.in_train_validation_interval,
                hook_mod = 0,
                synchronous = True,
                required_tensor_list = [],
                hook_name = 'In_train_validation',
                self_or_cls = self,
                validation_dataset = validation_dataset,
                fdmapper = fdmapper)
            # 最初と最後に実行されるように設定
            itvalidationhook.set_as_initial_hook()
            itvalidationhook.set_as_final_hook()
            # hookが実行されるように登録する
            self.register_trainhook(itvalidationhook)
        
        return


    def in_train_validation(self, tensorvaluedict, validation_dataset, fdmapper=None):
        ''' 学習中に成績確認を行うためのvalidationを実行するhook用関数 '''
        from ..dataset import TFRecordDataSet
        from ..logger import get_default_logger
        logger = get_default_logger()

        logger.log('In train validationを実行しています。')
        # 準備
        # トレーニングをオフにする
        self.session.session.run(self.set_training_false_op)

        is_tfrecord = isinstance(validation_dataset, TFRecordDataSet)
        if is_tfrecord:
            # validation用のイテレータを初期化する
            fd = { validation_dataset.srclist_ph: list(validation_dataset.validationsrclist) }
            self.session.session.run(validation_dataset.validation_iterator_initializer(),
                             feed_dict = fd)
            self.in_train_validation_initialized = True

        self.session.run(self.validation_step_reset_op, run_hooks=False)
        if is_tfrecord:
            try:
                while True:
                    self.session.run(self.validation_op, run_hooks=False)
            except tf.errors.OutOfRangeError:
                pass
                
        else:
            q = validation_dataset.obtain_serial_minibatch_queue(minibatchsize)
            while q.hasnext():
                minibatch = q.pop()
                fd = self._create_fd(minibatch, fdmapper, False)
                self.session.run(self.validation_op, feed_dict=fd, run_hooks=False)

        # accuracyを取り出す
        v_accuracy = self.session.session.run(self.validation_overall_accuracy)

        # 後片付け
        self.session.session.run(self.set_training_true_op)
        if is_tfrecord:
            # random minibatchをイニシャライズする
            fd = {
                validation_dataset.srclist_ph: list(validation_dataset.srclist)
            }
            self.session.session.run(validation_dataset.train_iterator_initializer(), feed_dict=fd)
        logger.log('In train validationを終了します。validation overall accuracy = ' + str(v_accuracy))
        return

name = 'classifier'

