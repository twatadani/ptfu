''' tfrecorddataset.py: TFRecordDataSetを記述する '''

from .dataset import DataSet
import tensorflow as tf

class TFRecordDataSet(DataSet):
    ''' TFRecord形式のデータセット '''

    def __init__(self, srclist, labellist, validationsrclist=None, **options):
        ''' TFRecordDataSetのイニシャライザ
        validationsrclist: Classifierなどin_train_validationを行う場合のvalidation tfrecordファイルのリスト
        optionsの与え方
        minibatchsize: 必須オプション。学習・検証のミニバッチサイズを与える
        tensorshape: 必須ではないが推奨。辞書型でTFRecord内のラベルがキー、tensorのshapeが値。但しミニバッチ次元は削減した形で与える。たとえば(-1, 32, 32, 3)で動かすネットワークでは(32, 32, 3)を与える。
        augment_func_dict: data augmentationのための関数を格納した辞書。キーにtensorのラベル文字列、valueに関数を与える。例えば、 { 'data': func1, ... }
        validation_augment_func_dict: validation時のdata augmentation関数を格納した辞書。書式はaugment_func_dictと同様。
        個々の関数については、 def func1(tensor)でreturn valueもtensorとなるようにする
        label_dtype: 辞書形式。{ 'label1': tf.dtype, ...}で指定する。 
        gpu_devices: 必須ではない。使用可能なGPUデバイスのリスト。augmentationの処理が重いときなどに用いる
        '''
        from .dataset import LabelStyle
        super(TFRecordDataSet, self).__init__(srclist, labellist, LabelStyle.TFRECORD,
                                              None, # datatypeはNone
                                              **options)
        
        # TFRecordDataSetではminibatchsizeを予め与えておく必要がある
        assert 'minibatchsize' in options, 'TFRecordDataSetの初期化にはminibatchsizeが必要です'
        self.minibatchsize = options['minibatchsize']

        self.validationsrclist = validationsrclist

        # データセット内のtensor shapeを持つ
        if 'tensorshape' in options:
            self.tensorshape = options['tensorshape']
        else:
            self.tensorshape = None

        # srclistをグラフ内に持つ
        self.srclist_ph = tf.placeholder(dtype = tf.string,
                                         shape = (None),
                                         name='src_list')
        
        if 'label_dtype' in options:
            print('label_dtype found')
            self.label_dtype = options['label_dtype']
        else:
            print('label_dtype NOT found')
            self.label_dtype = None

        if 'augment_func_dict' in options:
            assert isinstance(options['augment_func_dict'], dict)
            self.augment_func_dict = options['augment_func_dict']
        else:
            self.augment_func_dict = {}

        if 'validation_augment_func_dict' in options:
            assert isinstance(options['validation_augment_func_dict'], dict)
            self.validation_augment_func_dict = options['validation_augment_func_dict']
        else:
            self.validation_augment_func_dict = {}

        if 'parallel' in options:
            self.parallel = options['parallel']
        else:
            self.parallel = 3

        if 'gpu_devices' in options:
            self.gpu_devices = options['gpu_devices']
        else:
            self.gpu_devices = None

        # 学習用のminibatch iteratorを準備する
        self._prepare_train_iterator(self.minibatchsize)
        return

    def datanumber(self):
        ''' このデータセット内のデータ数を得る '''
        return self._datanumber_common(self.srclist)

    def validation_datanumber(self):
        ''' 検証用データセット内のデータ数を得る '''
        return self._datanumber_common(self.validationsrclist)

    @staticmethod
    def _datanumber_common(srclist):
        ''' srclistで指定されたTFRecord datasetのデータ数を得る '''
        import tensorflow as tf
        options = tf.python_io.TFRecordOptions(
            compression_type = tf.python_io.TFRecordCompressionType.GZIP)
        number = 0
        for src in srclist:
            iterator = tf.python_io.tf_record_iterator(src, options=options)
            number += len(list(iterator))
        return number

    def obtain_minibatch(self, minibatchsize):
        ''' ミニバッチデータを取得する。TFRecordDataSetの場合は、実際にはミニバッチデータを取得するオペレーションを定義する '''
        from ..kernel import kernel
        training_tensor = kernel.get_training_tensor()
        minibatch_op = tf.cond(training_tensor, # prediction
            lambda: self.train_iterator.get_next(), # true func
            lambda: self.validation_iterator.get_next(), # false func
            name = 'minibatch_selector'
        )
        return minibatch_op

    def _record_parse(self, example):
        ''' 内部的に使用するTFRecordのパース用関数 '''
        from ..kernel import kernel
        featuredict = { l: tf.FixedLenFeature([], dtype=tf.string) for l in self.labellist }
        #featuredict = {}
        #for l in self.labellist:
            #featuredict[l] = tf.FixedLenFeature([], dtype=tf.string)

        features = tf.parse_single_example(example,features = featuredict)
        training_tensor = kernel.get_training_tensor()
        parseddict = {}
        for l in self.labellist:
            value = features[l]
            dtype = self.label_dtype[l] if self.label_dtype is not None else tf.float32
            if dtype in (tf.float16, tf.float32, tf.float64, tf.int32, tf.uint16, tf.uint8, tf.int16, tf.int8, tf.int64): # これらの型しか許されない
                decoded = tf.io.decode_raw(value, out_type = dtype)
                if self.tensorshape is None:
                    parseddict[l] = decoded
                elif l in self.tensorshape:
                    reshaped = tf.reshape(tensor = decoded, shape = self.tensorshape[l])
                    if l in self.augment_func_dict:
                        train_augment = self.augment_func_dict[l](reshaped)
                    else:
                        train_augment = reshaped
                    if l in self.validation_augment_func_dict:
                        validation_augment = self.validation_augment_func_dict[l](reshaped)
                    else:
                        validation_augment = reshaped
                    parseddict[l] = tf.cond(
                        training_tensor, # prediction
                        lambda: train_augment, # true func
                        lambda: validation_augment # false func
                    )
                    if l in self.augment_func_dict or l in self.validation_augment_func_dict:
                        non_augment_label = l + '_noaugment'
                        parseddict[non_augment_label] = reshaped
                else:
                    parseddict[l] = decoded
            elif dtype == tf.string:
                parseddict[l] = value
        return parseddict

    def _prepare_train_iterator(self, minibatchsize):
        ''' 学習時用のminibatch iteratorを準備する '''
        from ..kernel import kernel
        dataset = tf.data.TFRecordDataset(self.srclist_ph, compression_type='GZIP')
        if hasattr(tf.data, 'experimental') and hasattr(tf.data.experimental, 'shuffle_and_repeat'):
            dataset_shuffle_repeat = dataset.apply(
                tf.data.experimental.shuffle_and_repeat(buffer_size = minibatchsize * 10))
        else:
            dataset_shuffle_repeat = dataset.apply(
                tf.contrib.data.shuffle_and_repeat(buffer_size = minibatchsize * 10))

        map_device = self.gpu_devices[0] if self.gpu_devices is not None else '/CPU:0'

        with tf.device(map_device):
            if hasattr(tf.data, 'experimental') and hasattr(tf.data.experimental, 'map_and_batch'):
                dataset_map_batch = dataset_shuffle_repeat.apply(
                    tf.data.experimental.map_and_batch(
                        map_func = self._record_parse,
                        batch_size = minibatchsize,
                        num_parallel_calls = self.parallel))
            else:
                dataset_map_batch = dataset_shuffle_repeat.apply(tf.contrib.data.map_and_batch(
                    map_func = self._record_parse,
                    batch_size = minibatchsize,
                    num_parallel_calls = self.parallel))
            prefetch = dataset_map_batch.prefetch(buffer_size = 20)
            self.train_iterator = prefetch.make_initializable_iterator()
            self.train_initializer = self.train_iterator.initializer

            # validation用
            if hasattr(tf.data, 'experimental') and hasattr(tf.data.experimental, 'map_and_batch'):
                validation_map_batch = dataset.apply(tf.data.experimental.map_and_batch(
                    map_func = self._record_parse,
                    batch_size = minibatchsize,
                    num_parallel_calls = self.parallel))
            else:
                validation_map_batch = dataset.apply(tf.contrib.data.map_and_batch(
                    map_func = self._record_parse,
                    batch_size = minibatchsize,
                    num_parallel_calls = self.parallel))
            validation_prefetch = validation_map_batch.prefetch(buffer_size = 20)
            self.validation_iterator = validation_prefetch.make_initializable_iterator()
            self.validation_initializer = self.validation_iterator.initializer
            return

    def train_iterator_initializer(self):
        ''' dataset iteratorのinitializer opを返す '''
        return self.train_initializer

    def validation_iterator_initializer(self):
        return self.validation_initializer
