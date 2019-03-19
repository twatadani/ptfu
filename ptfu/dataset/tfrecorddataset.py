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
        label_dtype: 辞書形式。{ 'label1': tf.dtype, ...}で指定する。 '''
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

        if 'parallel' in options:
            self.parallel = options['parallel']
        else:
            self.parallel = 3

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
        minibatch_op = tf.cond(training_tensor,
            lambda: self.train_iterator.get_next(),
            lambda: self.validation_iterator.get_next(),
            name = 'minibatch_selector'
        )
        return minibatch_op

    def _record_parse(self, example):
        ''' 内部的に使用するTFRecordのパース用関数 '''
        featuredict = {}
        for l in self.labellist:
            featuredict[l] = tf.FixedLenFeature([], dtype=tf.string)

        #print(featuredict)
        features = tf.parse_single_example(example,features = featuredict)
        #print(features)
        parseddict = {}
        for l in self.labellist:
            value = features[l]
            dtype = self.label_dtype[l] if self.label_dtype is not None else tf.float32
            if dtype in (tf.float16, tf.float32, tf.float64, tf.int32, tf.uint16, tf.uint8, tf.int16, tf.int8, tf.int64): # これらの型しか許されない
                #print(l)
                decoded = tf.io.decode_raw(value, out_type = dtype)
                if self.tensorshape is None:
                    parseddict[l] = decoded
                    #parsedlist.append(decoded)
                elif l in self.tensorshape:
                    parseddict[l] = tf.reshape(tensor = decoded,
                                               shape = self.tensorshape[l])
                    #parsedlist.append(tf.reshape(tensor = decoded,
                    #                             shape = self.tensorshape[l]))
                else:
                    parseddict[l] = decoded
            elif dtype == tf.string:
                #print('dtypeがtf.stringです label:', l)
                #print('valueのtype:', type(value))
                #print('valueのshape:', value.shape)
                #decoded = tf.io.decode_raw(value, out_type = dtype)
                #print('decodedのtype:', type(decoded))
                parseddict[l] = value
        return parseddict

    def _prepare_train_iterator(self, minibatchsize):
        ''' 学習時用のminibatch iteratorを準備する '''
        from ..kernel import kernel
        dataset = tf.data.TFRecordDataset(self.srclist_ph, compression_type='GZIP')
        if hasattr(tf.data.experimental, 'shuffle_and_repeat'):
            dataset_shuffle_repeat = dataset.apply(
                tf.data.experimental.shuffle_and_repeat(buffer_size = minibatchsize * 10))
        else:
            dataset_shuffle_repeat = dataset.apply(
                tf.contrib.data.shuffle_and_repeat(buffer_size = minibatchsize * 10))
        if hasattr(tf.data.experimental, 'map_and_batch'):
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
        prefetch = dataset_map_batch.prefetch(buffer_size = minibatchsize)
        self.train_iterator = prefetch.make_initializable_iterator()
        self.train_initializer = self.train_iterator.initializer

        # validation用
        if hasattr(tf.data.experimental, 'map_and_batch'):
            validation_map_batch = dataset.apply(tf.data.experimental.map_and_batch(
                map_func = self._record_parse,
                batch_size = minibatchsize,
                num_parallel_calls = self.parallel))
        else:
            validation_map_batch = dataset.apply(tf.contrib.data.map_and_batch(
                map_func = self._record_parse,
                batch_size = minibatchsize,
                num_parallel_calls = self.parallel))
        validation_prefetch = validation_map_batch.prefetch(buffer_size = minibatchsize)
        self.validation_iterator = validation_prefetch.make_initializable_iterator()
        self.validation_initializer = self.validation_iterator.initializer
        return

    def train_iterator_initializer(self):
        ''' dataset iteratorのinitializer opを返す '''
        return self.train_initializer

    def validation_iterator_initializer(self):
        return self.validation_initializer
