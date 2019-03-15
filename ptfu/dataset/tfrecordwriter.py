''' tfrecordwriter.py: TFRecordに対応したStoreTypeWriter TFRecordWriterを記述する '''

from .archivewriter import ArchiveWriter

import os
import os.path

import tensorflow as tf
import numpy as np

# class内に入れるとpickle化できないため、トップレベルにおく
def default_feature_func(name, datadict):
    ''' name, ndarrayからデフォルトのfeaturesを作成する関数 '''
    feature = {}
    for key in datadict:
        value = datadict[key]
        if isinstance(value, str):
            feature[key] = TFRecordWriter.create_feature(value.encode)
        elif isinstance(value, np.ndarray):
            feature[key] = TFRecordWriter.create_feature(value.tobytes)
        else:
            raise ValueError('default_feature_func cannot process data: type(data) is', type(value))
            
    features = tf.train.Features(feature = feature)
    return features

class TFRecordWriter(ArchiveWriter):
    ''' TFRecord形式のデータセット格納を行うWriter '''

        
    def __init__(self, dstpath, featurefunc=default_feature_func):
        ''' TFRecordWriterのイニシャライザ
        dstpath: 書き込みを行うTFRecordファイル
        featurefunc: _write_func()が呼ばれた際に、name, ndarrayからどのようにtf.train.Featuresを作成するかを規定する関数。tf.train.Features = featurefunc(name, ndarray)となるような関数を与える。Noneの場合はデフォルトが使用される。 '''
        from .storetype import StoreType
        super(TFRecordWriter, self).__init__(StoreType.TFRECORD, dstpath)
        self.featurefunc = featurefunc
        self.writer = None # TFRecordWriter object
        return

    def __exit__(self, exc_type, exc_value, traceback):
        self.writer.__exit__(exc_type, exc_value, traceback)
        return

    def _open_dst(self):
        '''  アーカイブファイルをオープンし, self.fpを設定する。 '''
        parent_dir = os.path.dirname(self.dstpath)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        option = tf.python_io.TFRecordOptions(
            compression_type = tf.python_io.TFRecordCompressionType.GZIP
        )
        self.writer = tf.python_io.TFRecordWriter(self.dstpath, options=option)
        self.writer.__enter__()
        return

    def _close_dst(self):
        ''' アーカイブファイルをクローズする。 '''
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
        return

    def _write_func(self, name, datadict):
        ''' ソースがオープンされていることを前提にname, datadictで与えられる1件のデータを書き込む '''
        features = self.featurefunc(name, datadict)
        example = tf.train.Example(features=features)
        self.writer.write(example.SerializeToString())
        return

    @staticmethod
    def create_feature(encode_func):
        ''' encode_funcを与えてtf.train.Featureを作成する '''
        ret = tf.train.Feature(bytes_list=tf.train.BytesList(
            value=[encode_func()]))
        return ret
    

