''' dataset.py - datasetモジュール: DataSetを定義する '''

from enum import Enum, auto

class LabelStyle(Enum):
    ''' データセット内のラベル付けの方式を表現するenum '''

    LABEL_BY_FILENAME = auto() # ファイル名によるラベル
    LABEL_BY_CSV = auto() # CSVファイルによるラベル
    LABEL_BY_FUNC = auto() # フィルター関数でラベルを切り分ける方式 フィルター関数を別に提供する必要がある
    TFRECORD = auto() # TFRecord内ですでにラベル付けされている状態

class DataSet:
    ''' 1組のデータセットを表現するクラス 
    本ライブラリの枠組みでは、1件のデータは
    { label1: ..., label2: ..., label3: ..., }
    といった辞書データ形式で表現される。
    例えば、'img'のみからなるデータは { 'img': binarydata }、
    'img'と'label'からなるデータは{ 'img': binarydata, 'label': 'somelabel' }
    という形式になる。
    '''

    def __init__(self, srclist, labellist, labelstyle, **options):
        ''' DataSetのイニシャライザ
        通常、ライブラリユーザはこのイニシャライザを直接呼び出すのではなく、load関数を使う。
        srclist: データセット格納ファイルのリスト
        labellist: ['label1, 'label2', ...]のようなデータラベルのリスト
        labelstyle: どのような方式でラベルデータを設定するか。LabelStyle enumから選択する。
        options: labelstyleに依存するオプション '''

        if hasattr(srclist, __getitem__):
            self.srclist = srclist
        elif srclist is not None:
            self.srclist = [self.srclist]
        else:
            self.srclist = []

        self.labellist = labellist
        self.labelstyle = labelstyle
        self.options = options
        return

    def obtain_minibatch(self, minibatchsize):
        ''' minibatchsizeで指定されたサイズのミニバッチを取得する。
        ミニバッチデータは
        { label1: (minibatchsize, ....), label2: (minibatchsize, ....) }
        の形式 '''
        raise NotImplementedError

class NPYDataSet(DataSet):
    ''' NPYファイルで格納されているデータセット '''

    import random
    import numpy as np

    def __init__(self, srclist, labellist, labelstyle, **options):
        ''' NPUYDataSetのイニシャライザ 
        options内に
        storetype: StoreTypeメンバ
        labelfunc: 1データ当たりのlabel切り分け辞書 = labelfunc(name, data)
        となるようなstoretype, labelfuncが必須 '''
        super(NPYDataSet, self).__init__(srclist, labellist, labelstyle, options)

        self.readers = []
        self.namelist = [] # データネームのリスト 実際はreaderごとにリストにするので二重リスト
        self.list_for_minibatch = [] # ミニバッチ取得用のリスト
        assert labelfunc is not None
        self.labelfunc = options[labelfunc]
        for srcfile in srclist:
            reader = SrcReader(DataType.NPY, options[storetype], srcfile)
            self.readers.append(reader)
            namelist.append(reader.namelist())
        return

    def obtain_minibatch(self, minibatchsize):
        ''' minibatchsizeで指定されたサイズのミニバッチを取得する '''
        
        if len(self.list_for_minibatch) < minibatchsize:
            self._extend_list

        # 返り値となるdictを準備
        minibatch_dict = None

        # ミニバッチ対象となる名前リストをスライスする
        minibatch_namelist = self.list_for_minibatch[0:minibatchsize]
        # データを取得する
        for name in minibatch_namelist:
            for i in range(len(readers)):
                if name in namelist[i]:
                    data = readers[i].getbyname(name)
                    datadict = self.labelfunc(name, data)
                    minibatch_dict = self._mergedict(minibatch_dict, datadict)
        
        # 取得したミニバッチ分のデータを削除する
        self.list_for_minibatch = self.list_for_minibatch[minibatchsize:
                                                          len(self.list_for_minibatch)]
        return minibatch_dict


    @staticmethod
    def _mergedict(minibatch_dict, datadict):
        ''' ミニバッチデータの辞書に1件のデータ辞書をmergeし、新しいminibatch dictを返す内部用関数 '''
        if minibatch_dict is None:
            return datadict
        else:
            for key in datadict.keys():
                value = np.concatenate(minibatch_dict[key], datadict[key], axis=0)
                minibatch_dict[key] = value
            return minibatch_dict

    def _extend_list(self):
        ''' minibatchリストが短くなったときに延長する内部用関数 '''
        newnamelist = []
        for sublist in self.namelist:
            newnamelist.extend(sublist)
        random.shuffle(newnamelist)
        self.list_for_minibatch.extend(newnamelist)
        return

class TFRecordDataSet(DataSet):
    ''' TFRecord形式のデータセット '''

    import tensorflow as tf

    def __init__(self, srclist, labellist, **options):
        ''' TFRecordDataSetのイニシャライザ '''
        super(TFRecordDataSet, self).__init__(srclist, labellist, None, options)
        
        if 'label_dtype' in options:
            self.label_dtype = options[label_dtype]
        else:
            self.label_dtype = None

        if 'parallel' in options:
            self.parallel = options[parallel]
        else:
            self.parallel = 3

    def obtain_minibatch(self, minibatchsize):
        ''' ミニバッチデータを取得する。TFRecordDataSetの場合は、実際にはミニバッチデータを取得するオペレーションを定義する '''
        dataset_shuffle_repeat = tf.data.TFRecordDataset(self.srclist).apply(
            tf.contrib.data.shuffle_and_repeat(buffer_size = minibatchsize * 2))
        dataset_map_batch = dataset_shuffle_repeat.apply(tf.contrib.data.map_and_batch(
            map_func = self._record_parse, #todo
            batch_size = minibatchsize,
            num_parallel_calls = self.parallel)) # todo
        prefetch = dataset_map_batch.prefetch(buffer_size = 10)
        iterator = prefetch.make_one_shot_iterator()
        return iterator.get_next()

    def _record_parse(self, example):
        ''' 内部的に使用するTFRecordのパース用関数 '''
        featuredict = {}
        for l in self.labellist:
            featuredict[l] = tf.FixedLenFeature([], dtype=tf.string)

        features = tf.parse_single_example(example,features = featuredict)

        parseddict = {}
        for l in self.labellist:
            value = features[l]
            dtype = self.label_dtype[l] if self.label_dtype is not None else tf.float32
            parseddict[l] = tf.io.decode_row(value, out_type = dtype)
        return parseddict
        
                                               
        

                                                




        
