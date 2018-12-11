''' dataset.py - datasetモジュール: DataSetを定義する '''

from enum import Enum, auto
import os
import os.path
import pickle

class LabelStyle(Enum):
    ''' データセット内のラベル付けの方式を表現するenum '''

    LABEL_BY_FILENAME = auto() # ファイル名によるラベル
    LABEL_BY_CSV = auto() # CSVファイルによるラベル
    LABEL_BY_FUNC = auto() # フィルター関数でラベルを切り分ける方式 フィルター関数を別に提供する必要がある
    TFRECORD = auto() # TFRecord内ですでにラベル付けされている状態

class DataSetInfo:
    ''' DataSetをマルチプロセスで使用するためのpicklableオブジェクト '''

    def __init__(self, arg_dictionary):
        ''' arg_dictionaryはDataSetオブジェクトを構築するための引数の辞書
        ただし、'class': クラスオブジェクトのフィールドを持つ '''
        self.dic = arg_dictionary
        return
    
    def constructDataSet(self):
        ''' このDataSetInfoからDataSetオブジェクトを構築する '''
        cls = self.dic['class']
        dataset = cls(self.dic['srclist'], self.dic['labellist'], self.dic['labelstyle'],
                      **self.dic['options'])
        return dataset
    

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
        srclist: データセット格納ファイルのリスト
        labellist: ['label1, 'label2', ...]のようなデータラベルのリスト
        labelstyle: どのような方式でラベルデータを設定するか。LabelStyle enumから選択する。
        options: labelstyleに依存するオプション '''

        if isinstance(srclist, list) or isinstance(srclist, tuple):
            self.srclist = srclist
        elif srclist is not None:
            self.srclist = [srclist]
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

    def toDataSetInfo(self):
        ''' このオブジェクトをDataSetInfoに変換する '''
        dic = {}
        cls = self.__class__
        dic['class'] = cls
        dic['srclist'] = self.srclist
        dic['labellist'] = self.labellist
        dic['labelstyle'] = self.labelstyle
        dic['options'] = self.options
        return DataSetInfo(dic)
        

class NPYDataSet(DataSet):
    ''' NPYファイルで格納されているデータセット '''

    # class variable
    diskcachedict = {}
    
    def __init__(self, srclist, labellist, labelstyle, **options):
        ''' NPUYDataSetのイニシャライザ 
        options内に
        storetype: StoreTypeメンバ
        labelfunc: 1データ当たりのlabel切り分け辞書 = labelfunc(name, data)
        となるようなstoretype, labelfuncが必須 
        必須ではないオプション:
        use_diskcache: TrueまたはFalseで指定。デフォルトはFalse
        temporary directoryに一度読み込んだデータについてはlabelfuncを適用した辞書の状態でキャッシュを作成する。二度目以降の読み込みはdisk cacheを優先する '''
        from . import SrcReader, DataType
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
        super(NPYDataSet, self).__init__(srclist, labellist, labelstyle, **options)

        self.readers = []
        self.namelist = [] # データネームのリスト 実際はreaderごとにリストにするので二重リスト

        self.texecutor = ThreadPoolExecutor()
        self.pexecutor = ProcessPoolExecutor(1)

        assert 'labelfunc' in options
        self.labelfunc = options['labelfunc']
        assert 'storetype' in options
        futures = []
        for srcfile in self.srclist:
            reader = SrcReader(DataType.NPY, options['storetype'], srcfile)
            self.readers.append(reader)
            futures.append(self.texecutor.submit(reader.namelist))
        wait(futures)
        for future in futures:
            self.namelist.append(future.result())

        self.namereaderdict = None
        self.nrdfuture = None
        self.nrddone = False
        self.nrdfuture = self.texecutor.submit(self._create_namereaderdict, self.namelist, self.readers)

        #self.namelist_flat = []
        #for sublist in self.namelist:
        #    self.namelist_flat.extend(sublist)

        # disk cacheの設定
        self.use_diskcache = False
        self.tempdir = None
        if 'use_diskcache' in options:
            self.use_diskcache = True
            self._create_tempdir()

        return

    def __del__(self):
        ''' デストラクタ '''
        self._cleanup_tempdir()
        return

    @staticmethod
    def _create_namereaderdict(namelist, readers):
        ''' 格納されている要素の名前からreaderを得られる辞書を作成する '''
        dic = {}
        for i, nlist in enumerate(namelist):
            for name in nlist:
                dic[name] = readers[i]
        return dic

    def _cleanup_tempdir(self):
        ''' テンポラリディレクトリをクリーンアップする '''
        if hasattr(self, 'tempdir') and self.tempdir is not None:
            self.tempdir.cleanup()
            self.tempdir = None
        return

    def _create_tempdir(self):
        ''' テンポラリディレクトリを作成する '''
        from tempfile import TemporaryDirectory

        if not self.srclist[0] in NPYDataSet.diskcachedict:
            if self.tempdir is None:
                print('NPYDataSetのディスクキャッシュを作成します')
                self.tempdir = TemporaryDirectory()
                NPYDataSet.diskcachedict[self.srclist[0]] = self.tempdir
        else:
            print('NPYDataSetのディスクキャッシュを流用します')
            self.tempdir = NPYDataSet.diskcachedict[self.srclist[0]]

    def obtain_minibatch(self, minibatchsize):
        ''' minibatchsizeで指定されたサイズのミニバッチを取得する '''
        import random
        from concurrent.futures import wait

        # name-reader dictの完成を待つ
        if self.nrddone is False:
            wait([self.nrdfuture])
            self.namereaderdict = self.nrdfuture.result()
            self.nrddone = True

        # 返り値となるdictを準備
        minibatch_dict = None

        # ミニバッチ対象となる名前リストを選択する
        minibatch_namelist = random.sample(self.namereaderdict.keys(), minibatchsize)

        # futures
        namefutures = []

        # データを取得する
        for name in minibatch_namelist:
            namefutures.append(
                self.texecutor.submit(self._obtain_name, name))

        wait(namefutures)
        datadicts = map(lambda x: x.result(), namefutures)
        keys = namefutures[0].result().keys()
        return self._mergedict(datadicts, keys)

    def _obtain_name(self, name):
        ''' obtain_minibatch の個々のデータを取得する '''
        datadict = None
        # まず、ディスクキャッシュを探す
        if self.use_diskcache:
            path = os.path.join(self.tempdir.name, name + '.pkl')
            try:
                if os.path.exists(path): # ディスクキャッシュにヒット
                    with open(path, mode='rb') as fp:
                        datadict = pickle.load(fp)
                    
            except:
                datadict = None
        if datadict is None: # キャッシュにヒットしなかった場合
            reader = self.namereaderdict[name]
            obtained = False
            trycount = 0
            while (not obtained) and (trycount < 5):
                try:
                    trycount += 1
                    data = reader.getbyname(name)
                    obtained = True
                    if trycount >= 2:
                        print(trycount, '回で読み込み成功しました')
                except:
                    print('データ読み込みに失敗したため、リトライします trycount=', trycount)
            datadict = self.labelfunc(name, data)
            if self.use_diskcache: # ディスクキャッシュに保存する
                future = self.pexecutor.submit(self._save_pickle, datadict, path)
        return datadict

    @staticmethod
    def _save_pickle(datadict, path):
        try:
            with open(path, mode='wb') as fp:
                pickle.dump(datadict, fp, protocol=pickle.HIGHEST_PROTOCOL)
        except: # 失敗したら消す 所詮キャッシュなので消しても大丈夫
            os.remove(path)
        return

    @staticmethod
    def _mergedict(datadicts, keys):
        ''' ミニバッチデータの辞書に1件のデータ辞書をmergeし、新しいminibatch dictを返す内部用関数 '''
        import numpy as np
        
        datadictlist = tuple(datadicts)
        minibatchdict = {}
        for key in keys:
            mapped = map(lambda x: x[key], datadictlist)
            listed = tuple(mapped)
            minibatchdict[key] = np.concatenate(listed, axis=0)
        return minibatchdict

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
        
                                               
        

                                                




        
