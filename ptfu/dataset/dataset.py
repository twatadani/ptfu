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
        #if hasattr(srclist, '__getitem__'):
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

class NPYDataSet(DataSet):
    ''' NPYファイルで格納されているデータセット '''

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
        from concurrent.futures import ThreadPoolExecutor
        super(NPYDataSet, self).__init__(srclist, labellist, labelstyle, **options)

        self.readers = []
        self.namelist = [] # データネームのリスト 実際はreaderごとにリストにするので二重リスト

        assert 'labelfunc' in options
        self.labelfunc = options['labelfunc']
        assert 'storetype' in options
        for srcfile in self.srclist:
            reader = SrcReader(DataType.NPY, options['storetype'], srcfile)
            self.readers.append(reader)
            self.namelist.append(reader.namelist())

        self.namelist_flat = []
        for sublist in self.namelist:
            self.namelist_flat.extend(sublist)

        # disk cacheの設定
        self.use_diskcache = False
        self.tempdir = None
        if 'use_diskcache' in options:
            self.use_diskcache = True
            self._create_tempdir()

        self.executor = ThreadPoolExecutor(max_workers=8)
        return

    def __del__(self):
        ''' デストラクタ '''
        self._cleanup_tempdir()
        return

    def _cleanup_tempdir(self):
        ''' テンポラリディレクトリをクリーンアップする '''
        if self.tempdir is not None:
            self.tempdir.cleanup()
            self.tempdir = None
        return

    def _create_tempdir(self):
        ''' テンポラリディレクトリを作成する '''
        from tempfile import TemporaryDirectory
        if self.tempdir is None:
            self.tempdir = TemporaryDirectory()

    def obtain_minibatch(self, minibatchsize):
        ''' minibatchsizeで指定されたサイズのミニバッチを取得する '''
        import random
        from concurrent.futures import wait

        # 返り値となるdictを準備
        minibatch_dict = None

        # ミニバッチ対象となる名前リストを選択する
        minibatch_namelist = random.sample(self.namelist_flat, minibatchsize)

        # futures
        namefutures = []

        # データを取得する
        for name in minibatch_namelist:
            namefutures.append(
                self.executor.submit(self._obtain_name, name))

            # datadict = None
            # # まず、ディスクキャッシュを探す
            # if self.use_diskcache:
            #     path = os.path.join(self.tempdir.name, name + '.pkl')
            #     try:
            #         if os.path.exists(path): # ディスクキャッシュにヒット
            #             datadict = pickle.load(path)
            #             print('disk cache hit!')
            #     except:
            #         datadict = None
                
            # if datadict is None: # キャッシュにヒットしなかった場合
            #     for i in range(len(self.readers)):
            #         if name in self.namelist[i]:
            #             obtained = False
            #             trycount = 0
            #             while (not obtained) and (trycount < 5):
            #                 try:
            #                     trycount += 1
            #                     data = self.readers[i].getbyname(name)
            #                     obtained = True
            #                     if trycount >= 2:
            #                         print(trycount, '回で読み込み成功しました')
            #                 except:
            #                     print('データ読み込みに失敗したため、リトライします trycount=', trycount)
            #                 datadict = self.labelfunc(name, data)
            #                 if self.use_diskcache: # ディスクキャッシュに保存する
            #                     future = self.executor.submit(self._save_pickle, datadict, path)
            # minibatch_dict = self._mergedict(minibatch_dict, datadict)

        wait(namefutures)
        datadicts = map(lambda x: x.result(), namefutures)
        keys = namefutures[0].result().keys()
        # print('5 shape:', list(datadicts)[5]['fourier'].shape) # OK
        return self._mergedict(datadicts, keys)

    def _obtain_name(self, name):
        ''' obtain_minibatch の個々のデータを取得する '''
        datadict = None
        # まず、ディスクキャッシュを探す
        if self.use_diskcache:
            path = os.path.join(self.tempdir.name, name + '.pkl')
            try:
                #print('search disk cache')
                if os.path.exists(path): # ディスクキャッシュにヒット
                    #print('found disk cache')
                    with open(path, mode='rb') as fp:
                        datadict = pickle.load(fp)
                        #print('disk cache hit!')
                #else:
                    #print('not found')
                    
            except:
                #print('exception while loading disk cache')
                #import traceback
                #traceback.print_exc()
                datadict = None
                
        if datadict is None: # キャッシュにヒットしなかった場合
            for i in range(len(self.readers)):
                if name in self.namelist[i]:
                    obtained = False
                    trycount = 0
                    while (not obtained) and (trycount < 5):
                        try:
                            trycount += 1
                            data = self.readers[i].getbyname(name)
                            obtained = True
                            if trycount >= 2:
                                print(trycount, '回で読み込み成功しました')
                        except:
                            print('データ読み込みに失敗したため、リトライします trycount=', trycount)
                        datadict = self.labelfunc(name, data)
                        if self.use_diskcache: # ディスクキャッシュに保存する
                            future = self.executor.submit(self._save_pickle, datadict, path)
        #print('datadict[fourier].shape:', datadict['fourier'].shape) # OK
        return datadict

    def _save_pickle(self, datadict, path):
        try:
            with open(path, mode='wb') as fp:
                pickle.dump(datadict, fp, protocol=pickle.HIGHEST_PROTOCOL)
                #print('pickle save success!', path)
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
            #print('key = ', key)
            mapped = map(lambda x: x[key], datadictlist)
            listed = tuple(mapped)
            #print(listed)
            #print('len(listed)=', len(listed))
            minibatchdict[key] = np.concatenate(listed, axis=0)
        return minibatchdict

        # if minibatch_dict is None:
        #     return datadict
        # else:
        #     for key in datadict.keys():
        #         value = np.concatenate([minibatch_dict[key], datadict[key]], axis=0)
        #         minibatch_dict[key] = value
        #     return minibatch_dict

    #def _extend_list(self):
    #    ''' minibatchリストが短くなったときに延長する内部用関数 '''
    #    import random
    #    newnamelist = []
    #    for sublist in self.namelist:
    #        newnamelist.extend(sublist)
    #    random.shuffle(newnamelist)
    #    self.list_for_minibatch.extend(newnamelist)
    #    return

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
        
                                               
        

                                                




        
