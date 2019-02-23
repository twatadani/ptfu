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

    #@staticmethod
    def to_minibatchdict(self, minibatchdata, labellist, **options):
        ''' datasetから取り出した形式のリストで与えられるminibatchdataをdictの形に変換する '''
        raise NotImplementedError

    #@staticmethod
    def to_minibatchdict_byfunc(self, minibatcdata, labellist, **options):
        ''' to_minibatcdictのLABEL_BY_FUNC版 
        optionsにlabelfuncを与える必要がある。labelfuncは1件のデータに対して
        minibatchdict = labelfunc(minibatchdata, labellist)となるような関数
        '''
        labelfunc = options['labelfunc']
        return labelfunc(minibatcdata, labellist)


LabelStyle.LABEL_BY_FUNC.to_minibatchdict = LabelStyle.to_minibatchdict_byfunc




#class DataSetInfo:
    #''' DataSetをマルチプロセスで使用するためのpicklableオブジェクト '''

    #def __init__(self, arg_dictionary):
        #''' arg_dictionaryはDataSetオブジェクトを構築するための引数の辞書
        #ただし、'class': クラスオブジェクトのフィールドを持つ '''
        #self.dic = arg_dictionary
        #return
    
    #def constructDataSet(self):
        #''' このDataSetInfoからDataSetオブジェクトを構築する '''
        #cls = self.dic['class']
        #dataset = cls(self.dic['srclist'], self.dic['labellist'], self.dic['labelstyle'],
                      #**self.dic['options'])
        #return dataset
    

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
        srclist: データセット格納ファイルのコレクション 単独の場合は単にファイル名でもよい
        labellist: データラベルのリスト。例えば分類タスクで車、信号、家を分類する場合には['car', 'signal', 'house']を与える
        labelstyle: LabelStyle enumで選択されるラベルの指定方法
        options: 
        labelstyleで使用するオプションは個々で指定する。ほか、
        storetype: 明示的にソースアーカイブのStoreTypeを指定する場合に使用        
         '''

        import ptfu.functions as f
        
        # self.srclistはsetとして保有する
        if isinstance(srclist, list) or isinstance(srclist, tuple):
            self.srclist = set(srclist)
        elif isinstance(srclist, set):
            self.srclist = srclist
        else:
            self.srclist = set()
            self.srclist.add(srclist)

        # storetypeは明示的に指定されなければ自動検出する
        if 'storetype' in options:
            self.srcstoretype = options['storetype']
        else:
            self.srcstoretype = f.autodetect_storetype(self.srclist)
            assert self.srcstoretype is not None, self.srclist

        self.labellist = labellist
        self.labelstyle = labelstyle
        
        # データ件数のキャッシュ
        self.datanumber_cache = None

        # ランダムミニバッチキュー関連のインスタンス変数
        self.randomq = None
        self.randomq_minibatchsize = None
        self.randomq_future = None

        # シリアルミニバッチキュー関連のインスタンス変数
        self.serialq = None
        self.serialq_future = None        

        # todo オプションの処理
        self.options = options
        return

    def datanumber(self):
        ''' このデータセットのデータ件数を取得する '''
        if self.datanumber_cache is not None:
            return self.datanumber_cache
        else:
            self.datanumber_cache = 0
            for src in self.srclist:
                areader = self.srcstoretype.reader(src, use_cache=False)
                self.datanumber_cache += areader.datanumber()
            return self.datanumber_cache

    def start_random_minibatch_queue(self, minibatchsize):
        ''' バックグラウンドでランダムにミニバッチを作成するプロセスを立ち上げる '''
        from .dataqueue import DataQueue
        from ..kernel import kernel
        if self.randomq is not None:
            self.stop_random_minibatch_queue()

        executor = kernel.pexecutor
        self.randomq = DataQueue()
        self.randomq_minibatchsize = minibatchsize
        self.randomq_future = executor.submit(self.__class__._randomq_worker, self.randomq, minibatchsize,
                                            self.srcstoretype, self.srclist,
                                            self.labellist, self.labelstyle, self.options)
        return self.randomq

    def stop_random_minibatch_queue(self):
        ''' ミニバッチキューイングプロセスを停止する '''
        if self.randomq is not None:
            self.randomq = None
            self.randomq_minibatchsize = None
            self.randomq_future.cancel()
            self.randomq_future = None
        return

    def obtain_random_minibatch(self, minibatchsize):
        ''' ランダムに取得されるミニバッチを1回分取得する '''
        if (self.randomq is not None) and (self.randomq_minibatchsize == minibatchsize):
            #print('キューを使用してミニバッチを取り出します。')
            minibatch = self.randomq.pop()
            return minibatch
        else:
            #print('warning: キューを使わずにミニバッチを取り出します。')
            areaders = []
            for src in self.srclist:
                areader = self.srcstoretype.reader(src, use_cache=False)
                areaders.append(areader)
            minibatchdata = self.__class__._random_minibatchdata(areaders, minibatchsize)
            minibatchdict = self.labelstyle.to_minibatchdict(self.labelstyle, minibatchdata, self.labellist, **self.options)
            return minibatchdict

    def obtain_serial_minibatch_queue(self, minibatchsize):
        ''' 全データを順に取得するミニバッチキューを作成する '''
        from .dataqueue import DataQueue
        from ptfu import kernel
        # minibatch数を計算する
        ndata = self.datanumber()
        nbatch = ndata // minibatchsize
        residue = ndata % minibatchsize
        if residue != 0:
            nbatch += 1
        self.serialq = DataQueue(nbatch)
        executor = kernel.pexecutor
        self.serialq_future = executor.submit(self.__class__._serialq_worker, self.serialq, minibatchsize,
                                            self.srcstoretype, self.srclist, self.labellist, self.labelstyle, self.options)
        return self.serialq

    @staticmethod
    def _serialq_worker(queue, minibatchsize, storetype, srclist, labellist, labelstyle, options):
        ''' 全データを順に取得し、キューに入れる作業を行うワーカー関数 '''
        areaders = []
        namelists = []
        for src in srclist:
            areader = storetype.reader(src, use_cache=False)
            areaders.append(areader)
            namelist = areader.namelist()
            namelists.append(namelist)
        batch_counter = 0 # バッチ番号
        current_listindex = 0 # 現在使用しているnamelistの番号
        last_index = 0 # namelist内で最後に取得したindex
        nbatch = queue.datanumber()
        while batch_counter < nbatch:
            current_list = namelists[current_listindex]
            batch_start = last_index
            batch_end = last_index + minibatchsize
            if batch_end <= len(current_list): # ミニバッチがはみでず1リストで完結する場合
                use_names = current_list[batch_start:batch_end]
                reader = areaders[current_listindex]
                minibatchdata = reader.getbylist(use_names)
                # indexを加算
                last_index += minibatchsize
                #current_listindex #はいじらない
                batch_counter += 1
            else: # ミニバッチがはみ出て2リストにまたがる場合
                batch_end = len(current_list)
                # とりあえず全部取得する
                if batch_end == batch_start + 1: # ちょうどバッチが終わっていた場合は空リスト
                    minibatchdata = []
                else:
                    reader = areaders[current_listindex]
                    use_names = current_list[batch_start:batch_end]
                    minibatchdata = reader.getbylist(use_names)
                    
                    # 次のリストに移行する
                    current_listindex += 1
                    if current_listindex < len(namelists): # 次のリストがまだある場合のみ
                        current_list = namelists[current_listindex]
                        batch_start = 0
                        batch_end = minibatchsize - len(use_names) + 1
                        reader = areaders[current_listindex]
                        use_names = current_list[batch_start:batch_end]
                        minibatchdata.extend(reader.getbylist(use_names))

                    # indexを加算
                    last_index += batch_end
                    batch_counter += 1
            minibatchdict = labelstyle.to_minibabtchdict(minibatchdata, labellist, options)
            queue.push(minibatchdict)
        return

    @staticmethod
    def _randomq_worker(queue, minibatchsize, storetype, srclist, labellist, labelstyle, options, qsize_threashold = 50):
        ''' ランダムにミニバッチを取得し、キューに入れる作業を繰り返すワーカー関数 '''
        from time import sleep
        print('_randomq_workerが起動されました')
        try:
            areaders = []
            for src in srclist:
                areader = storetype.reader(src, use_cache=True)
                areaders.append(areader)
        
        # ミニバッチを取り出すループ
        
            while True:
                #print('loop 1')
                minibatchdata = DataSet._random_minibatchdata(areaders, minibatchsize)
                #print('loop 2')
                minibatchdict = labelstyle.to_minibatchdict(labelstyle, minibatchdata, labellist, **options)
                #print('loop 3')
                queue.push(minibatchdict)
                #print ('loop 4')
                if queue.qsize() > qsize_threashold:
                    sleep(2)
        except:
            import traceback
            traceback.print_exc()

    @staticmethod
    def _random_minibatchdata(readers, minibatchsize):
        ''' readersまで設定した状態でランダムなminibatchを1回分取得する '''
        import ptfu.functions as f
        import random
        from .datatype import DataType
        readers_shuffled = random.sample(readers, len(readers))
        indexes = f.random_split_index(minibatchsize, len(readers))
        minibatchdata = []
        for i, reader in enumerate(readers_shuffled):
            #print('reader', reader)
            namelist = reader.namelist(DataType.NPY)
            #print('i:', i, 'len(namelist):', len(namelist), 'indexes[i]', indexes[i])
            try:
                getlist = random.sample(namelist, indexes[i])
                #print('len(getlist)=', len(getlist))
                data = reader.getbylist(getlist, DataType.NPY)
                #print('len(data)=', len(data))
            except:
                import traceback
                traceback.print_exc()
            
            
            minibatchdata.extend(data)
        #print('returning _random_minibatchdata')
        return minibatchdata


    #def __init__(self, srclist, labellist, labelstyle, **options):
        #''' DataSetのイニシャライザ
        #srclist: データセット格納ファイルのリスト
        #labellist: ['label1, 'label2', ...]のようなデータラベルのリスト
        #labelstyle: どのような方式でラベルデータを設定するか。LabelStyle enumから選択する。
        #options: labelstyleに依存するオプション '''

        #if isinstance(srclist, list) or isinstance(srclist, tuple):
            #self.srclist = srclist
        #elif srclist is not None:
            #self.srclist = [srclist]
        #else:
            #self.srclist = []

        #self.labellist = labellist
        #self.labelstyle = labelstyle
        #self.options = options
        #return

    #def obtain_minibatch(self, minibatchsize):
        #''' minibatchsizeで指定されたサイズのミニバッチを取得する。
        #ミニバッチデータは
        #{ label1: (minibatchsize, ....), label2: (minibatchsize, ....) }
        #の形式 '''
        #raise NotImplementedError

    #def datanumber(self):
        #''' このデータセットのデータ数を得る '''
        #raise NotImplementedError
        

    #def toDataSetInfo(self):
        #''' このオブジェクトをDataSetInfoに変換する '''
        #dic = {}
        #cls = self.__class__
        #dic['class'] = cls
        #dic['srclist'] = self.srclist
        #dic['labellist'] = self.labellist
        #dic['labelstyle'] = self.labelstyle
        #dic['options'] = self.options
        #return DataSetInfo(dic)
        



        

#class PILDataSet(DataSet):
    #''' PILでロードできるフォーマットで格納されたデータセット '''

    #def __init__(self, srclist, labellist, labelstyle, datatype, **options):
        #''' PILDataSetのイニシャライザ
        #srclist: データセット格納ファイルのリスト
        #labellist: ['label1, 'label2', ...]のようなデータラベルのリスト
        #labelstyle: どのような方式でラベルデータを設定するか。LabelStyle enumから選択する。
        #datatype: 具体的なデータタイプ。DataType enumから選択する
        #options内に
        #storetype: StoreTypeメンバ、
        #labelfunc: 1データ当たりのlabel切り分け辞書 = labelfunc(name, data)となるような
        #storetype, labelfuncが必須 '''
        #from . import SrcReader, DataType
        #super(PILDataSet, self).__init__(srclist, labellist, labelstyle, **options)

        #self.readers = []
        #self.datatype = datatype
        #for srcfile in self.srclist:
            #reader = SrcReader(datatype, options['storetype'], srcfile)
            #self.readers.append(reader)

        #assert 'labelfunc' in options
        #self.labelfunc = options['labelfunc']
        #return

    #def datanumber(self):
        #''' このデータセットのデータ数を得る '''
        #num = 0
        #for reader in self.readers:
            #num += reader.datanumber()
        #return num

    #def obtain_minibatch(self, minibatchsize):
        #''' minibatchsizeで指定されたサイズのミニバッチを取得する。
        #ミニバッチデータは
        #{ label1: (minibatchsize, ....), label2: (minibatchsize, ....) }
        #の形式 '''
        #raise NotImplementedError

    #def obtain_wholedata(self):
        #''' すべてのデータを取得する。返り値は
        #{ label1: (minibatchsize, ...), label2: (minibatchsize, ...) }
        #の辞書形式 '''
        #datadict = None

        #for reader in self.readers:
            #iterator = reader.iterator()
            #for name, npy in iterator:
                #singledict = self.labelfunc(name, npy)
                #assert singledict is not None
                #if datadict is None:
                    #datadict = singledict
                #else:
                    #datadict = NPYDataSet._mergedict([datadict, singledict], datadict.keys())

        #return datadict

#class JPGDataSet(PILDataSet):
    #''' JPEG画像が格納されているデータセット '''

    #def __init__(self, srclist, labellist, labelstyle, **options):
        #''' JPGDataSetのイニシャライザ
        #srclist: データセット格納ファイルのリスト
        #labellist: ['label1, 'label2', ...]のようなデータラベルのリスト
        #labelstyle: どのような方式でラベルデータを設定するか。LabelStyle enumから選択する。
        #options内に
        #storetype: StoreTypeメンバ、
        #labelfunc: 1データ当たりのlabel切り分け辞書 = labelfunc(name, data)となるような
        #storetype, labelfuncが必須 '''
        #from . import DataType
        #super(JPGDataSet, self).__init__(srclist, labellist, labelstyle, DataType.JPG, **options)
        #return

#class PNGDataSet(PILDataSet):
    #''' PNG画像が格納されているデータセット '''

    #def __init__(self, srclist, labellist, labelstyle, **options):
        #''' PNGDataSetのイニシャライザ
        #srclist: データセット格納ファイルのリスト
        #labellist: ['label1, 'label2', ...]のようなデータラベルのリスト
        #labelstyle: どのような方式でラベルデータを設定するか。LabelStyle enumから選択する。
        #datatype: 具体的なデータタイプ。DataType enumから選択する
        #options内に
        #storetype: StoreTypeメンバ、
        #labelfunc: 1データ当たりのlabel切り分け辞書 = labelfunc(name, data)となるような
        #storetype, labelfuncが必須 '''
        #from . import DataType
        #super(PNGDataSet, self).__init__(srclist, labellist, labelstyle, DataType.PNG, **options)
        #return
