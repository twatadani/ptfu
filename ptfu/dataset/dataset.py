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

    def to_minibatchdict(self, minibatchdata, labellist, **options):
        ''' datasetから取り出した形式のリストで与えられるminibatchdataをdictの形に変換する '''
        raise NotImplementedError

    def to_minibatchdict_byfunc(self, minibatchdata, labellist, **options):
        ''' to_minibatcdictのLABEL_BY_FUNC版 
        optionsにlabelfuncを与える必要がある。labelfuncは1件のデータに対して
        minibatchdict = labelfunc(minibatchdata, labellist)となるような関数
        '''
        labelfunc = options['labelfunc']
        return labelfunc(minibatchdata, labellist)


LabelStyle.LABEL_BY_FUNC.to_minibatchdict = LabelStyle.to_minibatchdict_byfunc

class DataSet:
    ''' 1組のデータセットを表現するクラス 
    本ライブラリの枠組みでは、1件のデータは
    { label1: ..., label2: ..., label3: ..., }
    といった辞書データ形式で表現される。
    例えば、'img'のみからなるデータは { 'img': binarydata }、
    'img'と'label'からなるデータは{ 'img': binarydata, 'label': 'somelabel' }
    という形式になる。
    '''

    def __init__(self, srclist, labellist, labelstyle, datatype, **options):
        ''' DataSetのイニシャライザ
        srclist: データセット格納ファイルのコレクション 単独の場合は単にファイル名でもよい
        labellist: データラベルのリスト。例えば分類タスクで車、信号、家を分類する場合には['car', 'signal', 'house']を与える
        labelstyle: LabelStyle enumで選択されるラベルの指定方法
        datatype: 格納されている個々のデータのデータタイプ DataType enumで指定する
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

        self.datatype = datatype

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
        self.randomq_futures = None

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
                self.datanumber_cache += areader.datanumber(self.datatype)
            return self.datanumber_cache

    def start_random_minibatch_queue(self, minibatchsize, nworker = 3):
        ''' バックグラウンドでランダムにミニバッチを作成するプロセスを立ち上げる '''
        from .dataqueue import DataQueue
        from ..kernel import kernel
        if self.randomq is not None:
            self.stop_random_minibatch_queue()

        executor = kernel.pexecutor
        self.randomq = DataQueue()
        self.randomq_minibatchsize = minibatchsize
        self.randomq_futures = []
        for _ in range(nworker):
            self.randomq_futures.append(executor.submit(self.__class__._randomq_worker, self.randomq, minibatchsize,
                                                        self.srcstoretype, self.srclist,
                                                        self.labellist, self.labelstyle, self.datatype, self.options))
        return self.randomq

    def stop_random_minibatch_queue(self):
        ''' ミニバッチキューイングプロセスを停止する '''
        from time import sleep
        if self.randomq is not None:
            self.randomq = None
            self.randomq_minibatchsize = None
            #result = False
            for f in self.randomq_futures:
                f.cancel()
        return

    def obtain_random_minibatch(self, minibatchsize):
        ''' ランダムに取得されるミニバッチを1回分取得する '''
        if (self.randomq is not None) and (self.randomq_minibatchsize == minibatchsize):
            minibatch = self.randomq.pop()
            return minibatch
        else:
            areaders = []
            for src in self.srclist:
                areader = self.srcstoretype.reader(src, use_cache=False)
                areaders.append(areader)
            minibatchdata = self.__class__._random_minibatchdata(areaders, minibatchsize, self.datatype)
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
                                            self.srcstoretype, self.srclist, self.labellist, self.labelstyle, self.datatype, self.options)
        return self.serialq

    @staticmethod
    def _serialq_worker(queue, minibatchsize, storetype, srclist, labellist, labelstyle, datatype, options):
        ''' 全データを順に取得し、キューに入れる作業を行うワーカー関数 '''
        from ptfu import get_default_logger
        logger = get_default_logger()
        try:
            areaders = []
            namelists = []
            for src in srclist:
                areader = storetype.reader(src, use_cache=False)
                areaders.append(areader)
                namelist = areader.namelist(datatype)
                namelists.append(namelist)
            batch_counter = 0 # バッチ番号
            current_listindex = 0 # 現在使用しているnamelistの番号
            last_index = 0 # namelist内で最後に取得したindex
            nbatch = queue.datanumber()
            while batch_counter < nbatch:
                current_list = list(namelists[current_listindex])
                batch_start = last_index
                batch_end = last_index + minibatchsize
                if batch_end <= len(current_list): # ミニバッチがはみでず1リストで完結する場合
                    use_names = current_list[batch_start:batch_end]
                    reader = areaders[current_listindex]
                    minibatchdata = reader.getbylist(use_names, datatype)
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
                        minibatchdata = reader.getbylist(use_names, datatype)
                    
                        # 次のリストに移行する
                        current_listindex += 1
                        #logger.debug('次のリストに移行します。')
                        #logger.debug('current_listindex: ' + str(current_listindex))
                        #logger.debug('batch_counter: ' + str(batch_counter))
                        #logger.debug('len(namelists): ' + str(len(namelists)))
                        if current_listindex < len(namelists): # 次のリストがまだある場合のみ
                            current_list = list(namelists[current_listindex])
                            batch_start = 0
                            batch_end = minibatchsize - len(use_names) + 1
                            last_index = 0
                            reader = areaders[current_listindex]
                            use_names = current_list[batch_start:batch_end]
                            minibatchdata.extend(reader.getbylist(use_names, datatype))

                        # indexを加算
                        last_index += batch_end
                        batch_counter += 1
                if minibatchdata is not None and len(minibatchdata) > 0:
                    minibatchdict = labelstyle.to_minibatchdict(labelstyle, minibatchdata, labellist, **options)
                    queue.push(minibatchdict)
                else:
                    if minibatchdata is None:
                        logger.warning('in serialq worker, minibatchdata is None')
                    else:
                        logger.warning('in serialq worker, minibatchdata is empty list')
            return
        except:
            import traceback
            traceback.print_exc()
            logger.debug('len(namelists) = ' + str(len(namelists)))
            logger.debug('current listindex=' + str(current_listindex))
            return

    @staticmethod
    def _randomq_worker(queue, minibatchsize, storetype, srclist, labellist, labelstyle, datatype, options, qsize_threashold = 50):
        ''' ランダムにミニバッチを取得し、キューに入れる作業を繰り返すワーカー関数 '''
        from time import sleep
        #print('_randomq_workerが起動されました')
        try:
            areaders = []
            for src in srclist:
                areader = storetype.reader(src, use_cache=True)
                areaders.append(areader)
        
        # ミニバッチを取り出すループ
        
            while True:
                if queue.qsize() <= qsize_threashold:
                    minibatchdata = DataSet._random_minibatchdata(areaders, minibatchsize, datatype)
                    minibatchdict = labelstyle.to_minibatchdict(labelstyle, minibatchdata, labellist, **options)
                    queue.push(minibatchdict)
                else:
                    sleep(0.215)
        except:
            import traceback
            traceback.print_exc()

    @staticmethod
    def _random_minibatchdata(readers, minibatchsize, datatype):
        ''' readersまで設定した状態でランダムなminibatchを1回分取得する '''
        import ptfu.functions as f
        import random
        from .datatype import DataType
        readers_shuffled = random.sample(readers, len(readers))
        indexes = f.random_split_index(minibatchsize, len(readers))
        minibatchdata = []
        for i, reader in enumerate(readers_shuffled):
            namelist = reader.namelist(datatype)
            try:
                getlist = random.sample(namelist, indexes[i])
                data = reader.getbylist(getlist, datatype)
            except:
                import traceback
                traceback.print_exc()
            
            
            minibatchdata.extend(data)
        return minibatchdata
