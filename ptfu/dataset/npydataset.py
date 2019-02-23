''' npydataset.py: NPYDataSetを記述する '''

from .dataset import DataSet

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
        from .datatype import DataType
        from .storetype import StoreType
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
        # disk cacheの設定
        self.use_diskcache = False
        self.tempdir = None
        if 'use_diskcache' in options:
            self.use_diskcache = True
            self._create_tempdir()

        return

    def __del__(self):
        ''' デストラクタ '''
        if hasattr(self, 'tempdir_owner') and self.tempdir_owner is True:
            self._cleanup_tempdir()
        if self.texecutor is not None:
            self.texecutor.shutdown()
        if self.pexecutor is not None:
            self.pexecutor.shutdown()
        return

    def datanumber(self):
        ''' このデータセットのデータ数を得る '''
        num = 0
        for reader in self.readers:
            num += reader.datanumber()
        return num

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
        from ..logger import get_default_logger

        logger = get_default_logger()

        if not self.srclist[0] in NPYDataSet.diskcachedict:
            if self.tempdir is None:
                logger.debug('NPYDataSetのディスクキャッシュを作成します')
                self.tempdir = TemporaryDirectory()
                NPYDataSet.diskcachedict[self.srclist[0]] = self.tempdir
                self.tempdir_owner = True
        else:
            logger.debug('NPYDataSetのディスクキャッシュを流用します')
            self.tempdir = NPYDataSet.diskcachedict[self.srclist[0]]
            self.tempdir_owner = False

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

        from ..logger import get_default_logger
        logger = get_default_logger()

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
                        logger.debug(str(trycount) + '回で読み込み成功しました')
                except:
                    logger.warning('データ読み込みに失敗したため、リトライします trycount = '+ str(trycount))
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
            listed = None
            for elementdict in datadictlist:
                element = elementdict[key]
                if isinstance(element, np.ndarray):
                    if listed is None:
                        listed = element
                    else:
                        listed = np.concatenate([listed, element], axis=0)
                else:
                    listed = NPYDataSet._flatextend(listed, element)
            minibatchdict[key] = listed

        return minibatchdict

    @staticmethod
    def _flatextend(elem1, elem2):
        ''' elem1, elem2を結合してフラットなリストにする '''
        baselist = []
        if elem1 is not None:
            if isinstance(elem1, list):
                baselist = elem1
            else:
                baselist = [ elem1 ]
                # この時点でbaselistは必ずlist
                
        if elem2 is not None:
            if isinstance(elem2, list):
                baselist.extend(elem2)
            else:
                baselist.append(elem2)
        return baselist