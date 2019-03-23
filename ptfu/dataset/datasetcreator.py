''' datasetcreatorモジュール: データセットの作成を行う '''

from enum import Enum, auto

class SplitManner(Enum):

    BY_N_OF_GROUPS = auto() # グループ数を規定する
    BY_N_IN_SINGLEGROUP = auto() # 1グループあたりのデータ数で規定する

class DatasetCreator:
    ''' ソースとデスティネーションの情報を与えてデータセット作成を行うクラス '''

    def __init__(self, srcdatatype, srcstoretype, src,
                dststoretype, dst, datasetname,
                ndstsplit = 1, dstsplitmanner=SplitManner.BY_N_OF_GROUPS, 
                 **options):
        ''' DataSetCreatorの初期化
        srcdatatype: ソースデータのフォーマット。DataType enumのいずれかを指定する
        srcstoretype: ソースデータのアーカイブ形式。StoreType enumのいずれかを指定する
        src: ソースデータの位置。アーカイブファイルのパス、複数の場合はそれらのリスト、タプル、セットで指定する
        dststoretype: 作成するデータセットのアーカイブ形式。StoreType enumのいずれかを指定する
        dst: 作成するデータセットの格納ディレクトリ
        datasetname: 作成するデータセットの名前 
        ndstsplit: 作成するデータセットを分割する場合の数 デフォルトは1
        dstsplitmanner: 分割方法。SplitManner enumで指定する。グループ数を指定する場合はBY_NGROUPS, 1グループあたりのデータ数を指定する場合はBY_NUMBER_IN_GROUP
        optionsの解説
        nesteddict: srcstoretypeがNESTEDの場合に使われる辞書。'outerstoretype', 'innerstoretype', 'innernames'が必要。
        featurefunc: dststoretypeがTFRECORDの場合に使われる。TFRecordWriterのdefault_feature_funcと同様、name, ndarrayを引数に取り、Featureを作成する関数を指定する。指定しない場合はdefault_feature_funcが使用される。
        '''
        import os.path
        from .storetype import StoreType

        self.srcdatatype = srcdatatype
        self.srcstoretype = srcstoretype
        self.src = src
        self.dstpath = dst
        self.dststoretype = dststoretype
        self.datasetname = datasetname
        self.ndstsplit = ndstsplit
        self.dstsplitmanner = dstsplitmanner

        # ソースを単数複数判定してareadersを作成する
        self.areaders = set()
        if src is None:
            raise ValueError('DatasetCreator.__init__(): src should not be None')
        elif isinstance(src, list) or isinstance(src, tuple) or isinstance(src, set): # 複数の場合
            srcset = set(src)
            for single_src in srcset:
                src_expanded = os.path.expandvars(os.path.expanduser(single_src))
                if srcstoretype == StoreType.NESTED:
                    raise ValueError('Nested reader from multiple source file is not supported')
                else:
                    areader = srcstoretype.reader(src_expanded, use_cache=False)
                    self.areaders.add(areader)
        else: # 単数の場合
            src_expanded = os.path.expandvars(os.path.expanduser(src))
            if srcstoretype == StoreType.NESTED:
                if not 'nesteddict' in options:
                    raise ValueError('Nested reader requires nesteddict parameter in options')
                nesteddict = options['nesteddict']
                for innername in nesteddict['innernames']:
                    areader = srcstoretype.reader(nesteddict['outerstoretype'],
                                                src_expanded,
                                                nesteddict['innerstoretype'],
                                                innername,
                                                use_cache=False)
                    self.areaders.add(areader)
            else:
                areader = srcstoretype.reader(src, use_cache=False)
                self.areaders.add(areader)

        # Split mannerに合わせてawritersを作成する
        self.awriters = set()
        self.ngroups = 1
        basename = os.path.join(dst, datasetname)
        if dstsplitmanner == SplitManner.BY_N_OF_GROUPS:
            self.ngroups = ndstsplit
        elif dstsplitmanner == SplitManner.BY_N_IN_SINGLEGROUP:
            ndata = self.datanumber()
            self.ngroups = self._count_split_groups(ndata, ndstsplit)
        else:
            raise ValueError('DatasetCreator.__init__(): invalid dstsplitmanner; ', dstsplitmanner)

        for i in range(self.ngroups):
            inter = '-'+ str(i) if self.ngroups > 1 else ''
            dstpath = basename + inter + '.' + dststoretype.getext()
            if (dststoretype == StoreType.TFRECORD) and ('featurefunc' in options):
                awriter = dststoretype.writer(dstpath, options['featurefunc'])
            else:
                awriter = dststoretype.writer(dstpath)
            self.awriters.add(awriter)
        return
        
    def datanumber(self):
        ''' データソースを元に全データ件数をカウントする '''
        count = 0
        for reader in self.areaders:
            count += reader.datanumber(self.srcdatatype, allow_cached=True)
        return count

    def srcstr(self):
        ''' ソースパスの文字列を返す。複数の場合はカンマで区切る '''
        srcstr = None
        if isinstance(self.src, list) or isinstance(self.src, tuple) or isinstance(self.src, set):
            for single_src in self.src:
                if srcstr is None:
                    srcstr = single_src
                else:
                    srcstr += ', ' + single_src
        else:
            srcstr = self.src
        return srcstr

    @staticmethod
    def _count_split_groups(ndata, ndstsplit):
        ''' ndata件のデータをndstsplit件ずつグループ分けしたときのグループ数をカウントする '''
        div = ndata // ndstsplit
        mod = ndata % ndstsplit
        # 割り切れるときは商自体がグループ数、割り切れないときは商より1多い
        return div if mod == 0 else div + 1

    def create(self, filter_func=None, logger=None):
        ''' 設定をfixしてデータセット作成を行う
        filter_func: 読み込まれたdatadictから書き込むdatadictへのフィルター関数 Noneの場合はそのまま書き込む
        logger: ロギングを行う。Noneの場合はptfu default loggerを使用する
        '''
        from concurrent.futures import wait
        from time import sleep
        from ..logger import get_default_logger
        from .dataqueue import DataQueue
        import ptfu.functions as f
    
        if logger is None:
            logger = get_default_logger()

        # データセット件数を調査
        ndata = self.datanumber()

        # 設定を表示する
        logger.log('データセット作成を開始します。')
        logger.log('元データのパス: ' + str(self.srcstr()))
        logger.log('元データのアーカイブ形式: ' + self.srcstoretype.name)
        logger.log('元データのデータ形式: ' + self.srcdatatype.name)
        logger.log('作成するデータセットのパス: ' + self.dstpath)
        logger.log('作成するデータセットのアーカイブ形式: ' +  self.dststoretype.name)
        logger.log('作成するデータセットファイルの個数: ' + str(self.ngroups))
        logger.log('データセット名: ' + self.datasetname)
        logger.log('検出されたデータ件数: ' + str(ndata))

        logger.log('##############################')
        
        # データセット作成開始

        # phase 1: ソースからデータを読み込み、キューに入れる
        logger.log('DatasetCreator phase 1を開始します。')
        q = DataQueue(ndata)
        #max_workers = max(f.cpu_count() // len(self.areaders), 1)
        for reader in self.areaders:
            reader.getallbyqueue(self.srcdatatype, q, max_workers=1)
        logger.log('DatasetCreator phase 1を終了します。')
        logger.log('phase 1終了時点: q.datanumber() = ' + str(q.datanumber()))
        
        # phase 2: filterfuncを適用し、新しいキューに入れる
        logger.log('DatasetCreator phase 2を開始します。')
        import ptfu.kernel
        executor = ptfu.kernel.pexecutor

        filteredq = None
        if filter_func is not None:
            filteredq = DataQueue(ndata)
            ncpu = max(f.cpu_count() // 2, 1)
            for _ in range(ncpu):
                future = executor.submit(DatasetCreator._dofilter, filter_func, q, filteredq)
        else:
            filteredq = q
        logger.log('DatasetCreator phase 2を終了します。')
        logger.log('phase 2終了時点: q.datanumber() = ' + str(q.datanumber()))
        logger.log('phase 2終了時点: q.poppednumber() = ' + str(q.poppednumber()))
        logger.log('phase 2終了時点: filteredq.datanumber() = ' + str(filteredq.datanumber()))  
        logger.log('phase 2終了時点: filteredq.poppednumber() = ' + str(filteredq.poppednumber()))

        # phase 3 キューを各アーカイブ用に分割する
        logger.log('DatasetCreator phase 3を開始します。')
        writeqs = self._splitq(filteredq, self.ndstsplit, self.dstsplitmanner)
        logger.log('DatasetCreator phase 3を終了します。')
        logger.log('phase 3終了時点: q.datanumber() = ' + str(q.datanumber()))
        logger.log('phase 3終了時点: q.poppednumber() = ' + str(q.poppednumber()))
        logger.log('phase 3終了時点: filteredq.datanumber() = ' + str(filteredq.datanumber()))  
        logger.log('phase 3終了時点: filteredq.poppednumber() = ' + str(filteredq.poppednumber()))
        logger.log('phase 3終了時点 writeqsのdatanumberとpoppednumber')
        for wq in writeqs:
            logger.log('  datanumber: ' + str(wq.datanumber()) + ', poppednumber: ' + str(wq.poppednumber()))

        # phase 4 キューに入ったデータをアーカイブに書き込む
        logger.log('DatasetCreator phase 4を開始します。')
        futures = []
        for i, writer in enumerate(self.awriters):
            logger.log('ワーカーをsubmitします。i=' + str(i))
            future = executor.submit(writer.writebyq, writeqs[i])
            futures.append(future)
        logger.log('DatasetCreator phase 4を終了します。')

        # phase 5 モニタリング
        logger.log('DatasetCreator phase 5を開始します。')
        while not self._allcompleted(futures):
            donecount = 0
            pushedcount = 0
            for wq in writeqs:
                donecount += wq.poppednumber()
                pushedcount += wq.pushednumber()
            logger.log(str(donecount) + '件の書き込みが終了しました。q.pushed: ' + str(q.pushednumber()) +
                ', q.popped: ' + str(q.poppednumber()) + ', filteredq.pushed: ' + str(filteredq.pushednumber()) +
                ', filteredq.popped: ' + str(filteredq.poppednumber()) +
                ', 最終キューにpushされた総数: ' + str(pushedcount))
            sleep(1)
        wait(futures)
        #for future in futures:
            #print(future.done())
        logger.log('データセット作成が終了しました。')
        return

    @staticmethod
    def _dofilter(filter_func, srcq, dstq):
        ''' srcqのデータを1件ずつ取り出し、filter_funcを適用してdstqに入れる '''
        try:
            while srcq.hasnext():
                data = srcq.pop() # (name, datadict)またはdatadict
                if data is not None:
                    if isinstance(data, dict):
                        filtered = filter_func(data)
                        dstq.push(filtered)
                    else:
                        filtered = filter_func(data[1])
                        tup = (data[0], filtered)
                        dstq.push(tup)
        except:
            import traceback
            traceback.print_exc()
            pass
        return

    @staticmethod
    def _allcompleted(futures):
        ''' すべてのfutureが終了しているかどうかを知る '''
        for future in futures:
            if not future.done():
                return False
        return True

    @staticmethod
    def _splitq(queue, n, split_manner):
        ''' queueをsplit_mannerに沿って分割する。nはsplit_mannerによって振る舞いを変える '''
        from .dataqueue import DataQueue
        import ptfu.kernel
        ndata = queue.datanumber()
        if split_manner == SplitManner.BY_N_IN_SINGLEGROUP:
            n_per_group = n
            ngroups = ndata // n_per_group
            if ndata % n_per_group > 0:
                ngroups += 1
        elif split_manner == SplitManner.BY_N_OF_GROUPS:
            ngroups = n
            n_per_group = ndata // ngroups
        else:
            raise ValueError('_splitq: split_manner invalid; ', split_manner)

        if ngroups == 1:
            # 分割しないときはそのままキューを返す
            return [ queue ]

        # キューに入れる予定数をリスト形式で作成する
        nlist = []
        total = 0
        for _ in range(ngroups):
            newn = n_per_group
            # totalがデータ数を超えないようにする安全策
            if total + newn > ndata:
                newn = ndata - total
            nlist.append(newn)
            total += newn

        # データを拾いこぼさないようにする安全策
        if total < ndata:
            residue = ndata - total
            for i in range(residue):
                nlist[i] += 1
        
        executor = ptfu.kernel.pexecutor
        futures = []
        newqs = []
        for n in nlist:
            newq = DataQueue(n)
            newqs.append(newq)
            futures.append(executor.submit(DatasetCreator._q_pipe, queue, newq, n))
        return newqs

    @staticmethod
    def _q_pipe(srcq, dstq, n):
        ''' srcqをソースとして、dstqにn件のデータを読み込むワーカー関数 '''
        try:
            #print('_q_pipeが呼び出されました')
            count = 0
            while count < n:
                data = srcq.pop()
                if data is not None:
                    dstq.push(data)
                count += 1
            #print('_q_pipeを終了します。')
            return
        except:
            import traceback
            traceback.print_exc()

