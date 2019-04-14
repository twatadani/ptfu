''' archivereader.py: ZIP, TARなどの格納形式からのデータ読み出しを担当するArchiveReader基底クラスを記述 '''

from io import BytesIO

class ArchiveReader:
    ''' 格納形式それぞれからデータを読み出すインターフェースを規定する基底クラス
    継承クラスへの実装メソッドの指針:
    namelist: 格納しているメンバの名前リスト(ないしイテレータ)を返す
    _find_name(fp, name): getbynameのデフォルト実装から呼ばれる。staticmethodとして実装。アーカイブはすでにオープンされている前提
    TypeReaderのreadメソッドで読めるオブジェクト(BytesIOまたはファイルパス)を返す
    _open_src(srcpath): staticmethodとして実装。アーカイブファイルのオープン。
    _close_src(fp): staticmethodとして実装。アーカイブファイルのクローズ。fp.close()だけでよいならデフォルト実装のままでよい。
    rawmemberview: 特定のメンバーのraw viewを返す。staticmethodとして実装 (主にNestedArchiveReader用)
    diskcache_supported: ディスクキャッシュが使用可能かどうかをboolで返す
    prepare_diskcache: ディスクキャッシュを準備し、DiskCacheオブジェクトを返す
    '''

    def __init__(self, storetype, srcpath, use_cache=True):
        ''' イニシャライザ
        srcdatatype: データのフォーマット
        srcpath: データ格納元のパス
        '''
        from .storetype import StoreType
        import os.path

        # StoreTypeの設定
        assert isinstance(storetype, StoreType)
        self.storetype = storetype

        self.srcpath = srcpath
        if isinstance(srcpath, str):
            self.srcpath = os.path.expanduser(self.srcpath)

        self.use_cache = use_cache
        if use_cache == True:
            from .cachewriter import CacheWriter
            from .cachereader import CacheReader
            print('ArchiveReaderのメモリキャッシュを準備します。')
            self.mcwriter = CacheWriter()
            self.mcreader = CacheReader(self.mcwriter.dstpath)

            self.diskcache = None
            if self.diskcache_supported() is True:
                print('ディスクキャッシュを準備します。')
                self.diskcache = self.prepare_diskcache()
            else:
                print('diskcache_supported is false')
            if self.diskcache is not None:
                self.dcreader = CacheReader(self.diskcache)

        self.datanumber_cache = None
        self.namelist_cache = None
        return

    @staticmethod
    def rawmemberview(fp, membername):
        ''' membernameを与えてこのアーカイブ内のmemberに対するビューを取得する
        fpは_open_srcで得られるもの。
        ビューはBufferedReader, File-like objectなど。read, seekができるオブジェクトを返す '''
        raise NotImplementedError

    def datanumber(self, datatype, allow_cached=True):
        ''' 格納されているデータ件数のうち、datatypeに合致するものを返す '''
        if allow_cached and self.datanumber_cache is not None:
            return self.datanumber_cache
        else:
            nlist = self.namelist(datatype, allow_cached)
            if hasattr(nlist, '__len__') and callable(nlist.__len__):
                return len(nlist)
            else:
                count = 0
                for _ in nlist:
                    count += 1
                return count

    def diskcache_supported(self):
        ''' ディスクキャッシュをサポートしているかどうか デフォルトはFalse'''
        return False

    def prepare_diskcache(self):
        ''' ディスクキャッシュを準備し、DiskCacheオブジェクトを返す。サポートされていない場合はNone '''
        return None


    def namelist(self, datatype, allow_cached=True):
        ''' 格納されているアーカイブメンバのうち、datatypeにマッチするものの名前のコレクションを返す
        具象クラスで実際の動作を定義する '''
        raise NotImplementedError

    def getbyname(self, name, datatype):
        ''' 名前とdatatypeを指定してアーカイブメンバを読み出す
        _find_nameの実装があることを前提にデフォルト実装する '''
        # まずキャッシュを探索する
        if self.use_cache:
            data = self.mcreader.getbyname(name, datatype)
            if data is not None:
                return data
        # キャッシュが見つからないときは普通に探す
        fp = self.__class__._open_src(self.srcpath)
        try:
            path = self.__class__._find_name(fp, name)
            data = datatype.reader.read(path)
            # キャッシュに書き込む
            if self.use_cache:
                self.mcwriter.write(name, data)
            self.__class__._close_src(fp)
            # BytesIOの場合はcloseしてメモリを解放する
            if isinstance(path, BytesIO):
                path.close()
            return data
        except OSError as e:
            self.__class__._close_src(fp)
            raise e


    def getbylist(self, list_of_name, datatype, max_workers=None):
        ''' メンバの名前リストとdatatypeを指定してアーカイブメンバを読み出す
        返り値はndarrayのコレクション。リスト内に該当する名前のデータが存在しない場合、返り値のリストは名前のリストよりも短くなる '''
        q = self.getbylist_q(list_of_name, datatype, max_workers)
        return q.readAll()

    def getbylist_q(self, list_of_name, datatype, queue=None, max_workers=None):
        ''' メンバの名前リストとdatatypeを指定してアーカイブメンバを読み出す
        返り値は該当のndarrayをpopできるDataQueue。queueにNoneを指定した場合は新しい専用のqueueを作成し返す。
        既存のqueueを指定した場合は既存のqueueにデータがpushされる '''
        import ptfu
        return self._getbyq_skeleton(ptfu.kernel.texecutor, list_of_name, datatype, queue, max_workers)

    def getallbyqueue(self, datatype, queue=None, max_workers=None):
        ''' datatypeに合致するデータすべてを受け渡すDataQueueを返す'''
        import ptfu
        return self._getbyq_skeleton(ptfu.kernel.pexecutor, self.namelist(datatype), datatype, queue, max_workers)

    def _getbyq_skeleton(self, executor, collection_of_name, datatype, queue=None, max_workers=None):
        ''' メンバの名前リストとdatatypeを指定してアーカイブメンバを読み出す
        返り値は該当のndarrayをpopできるDataQueue。queueにNoneを指定した場合は新しい専用のqueueを作成し返す。
        既存のqueueを指定した場合は既存のqueueにデータがpushされる '''
        from .dataqueue import DataQueue
        from ptfu.kernel import kernel
 
        if queue is None:
            queue = DataQueue(len(collection_of_name))

        # リストが0件の場合は処理加速のためすぐreturnする
        if len(collection_of_name) == 0:
            return queue
        
        # キャッシュを使う場合、まずキャッシュを探索する
        if self.use_cache == True:

            # メモリキャッシュの探索
            try:
                hitnames = self.mcreader.hitnames(collection_of_name, datatype)
                if len(hitnames) > 0:
                    cacheset = self.mcreader.getbylist(collection_of_name, datatype)
                else:
                    cacheset = {}
            except:
                import traceback
                traceback.print_exc()

            # 取得したデータをすべてqueueに入れる
            if len(cacheset) > 0:
                queue.putAll(cacheset)
                # 未取得データの差分を作成する
                collection_of_name = set(collection_of_name) - set(hitnames)

            # ディスクキャッシュの探索
            try:
                if self.diskcache is not None:
                    hitnames = self.dcreader.hitnames(collection_of_name, datatype)
                if len(hitnames) > 0:
                    cacheset = self.dcreader.getbylist(collection_of_name, datatype)
                else:
                    cacheset = {}
            except:
                import traceback
                traceback.print_exc()

            # 取得したデータをすべてqueueに入れる
            if len(cacheset) > 0:
                queue.putAll(cacheset)
                # ディスクキャッシュから読み込んだデータをメモリキャッシュに書き込む
                cacheq = DataQueue(len(cacheset))
                cacheq.putAll(cacheset)
                f = kernel.texecutor.submit(self.mcwriter.writebyq, cacheq)

            # ディスクキャッシュでは読み込みに失敗することがあるので、実際に読めた件数をチェックする
            nactual_read = len(cacheset)
            dreadset = { x[0] for x in cacheset }

            # 未取得データの差分を作成する
            collection_of_name = set(collection_of_name) - set(dreadset)

            if len(collection_of_name) == 0:
                # キャッシュだけですべて読み込めたら終了する
                return queue
        
        cacheq = DataQueue(len(collection_of_name)) if self.use_cache else None
        import ptfu.functions as f
        ncpu = max_workers
        if ncpu is None:
            ncpu = max(f.cpu_count() // 2, 1)
        ndata = len(collection_of_name)
        if ndata >= ncpu * 10: # CPU数の10倍以上のデータがあればスレッドを分ける
            futures = []
            splitted = f.splitlist(list(collection_of_name), ncpu)
            sum = 0
            for plist in splitted:
                sum += len(plist)
            for partial_list in splitted:
                futures.append(executor.submit(self.__class__._getlistq_worker,
                                                self.__class__._find_name,
                                                self.__class__._open_src,
                                                self.__class__._close_src,
                                                self.srcpath,
                                                partial_list,
                                                datatype,
                                                queue,
                                                cacheq))
        else: # 少ない場合は1つだけ
            texecutor = kernel.texecutor
            future = texecutor.submit(self.__class__._getlistq_worker,
                                    self.__class__._find_name,
                                    self.__class__._open_src,
                                    self.__class__._close_src,
                                    self.srcpath,
                                    collection_of_name,
                                    datatype,
                                    queue,
                                    cacheq)
        if self.use_cache == True:
            self.mcwriter.writebyq(cacheq)
        return queue

        
    @staticmethod
    def _getlistq_worker(findname_func, open_func, close_func, srcpath, partial_list, datatype, queue, cacheq):
        ''' マルチスレッド・マルチプロセス用ワーカー関数。partial_listで与えられた名前を持つメンバをqueueに読み出す '''
        from .memcache import MemCache
        fp = open_func(srcpath)
        for name in partial_list:
            try:
                if isinstance(fp, MemCache):
                    data = fp.read(name) # dataはdict
                else:
                    path = findname_func(fp, name)
                    data = datatype.reader().read(path) # dataはdict
                if data is not None:
                    tup = (name, data)
                    queue.push(tup)
                    # キャッシュを使用する設定の場合、キャッシュ用のキューにも書き込む
                    if cacheq is not None:
                        cacheq.push(tup)
                    if path is not None and isinstance(path, BytesIO):
                        path.close()
                else:
                    queue.esizelock.acquire()
                    queue.expected_size.value -= 1
                    queue.esizelock.release()
            except OSError as e:
                import ptfu
                logger = ptfu.kernel.logger()
                logger.warning(str(e))
            except:
                import traceback
                traceback.print_exc()
        close_func(fp)
        return

    @staticmethod
    def _close_src(fp):
        ''' fpで与えられたアーカイブをクローズする '''
        fp.close()
        return

name = 'archivereader'


