''' archivewriter.py: 各StoreTypeに対応したWriterの基底クラスを記述 '''

class ArchiveWriter:
    ''' データセット格納形式StoreTypeに対応したWriterの基底クラス
    継承クラスへの実装メソッドの指針:
    _open_dst: fpを返す
    _close_dst(self, fp): fp.close()するだけならデフォルト実装あり
    _write_func(self, name, ndarray): 1件のデータ書き込み
     '''

    def __init__(self, storetype, dstpath):

        self.storetype = storetype
        self.dstpath = dstpath
        self.fp = None
        return

    def __enter__(self):
        ''' withブロックでarchivewriterを使うためのメソッド
        デフォルト実装では_opne_dstを呼び出すのみ '''
        self._open_dst()
        return

    def __exit__(self, exc_type, exc_value, traceback):
        ''' withブロックでarchivewriterを使うためのメソッド
        デフォルト実装では_close_dstを呼び出すのみ '''
        self._close_dst()
        return

    def _open_dst(self):
        ''' アーカイブファイルをオープンし, self.fpを設定する。
        fpはクローズする際にfp.close()できるもの '''
        raise NotImplementedError

    def _close_dst(self):
        ''' アーカイブファイルをクローズする。 '''
        if self.fp is not None:
            self.fp.close()
        self.fp = None
        return

    def _write_func(self, name, ndarray):
        ''' ソースがオープンされていることを前提にname, ndarrayで
        与えられる1件のデータを書き込む '''
        raise NotImplementedError

    def write(self, name, ndarray):
        ''' nameとndarrayを指定して1件のデータを書き込む '''
        self._open_dst()
        self._write_func(name, ndarray)
        self._close_dst()
        return

    def writebylist(self, collection_of_data):
        ''' collectionを与えて複数のデータを書き込む
        collection_of_dataの要素は(name, ndarray)のタプル '''
        self._open_dst()
        for data in collection_of_data:
            self._write_func(data[0], data[1])
        self._close_dst()
        return
    
    def writebyq(self, queue):
        ''' queueを与えて複数のデータを書き込む
        queueからpopされるデータは(name, ndarray)のタプル '''
        try:
            #print('ArchiveWriter writebyqが呼び出されました。')
            #print('queue: 全件数は' + str(queue.datanumber()))
            #self._open_dst()
        
            #print('writebyq _open_dstが終了しました。')
            with self:
                while queue.hasnext():
                    #print('whileループに入りました。')
                    data = queue.pop()
                    #print('popが終了しました。poppednumber=' + str(queue.poppednumber()))
                    if data is not None:
                        self._write_func(data[0], data[1])
                    #print('_write_funcが終了しました。')
                #print('writebyq whileループが終了しました。')
                #self._close_dst()
                #print('ArchiveWriter writebyqを終了します。')
                return
        except:
            import traceback
            traceback.print_exc()

    #def __init__(self, dstpath, **options):
        #''' ArchiveWriterのイニシャライザ
        #dstpath: 書き込みを行うディレクトリまたはアーカイブのパス
        #options: オプション 具象クラスで自由に規定する '''
        #self.dstpath = dstpath # 書き込みを行うディレクトリまたはアーカイブのパス
        #self.writestarted = False
        #self.options = options
        #return

    #def __del__(self):
        #''' StoreTypeWriterのデストラクタ
        #デフォルト動作ではclose_dstのみを行う '''
        #self.close_dst()

    #def open_dst(self):
        #''' アーカイブのオープン デフォルトではなにもしない 
        #具象クラスで必要ならばオーバーライドする '''
        #return

    #def close_dst(self):
        #''' アーカイブのクローズ デフォルトではなにもしない
        #具象クラスで必要ならばオーバーライドする '''
        #return

    #def appendNext(self, name, ndarray):
        #''' iteratorから得た1件のデータを書き込む '''
        #if self.writestarted == False:
            #self.open_dst()
            #self.writestarted = True
        #self._appendNext(name, ndarray)
        #return

    #def _appendNext(self, name, ndarray):
        #''' iteratorから得た1件のデータを書き込む。具象クラスで定義する '''
        #raise NotImplementedError