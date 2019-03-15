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

    def _write_func(self, name, datadict):
        ''' ソースがオープンされていることを前提にdatadictで
        与えられる1件のデータを書き込む'''
        raise NotImplementedError

    def write(self, datadict, name=None):
        ''' datadictを指定して1件のデータを書き込む nameはdatadict内にあれば指定する必要はない'''
        self._open_dst()
        if name is not None:
            self._write_func(name, datadict)
        elif 'name' in datadict:
            self._write_func(datadict['name'], datadict)
        else:
            self._close_dst()
            raise ValueError('name is not found neither in name param or datadict')
        self._close_dst()
        return

    def writebylist(self, collection_of_data):
        ''' collectionを与えて複数のデータを書き込む
        collection_of_dataの要素は(name, datadict)のタプル
        datadict内に'name'キーがある場合はdatadict単独でも可 '''
        self._open_dst()
        for data in collection_of_data:
            if isinstance(data, list) or isinstance(data, tuple):
                self._write_func(data[0], data[1])
            elif isinstance(data, dict):
                self._write_func(data['name'], data)
        self._close_dst()
        return
    
    def writebyq(self, queue):
        ''' queueを与えて複数のデータを書き込む
        queueからpopされるデータは(name, datadict)のタプル
        datadict内に'name'キーがある場合はdatadict単独でも可 '''
        try:
            with self:
                while queue.hasnext():
                    data = queue.pop()
                    if data is not None:
                        if isinstance(data, list) or isinstance(data, tuple):
                            self._write_func(data[0], data[1])
                        elif isinstance(data, dict):
                            self._write_func(data['name'], data)
                        else:
                            raise ValueError('writebyq: data must be tuple, list or dict')
                return
        except:
            import traceback
            traceback.print_exc()

