''' cachewriter.py: キャッシュ書き込み用のArchiveWriterを記述する '''

from .archivewriter import ArchiveWriter
from .memcache import MemCache

class CacheWriter(ArchiveWriter):
    ''' MemCacheをソースとするArchiveWriter 
    CacheWriterではself.dstpathはMemCacheオブジェクトである '''

    def __init__(self):
        from .storetype import StoreType
        super(CacheWriter, self).__init__(StoreType.MEMCACHE, MemCache())


    def _open_dst(self):
        ''' アーカイブファイルをオープンし, self.fpを設定する。 '''
        self.fp = self.dstpath
        return

    def _close_dst(self):
        ''' アーカイブファイルをクローズする。
        CacheWriterでは特に何もしない '''
        self.fp = None
        return

    def _write_func(self, name, ndarray):
        ''' ソースがオープンされていることを前提にname, ndarrayで
        与えられる1件のデータを書き込む 
        '''
        self.dstpath.write(name, ndarray)