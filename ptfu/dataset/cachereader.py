''' cachereader.py: キャッシュ読み込み用のArchiveReaderを記述する '''

from .archivereader import ArchiveReader
from .memcache import MemCache

class CacheReader(ArchiveReader):
    ''' MemCacheをソースとするArchiveReader 
    MemCacheではself.srcpathはMemCacheオブジェクトである '''

    def __init__(self, srccache):
        from .storetype import StoreType
        super(CacheReader, self).__init__(StoreType.MEMCACHE, srccache, False)
        return

    def namelist(self, datatype, allow_cached=True):
        ''' 格納されているアーカイブメンバのうち、datatypeにマッチするものの名前のコレクションを返す 
        cacheではそもそもマッチするものしかキャッシュされていないので、単純に名前のリストを返す '''
        
        if allow_cached and self.namelist_cache is not None:
            return self.namelist_cache
        else:
            self.namelist_cache = self.srcpath.namelist()
            return self.namelist_cache

    def hitnames(self, collection_of_name, datatype):
        ''' 現在のキャッシュ内にcollection_of_nameにヒットするもののコレクションを返す '''
        names = set(self.namelist(datatype, False))
        hitnames = set(collection_of_name) & names
        return hitnames

    def getbylist(self, list_of_name, datatype, max_workers=None):
        ''' getbylistのより速い実装 '''    
        hitnames = self.hitnames(list_of_name, datatype)
        m = map(lambda x: (x, self.srcpath.read(x)), hitnames) # 名前とデータのタプル
        result = list(m)
        result = [ x for x in result if x[1] is not None ]
        return result

    @staticmethod
    def _find_name(fp, name):
        ''' fp内からnameに該当するデータがあるかどうかを探す
        CacheReaderではfpはmemcacheオブジェクト
         '''
        if name in fp.cachedict:
            return fp.cachedict[name]
        else:
            return None

    @staticmethod
    def _open_src(srcpath):
        ''' アーカイブをオープンし、fpを返す。
        CacheReaderではmemcacheオブジェクトを返す '''
        return srcpath

    @staticmethod
    def _close_src(fp):
        ''' アーカイブをクローズする。
        CacheReaderでは特になにもしない(いちいちMemCacheを解放することはしない) '''
        return

name = 'cachereader'

