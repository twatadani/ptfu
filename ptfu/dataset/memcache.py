''' memcache.py: データ読み込みのメモリキャッシュを表すMemCacheを記述 '''

from io import BytesIO

from ptfu import get_default_logger

class MemCache:
    ''' メモリキャッシュを表すクラス
    内部にはcachedictがあり、
    key; name
    value; numpy ndarray
    の形で格納される '''

    def __init__(self): #, maximum_size=None
        # キャッシュ上限サイズについては未実装
        #self.max_size = maximum_size
        self.cachedict = {}
        return

    def __del__(self):
        ''' キャッシュをクリーンする '''
        self.cachedict = {}
        return

    def namelist(self):
        ''' 現在キャッシュにある項目の名前のコレクションを返す '''
        return self.cachedict.keys()

    def read(self, name):
        ''' nameに相当するキャッシュを読み込む。見つからない場合はNoneを返す '''
        if name in self.cachedict:
            return self.cachedict[name]
        else:
            return None

    def write(self, name, ndarray):
        ''' name, datatypeに相当するndarrayデータをキャッシュに書き込む '''
        self.cachedict[name] = ndarray
        return
