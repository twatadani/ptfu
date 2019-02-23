''' diskcache.py: データ読み込みのディスクキャッシュを表すDiskCacheを記述 '''

#from io import BytesIO

#from ptfu import get_default_logger
from tempfile import TemporaryDirectory
import os.path
import numpy as np

from glob import glob

class DiskCache:
    ''' Diskキャッシュを表すクラス
    内部的にはテンポラリディレクトリで管理されている '''

    def __init__(self):
        self.tmpdir = TemporaryDirectory()        
        return

    def __del__(self):
        ''' キャッシュをクリーンする '''
        print('ディスクキャッシュをクリーンアップします')
        self.tmpdir.cleanup()
        print('クリーンアップが終了しました。')
        return

    def namelist(self):
        ''' 現在キャッシュにある項目の名前のコレクションを返す '''
        globstr = os.path.join(self.tmpdir.name, '*.*')
        globresult = glob(globstr)
        #print('globresultは', len(globresult), '件でした')
        m = map(lambda x: os.path.basename(x), globresult)
        #print('sample of namelist of DiskCache:', list(m)[0])
        result = list(m)
        result = set(result)
        #print('resultの件数は', len(result), '件です。')
        return result


    def read(self, name):
        ''' nameに相当するキャッシュを読み込む。見つからない場合はNoneを返す '''
        fullpath = os.path.join(self.tmpdir.name, name)
        #print('ディスクキャッシュの読み込みfullpath:', fullpath)
        if os.path.exists(fullpath):
            return np.load(fullpath)
        else:
            return None

    def write(self, name, ndarray):
        ''' name, datatypeに相当するndarrayデータをキャッシュに書き込む '''
        fullpath = os.path.join(self.tmpdir, name)
        np.save(fullpath, ndarray, allow_pickle=False)
        return