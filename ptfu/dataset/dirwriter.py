''' dirwriter.py: ディレクトリ直接形式のStoreTypeWriterを記述する '''

from .archivewriter import ArchiveWriter

import os
import os.path

import numpy as np

class DirWriter(ArchiveWriter):
    ''' 生のディレクトリ内にファイルを格納するWriter '''

    def __init__(self, dstpath):
        ''' DirWriterのイニシャライザ
        dstpath: 書き込みを行うディレクトリ '''
        from .storetype import StoreType
        super(DirWriter, self).__init__(StoreType.DIR, dstpath)

    def _open_dst(self):
        ''' アーカイブファイルをオープンし、fpを返す。
        DirWriterではディレクトリの作成を行う '''
        os.makedirs(self.dstpath, exist_ok=True)
        return

    def _close_dst(self):
        ''' アーカイブファイルをクローズする。DirWriterでは何もしない '''
        return

    def _write_func(self, name, ndarray):
        ''' ソースがオープンされていることを前提にname, ndarrayで与えられる1件のデータを書き込む '''
        fullname = os.path.join(self.dstpath, name + '.npy')
        np.save(fullname, ndarray, allow_pickle=False)
        return
