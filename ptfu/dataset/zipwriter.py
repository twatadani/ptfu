''' zipwriter.py: Zipアーカイブに対応したStoreTypeWriter ZipWriterを記述する '''

from .archivewriter import ArchiveWriter

import os
import os.path
from io import BytesIO
from zipfile import ZipFile
import numpy as np

class ZipWriter(ArchiveWriter):
    ''' 生のディレクトリ内にファイルを格納するWriter '''

    def __init__(self, dstpath):
        ''' ZipWriterのイニシャライザ
        dstpath: 書き込みを行うzipファイル '''
        from .storetype import StoreType
        super(ZipWriter, self).__init__(StoreType.ZIP, dstpath)
        return
        #self.zfp = None # ZipFile object

    def _open_dst(self):
        ''' アーカイブファイルをオープンし、self.fpを設定する。 '''
        parent_dir = os.path.dirname(self.dstpath)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        self.fp = ZipFile(self.dstpath, mode='w')
        return

    def _write_func(self, name, ndarray):
        ''' ソースがオープンされていることを前提にname, ndarrayで
        与えられる1件のデータを書き込む '''
        bytesio = BytesIO()
        np.save(bytesio, ndarray, allow_pickle=False)
        self.fp.writestr(name + '.npy', bytesio.getbuffer())
        bytesio.close()
        return

    #def open_dst(self):
        #''' アーカイブのオープン ZipWriterではzipアーカイブのオープンを行う '''
        #if self.zfp is None:
            #parent_dir = os.path.dirname(self.dstpath)
            #if not os.path.exists(parent_dir):
                #os.makedirs(parent_dir)
            #self.zfp = ZipFile(self.dstpath, mode='w')
        #return

    #def close_dst(self):
        #''' アーカイブのクローズ ZipWriterではzipアーカイブのクローズを行う '''
        #if self.zfp is not None:
            #self.zfp.close()
            #self.zfp = None

    #def _appendNext(self, name, ndarray):
        #''' iteratorから得た1件のデータを書き込む '''
        #buf = BytesIO()
        #np.save(buf, ndarray, allow_pickle=False)
        #self.zfp.writestr(name + '.npy', buf.getbuffer())
        #buf.close()
        #return