''' zipwriter.py: Zipアーカイブに対応したStoreTypeWriter ZipWriterを記述する '''

from .archivewriter import ArchiveWriter

import os
import os.path
from io import BytesIO
from zipfile import ZipFile
import pickle

class ZipWriter(ArchiveWriter):
    ''' 生のディレクトリ内にファイルを格納するWriter '''

    def __init__(self, dstpath):
        ''' ZipWriterのイニシャライザ
        dstpath: 書き込みを行うzipファイル '''
        from .storetype import StoreType
        super(ZipWriter, self).__init__(StoreType.ZIP, dstpath)
        return

    def _open_dst(self):
        ''' アーカイブファイルをオープンし、self.fpを設定する。 '''
        parent_dir = os.path.dirname(self.dstpath)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        self.fp = ZipFile(self.dstpath, mode='w')
        return

    def _write_func(self, name, datadict):
        ''' ソースがオープンされていることを前提にname, ndarrayで
        与えられる1件のデータを書き込む '''
        bytesio = BytesIO()
        pickle.dump(datadict, bytesio)
        self.fp.writestr(name + '.pkl', bytesio.getbuffer())
        bytesio.close()
        return
