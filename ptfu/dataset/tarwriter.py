''' tarwriter.py: Tarアーカイブに書き込むStoreTypeWriterを記述する '''

from .archivewriter import ArchiveWriter

import os
import os.path
from io import BytesIO
from tarfile import TarFile, TarInfo
import pickle

class TarWriter(ArchiveWriter):
    ''' Tarアーカイブ内にファイルを格納するWriter '''

    def __init__(self, dstpath):
        ''' TarWriterのイニシャライザ
        dstpath: 書き込みを行うディレクトリ '''
        from .storetype import StoreType
        super(TarWriter, self).__init__(StoreType.TAR, dstpath)
        if os.path.splitext(dstpath)[-1] == '.tar':
            self.dstpath += '.gz'

    def _open_dst(self):
        ''' アーカイブファイルをオープンし、self.fpを設定する。 '''
        parent_dir = os.path.dirname(self.dstpath)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        self.fp = TarFile.open(name=self.dstpath, mode='w:gz')
        return

    def _write_func(self, name, datadict):
        ''' ソースがオープンされていることを前提にname, datadictで
        与えられる1件のデータを書き込む '''
        bytesio = BytesIO()
        pickle.dump(datadict, bytesio)
        info = TarInfo(name=name + '.pkl')
        info.size = len(bytesio.getbuffer())
        bytesio.seek(0)
        self.fp.addfile(info, fileobj=bytesio)
        bytesio.close()
        return

name = 'tarwriter'

