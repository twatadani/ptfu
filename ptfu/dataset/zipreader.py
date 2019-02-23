''' ZipReader.py: ZIPアーカイブ内に個別ファイルが格納されているタイプのデータ読みだしを行うZipReaderを記述。 '''

from .archivereader import ArchiveReader

from zipfile import ZipFile
from io import BytesIO

class ZipReader(ArchiveReader):
    ''' zipアーカイブされたデータを読み込むリーダー '''

    def __init__(self, srcpath, use_diskcache=True):
        from .storetype import StoreType
        super(ZipReader, self).__init__(StoreType.ZIP, srcpath, use_diskcache)
        return

    @staticmethod
    def _open_src(srcpath):
        ''' アーカイブをオープンし、fpを返す '''
        return ZipFile(srcpath, mode='r')

    def namelist(self, datatype, allow_cached=True):
        ''' 格納されているアーカイブメンバのうち、datatypeにマッチするものの名前のコレクションを返す '''
        if allow_cached and self.namelist_cache is not None:
            return self.namelist_cache
        else:
            fp = ZipReader._open_src(self.srcpath)
            nlist = fp.namelist()
            ZipReader._close_src(fp)
            self.namelist_cache = set([x for x in nlist if x.lower() == datatype.getext()])
            return self.namelist_cache

    @staticmethod
    def _find_name(fp, name):
        ''' fp内からnameに該当するデータがあるかどうかを探す '''
        data = fp.read(name)
        bytesio = BytesIO(data)
        return bytesio

