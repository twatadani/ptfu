''' TarReader.py: TARアーカイブ内に個別ファイルが格納されているタイプのデータ読みだしを行うTarReaderを記述。 '''

from .archivereader import ArchiveReader

from tarfile import TarFile
from io import BytesIO, FileIO
import os.path
from tempfile import TemporaryDirectory

class TarReader(ArchiveReader):
    ''' tarアーカイブされたデータを読み込むリーダー '''

    def __init__(self, srcpath, use_cache=True):
        from .storetype import StoreType
        super(TarReader, self).__init__(StoreType.TAR, srcpath, use_cache)
        return

    def diskcache_supported(self):
        return True

    def prepare_diskcache(self):
        ''' ディスクキャッシュを準備し、DiskCacheオブジェクトを返す '''
        from .diskcache import DiskCache
        from ..kernel import kernel
        dc = DiskCache()
        tmpdir = dc.tmpdir.name
        try:
            self.diskcache_future = kernel.texecutor.submit(TarReader._expand_diskcache, self.srcpath, tmpdir)
        except:
            import traceback
            traceback.print_exc()
        return dc

    @staticmethod
    def _expand_diskcache(tarpath, dstdir):
        ''' tarpathで指定されるtarアーカイブをdstdirに展開する '''
        from .. import get_default_logger
        try:
            logger = get_default_logger()
            logger.debug('TarReaderのディスクキャッシュを展開します。tarpath=' + str(tarpath) + ', dir=' + str(dstdir))
            with TarFile.open(name=tarpath, mode='r') as tf:
                tf.extractall(path=dstdir)
            logger.debug('TarReaderのディスクキャッシュ展開が終了しました。')
        except:
            import traceback
            traceback.print_exc()
        

    @staticmethod
    def rawmemberview(fp, membername):
        ''' membernameを与えてこのアーカイブ内のmemberに対するビューを取得する
        fpは_open_srcで得られるもの。TarReaderではTarFileオブジェクト。
        ビューはBufferedReader, File-like objectなど。read, seekができるオブジェクトを返す '''
        return fp.extractfile(membername)


    @staticmethod
    def _open_src(srcpath):
        ''' アーカイブをオープンし、fpを返す '''
        return TarFile.open(name=srcpath, mode='r')

    def namelist(self, datatype, allow_cached=True):
        ''' 格納されているアーカイブメンバのうち、datatypeにマッチするものの名前のコレクションを返す '''
        if allow_cached and self.namelist_cache is not None:
            return self.namelist_cache
        else:
            fp = TarReader._open_src(self.srcpath)
            nlist = fp.getnames()
            TarReader._close_src(fp)
            self.namelist_cache = set([x for x in nlist if x.lower().endswith(datatype.getext())])
            return self.namelist_cache

    @staticmethod
    def _find_name(fp, name):
        ''' fp内からnameに該当するデータがあるかどうかを探す '''
        # 劇的に遅い。
        stream = fp.extractfile(name) # stereamはBufferedReader
        stream.seek(0)
        return BytesIO(stream.read())

name = 'tarreader'

