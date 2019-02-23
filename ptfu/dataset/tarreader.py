''' TarReader.py: TARアーカイブ内に個別ファイルが格納されているタイプのデータ読みだしを行うTarReaderを記述。 '''

from .archivereader import ArchiveReader

from tarfile import TarFile
from io import BytesIO, FileIO
import os.path
from tempfile import TemporaryDirectory

class TarReader(ArchiveReader):
    ''' tarアーカイブされたデータを読み込むリーダー '''

    # class variable
    #diskcachedict = {}
    
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
        #print('TarReaderのprepare_diskcacheが呼び出されました。')
        dc = DiskCache()
        #print('prepare_diskcache 1 self.srcpath=', self.srcpath)
        tmpdir = dc.tmpdir.name
        #TarReader.diskcachedict[self.srcpath] = tmpdir
        #print('prepare_diskcache 2 tmpdir=', tmpdir)
        try:
            self.diskcache_future = kernel.texecutor.submit(TarReader._expand_diskcache, self.srcpath, tmpdir)
        except:
            import traceback
            traceback.print_exc()
        #print('prepare_diskcache 3')
        return dc

    @staticmethod
    def _expand_diskcache(tarpath, dstdir):
        ''' tarpathで指定されるtarアーカイブをdstdirに展開する '''
        from .. import get_default_logger
        #print('_expand_diskcacheが起動されました。')
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
            #print('tarreader len(nlist)=', len(nlist))
            TarReader._close_src(fp)
            self.namelist_cache = set([x for x in nlist if x.lower().endswith(datatype.getext())])
            #print('tarreader len(self.namelist_cache)=', len(self.namelist_cache))
            return self.namelist_cache

    @staticmethod
    def _find_name(fp, name):
        ''' fp内からnameに該当するデータがあるかどうかを探す '''
        # 劇的に遅い。
        stream = fp.extractfile(name) # stereamはBufferedReader
        stream.seek(0)
        return BytesIO(stream.read())


        #logger = get_default_logger()
        #self.tfp = None

        #if self.use_diskcache is True:
            #from concurrent.futures import ProcessPoolExecutor
            #from tempfile import TemporaryDirectory

            #if not srcpath in TarReader.diskcachedict: # 同じファイルに対するディスクキャッシュは1つのみ
                #self.tmpdir = TemporaryDirectory()
                #TarReader.diskcachedict[srcpath] = self.tmpdir
                #self.pexecutor = ProcessPoolExecutor(1)
                #future = self.pexecutor.submit(self._prepare_diskcache, srcpath, self.tmpdir)
                #self.diskcache_owner = True
            #else: # すでにディスクキャッシュがある場合
                #logger.debug('TarReaderのディスクキャッシュを流用します')
                #self.tmpdir = TarReader.diskcachedict[srcpath]
                #self.diskcache_owner = False
                
        #return

    #def __del__(self):
        #if hasattr(self, 'tmpdir') and self.tmpdir is not None:
            #if hasattr(self, 'diskcache_owner') and self.diskcache_owner == True:
                #try:
                    #self.tmpdir.cleanup()
                #except:
                    # エラーが起きてもなにもしない
                    #a = None
        #if hasattr(self, 'pexecutor') and self.pexecutor is not None:
            #try:
                #self.pexecutor.shutdown()
            #except:
                # エラーが起きても何もしない
                #a = None
        #super(TarReader, self).__del__()
        #return

    #def open_src(self):
        #''' アーカイブソースをオープンする TarReaderではTarFileをオープンする '''
        #if self.tfp is None:
            #self.tfp = TarFile.open(name=self.srcpath, mode='r')
        #return self.tfp

    #def close_src(self):
        #''' アーカイブソースをクローズする TarReaderではTarFileをクローズする '''
        #if self.tfp is not None:
            #self.tfp.close()
            #self.tfp = None

    #def get_bytesio_byname(self, name):
        #''' nameで指定されるアーカイブメンバをBytesIOに読み込み、返す '''
        #self.open_src()
        #stream = self.tfp.extractfile(name) # stereamはBufferedReader
        #stream.seek(0)
        #return BytesIO(stream.read())

    #def getbyname(self, name, treader):
        #''' nameとTypeReaderを指定してアーカイブメンバを読み出す '''
        #try:
            #if self.use_diskcache: # まずディスクキャッシュを探す
                #fullname = os.path.join(self.tmpdir.name, name)
                #if os.path.exists(fullname):
                    #arcobj = self.open_src()
                    #return treader.read_from_rawfile(self.tmpdir.name, name, arcobj)

            # キャッシュヒットしなかった場合 親クラスの実装を使う
            #return super(TarReader, self).getbyname(name, treader)
        #except:
            #import traceback
            #traceback.print_exc()

    #def storetype(self):
        #''' このArchiveReaderに対応するStoreTypeを返す '''
        #from .datatype import StoreType
        #return StoreType.TAR

    #def alllist(self):
        #''' このアーカイブ内のすべてのメンバをリストアップする '''
        #if self.tfp is None:
            #self.open_src()
        #return self.tfp.getnames()

    #@staticmethod
    #def _prepare_diskcache(srcpath, tmpdir):
        #''' 読み込み用のディスクキャッシュを準備する '''
        #import tarfile
        #from ..logger import get_default_logger
        #logger = get_default_logger()
        #try:
            #with tarfile.open(name=srcpath, mode='r') as tfp:
                #logger.debug('TarReaderのディスクキャッシュを作成します: ' + tmpdir.name)
                #tfp.extractall(path=tmpdir.name)
        #except:
            #import traceback
            #traceback.print_exc()
        #return