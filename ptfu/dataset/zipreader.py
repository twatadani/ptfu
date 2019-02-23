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

    #@staticmethod
    #def rawmemberview(fp, membername):
        #''' membernameを与えてこのアーカイブ内のmemberに対するビューを取得する
        #fpは_open_srcで得られるもの。ZipReaderの場合ZipFileオブジェクト。
        #ビューはBufferedReader, File-like objectなど。read, seekができるオブジェクトを返す '''
        #from io import BufferedReader
        #rawview = fp.open(membername, mode='r')
        #return BufferedReader(rawview)


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


    #def open_src(self):
        #''' アーカイブソースをオープンする ZipReaderではZipFileをオープンする '''
        #if self.zfp is None:
            #self.zfp = ZipFile(self.srcpath, mode='r')
        #return self.zfp

    #def close_src(self):
        #''' アーカイブソースをクローズする ZipReaderではZipFileをクローズする '''
        #if self.zfp is not None:
            #self.zfp.close()
            #self.zfp = None
        #return

    #def storetype(self):
        #''' このArchiveReaderに対応するStoreTypeを返す '''
        #from .datatype import StoreType
        #return StoreType.ZIP

    #def alllist(self):
        #''' このアーカイブ内のすべての要素を返す '''
        #if self.zfp is None:
            #self.open_src()
        #namelist = self.zfp.namelist()
        #return namelist
