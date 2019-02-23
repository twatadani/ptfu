''' cifar10batchreader.py: CIFAR 10のアーカイブ内のbatchを読み取るCifar10BatchReaderを記述する '''

from .archivereader import ArchiveReader

class Cifar10BatchReader(ArchiveReader):
    ''' CIFAR 10の個々のbatchを読み取るArchveReader。
    基本的にNestedArchiveReader経由で使うことを想定している。'''

    def __init__(self, srcpath, use_cache=True):
        from .storetype import StoreType
        super(Cifar10BatchReader, self).__init__(StoreType.CIFAR10BATCH, srcpath, use_cache)


    def namelist(self, datatype, allow_cached=True):
        ''' 格納されているアーカイブメンバのうち、datatypeにマッチするものの名前のコレクションを返す '''
        if self.namelist_cache is not None and allow_cached:
            return self.namelist_cache
        else:
            #print('namelistのためにcifar10batchreaderの_open_srcを呼び出します。')
            fp = self.__class__._open_src(self.srcpath)
            raw_namelist = fp[b'filenames']
            self.namelist_cache = set(map(lambda x: x.decode(), raw_namelist))
            #self.namelist_cache = fp[b'filenames']
            self.__class__._close_src(fp)
            return self.namelist_cache

    @staticmethod
    def _open_src(srcpath):
        ''' アーカイブをオープンし、fpを返す 
        fpはCifar10BatchReaderの場合、dict '''
        import _pickle as cPickle
        #print('type of srcpath:', type(srcpath))
        #print('srcpath;', srcpath)
        srcpath.seek(0)
        datadict = cPickle.load(srcpath, encoding='bytes')
        return datadict

    @staticmethod
    def _close_src(fp):
        ''' fpで与えられたアーカイブをクローズする '''
        fp = None
        return

    @staticmethod
    def _find_name(fp, name):
        ''' fp内からnameに該当するデータがあるかどうかを探す '''
        data = fp[b'data']
        names = fp[b'filenames']
        #print(type(name), name)
        bname = name.encode()
        if bname in names:
            return data[names.index(bname)]
        else:
            return None
