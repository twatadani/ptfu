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
            fp = self.__class__._open_src(self.srcpath)
            raw_namelist = fp[b'filenames']
            self.namelist_cache = set(map(lambda x: x.decode(), raw_namelist))
            self.__class__._close_src(fp)
            return self.namelist_cache

    @staticmethod
    def _open_src(srcpath):
        ''' アーカイブをオープンし、fpを返す 
        fpはCifar10BatchReaderの場合、dict '''
        import _pickle as cPickle
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
        ''' fp内からnameに該当するデータがあるかどうかを探す 
        CIFAR10BatchReaderの場合、datadictを作成できるようにするため
        (fp, 該当データのあるindex)を返すようにする '''
        names = fp[b'filenames']
        bname = name.encode()
        if bname in names:
            return (fp, names.index(bname))
        else:
            return None

name = 'cifar10batchreader'

