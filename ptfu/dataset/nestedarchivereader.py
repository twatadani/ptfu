''' nestedarchivereader.py: ネストされた構造のarchive readerを記述 '''

from .archivereader import ArchiveReader

class InnerView:
    ''' ネストされたアーカイブ構造をマルチプロセスでやりとりするためのクラス '''

    def __init__(self, outerstoretype, outersrcpath, innerstoretype, innername, use_cache):
        self.outerstoretype = outerstoretype
        self.outersrcpath = outersrcpath
        self.innerstoretype = innerstoretype
        self.innername = innername
        self.use_cache = use_cache


class NestedArchiveReader(ArchiveReader):
    ''' アーカイブ内にアーカイブがあるタイプのネストされたアーカイブを読むためのリーダー
    ただし、ArchiveReaderの原則に従い、1インスタンスではネストの内側の1個のアーカイブのみを扱う '''

    def __init__(self, outerstoretype, outersrcpath, 
                innerstoretype, innername, use_cache=True):
        ''' イニシャライザ
        outerstoretype: ネストの外側のアーカイブのStoreType
        outersrcpath: ネストの外側のアーカイブのパス
        innerstoretype: ネストの内側のアーカイブのStoreType
        innername: ネストの内側に格納されているアーカイブ名
        '''
        self.inner_view = InnerView(outerstoretype, outersrcpath, innerstoretype, innername, use_cache)
        super(NestedArchiveReader, self).__init__(innerstoretype, self.inner_view, use_cache)
        return

    def namelist(self, datatype, allow_cached=True):
        ''' 格納されているアーカイブメンバのうち、datatypeにマッチするものの名前のコレクションを返す
        具象クラスで実際の動作を定義する '''
        if self.namelist_cache is not None and allow_cached:
            return self.namelist_cache
        else:
            fp = self.__class__._open_src(self.inner_view)
            innerreader = fp[1][0]
            self.namelist_cache = innerreader.namelist(datatype, allow_cached)
            self.__class__._close_src(fp)
            return self.namelist_cache

    @staticmethod
    def _open_src(srcpath):
        ''' アーカイブをオープンし、fpを返す 
        srcpathはNestedArchiveReaderの場合InnerViewオブジェクト
        fpはNestedArchiveReaderの場合、((outerreader, outerfp), (innerreader, innerfp))
        '''
        if not isinstance(srcpath, InnerView):
            raise ValueError('srcpath should be an instance of InnerView', srcpath)
        outerreader = srcpath.outerstoretype.reader(srcpath.outersrcpath, srcpath.use_cache)
        fp0 = outerreader._open_src(outerreader.srcpath)
        view_of_inner = outerreader.__class__.rawmemberview(fp0, srcpath.innername)
        innerreader = srcpath.innerstoretype.reader(view_of_inner, srcpath.use_cache)
        fp1 = innerreader._open_src(innerreader.srcpath)
        return ((outerreader, fp0), (innerreader, fp1))


    @staticmethod
    def _close_src(fp):
        ''' fpで与えられたアーカイブをクローズする '''
        inner_class = fp[1][0].__class__
        inner_fp = fp[1][1]
        inner_class._close_src(inner_fp)

        outer_class = fp[0][0].__class__
        outer_fp = fp[0][1]
        outer_class._close_src(outer_fp)
        return


    @staticmethod
    def _find_name(fp, name):
        ''' fp内からnameに該当するデータがあるかどうかを探す '''
        inner_class = fp[1][0].__class__
        inner_fp = fp[1][1]
        return inner_class._find_name(inner_fp, name)

name = 'nestedarchivereader'

    
