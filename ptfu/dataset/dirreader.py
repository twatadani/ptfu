''' DirReader.py: 直接ディレクトリ内に個別ファイルが格納されているタイプのデータ読みだしを行うDirReaderを記述。 '''

from .archivereader import ArchiveReader

class DirReader(ArchiveReader):
    ''' ディレクトリ内に生ファイルが格納されている形式のデータを読み出すArchiveReader '''

    def __init__(self, srcpath, use_cache=False):
        from .storetype import StoreType
        super(DirReader, self).__init__(StoreType.DIR, srcpath, use_cache)


    def namelist(self, datatype, allow_cached=True):
        ''' 格納されているアーカイブメンバのうち、datatypeにマッチするものの名前のコレクションを返す '''
        if allow_cached and self.namelist_cache is not None:
            return self.namelist_cache
        import os, os.path, glob
        glob_elem = '**' + os.sep + '*'
        globstr_base = os.path.join(self.srcpath, glob_elem)
        globstr_large = globstr_base + datatype.getext().upper()
        globstr_small = globstr_base + datatype.getext()
        result_large = glob.glob(globstr_large)
        result_small = glob.glob(globstr_small)
        result = result_large
        result.extend(result_small)
        return set(result)

    @staticmethod
    def _find_name(fp, name):
        ''' fp内からnameに該当するデータがあるかどうかを探す 
        DirReaderではfpは単にディレクトリを表すパス文字列 '''
        import os.path
        path = os.path.join(fp, name)
        if not os.path.exists(path):
            import os, glob
            globstr = os.path.join(fp, '**' + os.sep + name)
            path = glob.glob(globstr)[0]
        if os.path.exists(path):
            return path
        else:
            raise OSError(fp + '内に名前' + name + 'を持つデータが見つかりませんでした。')

    @staticmethod
    def _open_src(srcpath):
        ''' アーカイブをオープンし、fpを返す 
        DirReaderではパスをそのまま返す '''
        return srcpath

    @staticmethod
    def _close_src(fp):
        ''' fpで与えられたアーカイブをクローズする
        DirReaderではダミー '''
        return
