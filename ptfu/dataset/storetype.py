''' StoreTypeモジュール: データセットの格納タイプを規定する '''

from enum import Enum, auto

class StoreType(Enum):
    ''' データセットが格納されているアーカイブタイプを規定するenum '''

    # enum values
    DIR = auto() # ディレクトリ内に個々のファイルが多数あるタイプ
    TAR = auto() # tar(.gz) 形式
    ZIP = auto() # zip形式
    TFRECORD = auto() # TFRecord形式
    MEMCACHE = auto() # メモリキャッシュ
    NESTED = auto() # nested
    CIFAR10BATCH = auto() # CIFAR10アーカイブ内のバッチ

    def getext(self):
        ''' 拡張子の文字列を得る '''
        return self.name.lower()

    @classmethod
    def fromsrcstring(cls, src):
        ''' パス文字列から一致するStoreType enumメンバを返す。
        拡張子なしの場合や該当するenumメンバがない場合はNoneが返る '''
        import os.path
        ext = os.path.splitext(src)[1]
        if ext == '':
            return None
        else:
            for storetype in cls:
                cext = storetype.getext()
                if cext != '':
                    if src.endswith(cext):
                        return storetype
            return None
            
# DIRは拡張子を持たないので、空文字列を返す
# 基本的にgetextは書き込み用
StoreType.DIR.getext = (lambda: '')
StoreType.TAR.getext = (lambda: 'tar.gz')
StoreType.MEMCACHE.getext = (lambda: '')
StoreType.NESTED.getext = (lambda: '')

# readerの実体をenumメンバに与える
from .dirreader import DirReader
from .zipreader import ZipReader
from .tarreader import TarReader
from .cachereader import CacheReader
from .nestedarchivereader import NestedArchiveReader
from .cifar10batchreader import Cifar10BatchReader

StoreType.DIR.reader = DirReader
StoreType.TAR.reader = TarReader
StoreType.ZIP.reader = ZipReader
StoreType.MEMCACHE.reader = CacheReader
StoreType.NESTED.reader = NestedArchiveReader
StoreType.CIFAR10BATCH.reader = Cifar10BatchReader

# writerの実体をenumメンバに与える
from .dirwriter import DirWriter
from .zipwriter import ZipWriter
from .tarwriter import TarWriter
from .tfrecordwriter import TFRecordWriter

StoreType.DIR.writer = DirWriter
StoreType.TAR.writer = TarWriter
StoreType.ZIP.writer = ZipWriter
StoreType.TFRECORD.writer = TFRecordWriter

name = 'storetype'

