# -*- coding: utf-8 -*-

''' DataTypeモジュール: データセットの元データ、保存データのタイプを規定する '''

from enum import Enum, auto

class DataType(Enum):
    ''' データセットに用いられるデータタイプを規定するenum '''

    # enum values
    PNG = auto()
    JPG = auto()
    DICOM = auto()
    NPY = auto() # .npy形式のndarray
    OTHER = auto() # その他

    def getext(self):
        ''' 拡張子の文字列を返す '''
        return self.name.lower()

    def reader(self):
        ''' このDataTypeに対応したTypeReaderの具象クラスを返す'''
        raise NotImplementedError

class StoreType(Enum):
    ''' データセットが格納されているアーカイブタイプを規定するenum '''

    # enum values
    DIR = auto() # ディレクトリ内に個々のファイルが多数あるタイプ
    TAR = auto() # tar(.gz) 形式
    ZIP = auto() # zip形式
    TFRECORD = auto() # TFRecord形式

    def getext(self):
        ''' 拡張子の文字列を得る '''
        return self.name.lower()

    def reader(self):
        ''' このStoreTypeに対応するArchiveReaderのクラスオブジェクトを返す '''
        raise NotImplementedError

    def readerfunc(self, typereader):
        ''' このStoreTypeではtypereaderがどのreadメソッドを使えば良いかを返す '''
        raise NotImplementedError

    def writer(self):
        ''' このStoreTypeに対応するArchiveWriterのクラスオブジェクトを返す '''
        raise NotImplementedError
