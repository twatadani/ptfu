''' DataTypeモジュール: データセットの元データ、保存データのタイプを規定する '''

from enum import Enum, auto

class DataType(Enum):
    ''' データセットに用いられるデータタイプを規定するenum '''

    # enum values
    PNG = auto()
    JPG = auto()
    DICOM = auto()
    NPY = auto() # .npy形式のndarray
    PKL = auto() # pickle
    CIFAR10 = auto() # CIFAR10 data
    OTHER = auto() # その他

    def getext(self):
        ''' 拡張子の文字列を返す '''
        return self.name.lower()

# DICOMは例外なのでgetextを定義しなおす
DataType.DICOM.getext = (lambda : 'dcm')

# 各メンバに対してreaderを設定する
from .jpgreader import JPGReader
from .pngreader import PNGReader
from .dicomreader import DICOMReader
from .npyreader import NPYReader
from .pklreader import PKLReader
from .cifar10datareader import Cifar10DataReader

DataType.PNG.reader = PNGReader
DataType.JPG.reader = JPGReader
DataType.DICOM.reader = DICOMReader
DataType.NPY.reader = NPYReader
DataType.PKL.reader = PKLReader
DataType.CIFAR10.reader = Cifar10DataReader
