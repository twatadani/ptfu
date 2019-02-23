''' cifar10datareader.py: CIFAR 10の格納データの読み取りを行う '''

from .typereader import TypeReader
import numpy as np

class Cifar10DataReader(TypeReader):
    ''' Cifar10の個別データを読み取るTypeReader '''

    def __init__(self):
        ''' TypeReader イニシャライザ
        srcdatatype: DataType Enumのメンバ '''
        from .datatype import DataType
        super(Cifar10DataReader, self).__init__(DataType.CIFAR10)

    def read(self, bytesio_or_file):
        ''' BytesIOまたはファイルパスを与え、ndarrayの形式で読み出す。
        Cifar10の場合はすでにndarrayの形で与えられるので、整形して返す '''
        #print('Cifar10DataReaderのreadが呼び出されました。')
        ndarray = bytesio_or_file
        ndarray = np.reshape(ndarray, (3, 32, 32)).transpose(1, 2, 0)
        #print('readの結果: ' + str(ndarray.shape))
        return ndarray
        