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
        CIFAR10の場合、_find_nameが特殊で(fp, index)の形式で与えられる。'''
        fp = bytesio_or_file[0]
        index = bytesio_or_file[1]

        data = fp[b'data']
        names = fp[b'filenames']
        labels = fp[b'labels']

        ndarray = data[index]
        ndarray = np.reshape(ndarray, (3, 32, 32)).transpose(1, 2, 0)

        datadict = {}
        datadict['name'] = names[index].decode()
        datadict['data'] = ndarray
        datadict['label'] = labels[index]

        return datadict

name = 'cifar10datareader'

        
