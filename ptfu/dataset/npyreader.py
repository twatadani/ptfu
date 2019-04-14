''' NPYReader.py: TypeReaderの子クラスで、numpy .npyフォーマットの読み出しを行うクラスNPYReaderを記述する '''

from .typereader import TypeReader

import numpy as np

class NPYReader(TypeReader):

    def __init__(self):
        from .datatype import DataType
        super(NPYReader, self).__init__(DataType.NPY)

    def read(self, path):
        ''' ファイルパスまたはBytesIOオブジェクトから読み込み、datadictを返す '''
        datadict = {}
        ndarray = np.load(path)
        datadict['data'] = ndarray
        return datadict

name = 'npyreader'


