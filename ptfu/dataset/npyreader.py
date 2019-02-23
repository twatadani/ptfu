''' NPYReader.py: TypeReaderの子クラスで、numpy .npyフォーマットの読み出しを行うクラスNPYReaderを記述する '''

from .typereader import TypeReader

import os.path
import numpy as np

class NPYReader(TypeReader):

    def __init__(self):
        from .datatype import DataType
        super(NPYReader, self).__init__(DataType.NPY)

    def read(self, path):
        ''' ファイルパスまたはBytesIOオブジェクトから読み込む '''
        return np.load(path)

    #def read_from_rawfile(self, srcpath, filename, arcobj):
        #''' 生のディレクトリ内からデータを読み出し、ndarray形式で返す。
        #srcpath: ソースディレクトリのパス
        #filename: ファイル名
        #srcpath, filenameはos.path.joinで結合するので、srcpathにファイル名まで記述して、filenameが空でも
       # よい。zip, tarとの引数の数をそろえるため2引数としている。 
        #arcobj: この関数では使用しない '''
        #fullpath = os.path.join(srcpath, filename)
        #return np.load(fullpath)

    #def read_from_bytes(self, bytesio):
        #''' インメモリに読み込まれたBytesIOオブジェクトからデータを読み出し、ndarray形式で返す。 '''
        #return np.load(bytesio)

    # singleton-like instance
    #reader = NPYReader()