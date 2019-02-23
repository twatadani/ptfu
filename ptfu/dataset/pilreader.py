''' PILReader.py: TypeReaderの子クラスで、PILで扱える画像フォーマットの読み出しを行う中間基底クラスPILReaderを記述する '''

from .typereader import TypeReader
from PIL import Image

import numpy as np

class PILReader(TypeReader):
    ''' PILで読み込める形式のTypeReaderの中間的実装 '''

    def __init__(self, datatype):
        super(PILReader, self).__init__(datatype)

    def read_from_rawfile(self, srcpath, filename, arcobj):
        ''' 生のディレクトリ内からデータを読み出し、ndarray形式で返す。
        srcpath: ソースディレクトリのパス
        filename: ファイル名
        srcpath, filenameはos.path.joinで結合するので、srcpathにファイル名まで記述して、filenameが空でも
        よい。zip, tarとの引数の数をそろえるため2引数としている。 
        arcobj: この関数では使用しない'''
        import os.path
        
        # 読み込むファイルのフルパス
        fullpath = os.path.join(srcpath, filename)
        # Image化
        img = Image.open(fullpath)
        return np.asarray(img)

    def read_from_bytes(self, bytesio):
        ''' インメモリに読み込まれたBytesIOオブジェクトからデータを読み出し、ndarray形式で返す。 '''
        img = Image.open(bytesio)
        return np.asarray(img)