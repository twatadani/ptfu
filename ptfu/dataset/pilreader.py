''' PILReader.py: TypeReaderの子クラスで、PILで扱える画像フォーマットの読み出しを行う中間基底クラスPILReaderを記述する '''

from .typereader import TypeReader
from PIL import Image

import numpy as np

class PILReader(TypeReader):
    ''' PILで読み込める形式のTypeReaderの中間的実装 '''

    def __init__(self, datatype):
        super(PILReader, self).__init__(datatype)

    def read(self, bytesio_or_file):
        ''' BytesIOまたはファイルパスを与え、datadictの形式で読み出す。 '''
        img = Image.open(bytesio)
        ndarray = np.asarray(img)
        datadict = {}
        datadict['data'] = ndarray
        return datadict

name = 'pilreader'
