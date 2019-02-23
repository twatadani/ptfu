''' DICOMReader.py: TypeReaderの子クラスで、DICOMフォーマットの読み出しを行うクラスDICOMReaderを記述する '''

from .typereader import TypeReader
import ptfu.dicomutil as dicomutil

import os.path

import pydicom
import numpy as np

class DICOMReader(TypeReader):

    def __init__(self):
        from .datatype import DataType
        super(DICOMReader, self).__init__(DataType.DICOM)

    def read_from_rawfile(self, srcpath, filename, arcobj):
        ''' 生のディレクトリ内からデータを読み出し、ndarray形式で返す。
        srcpath: ソースディレクトリのパス
        filename: ファイル名
        srcpath, filenameはos.path.joinで結合するので、srcpathにファイル名まで記述して、filenameが空でも
        よい。zip, tarとの引数の数をそろえるため2引数としている。 
        arcobj: この関数では使用しない'''

        fullpath = os.path.join(srcpath, filename)
        return DICOMReader.dcm2npy(pydicom.dcmread(fullpath))

    def read_from_bytes(self, bytesio):
        ''' インメモリに読み込まれたBytesIOオブジェクトからデータを読み出し、ndarray形式で返す。 '''
        return DICOMReader.dcm2npy(pydicom.dcmread(bytesio))

    @staticmethod
    def dcm2npy(dcmdataset, dtype=np.float32):
        ''' pydicom datasetからnumpyに変換する '''
        # pydicomの読み込みは完全でないので、データを変換する
  

        converted = dicomutil.bitconvert(dcmdataset)
        interarray = None
        if converted.PixelRepresentation == 0: # unsigned
            interarray = np.uint16(converted.pixel_array)
        else:
            interarray = converted.pixel_array
        nparray = dtype(interarray)

        # 'MONOCHROME1'と'MONOCHROME2'に対応
        photointerpretation = converted.data_element('PhotometricInterpretation').value
        if photointerpretation == 'MONOCHROME1':
            nparray = -nparray

        return nparray

    # singleton-like instance
    #reader = DICOMReader()