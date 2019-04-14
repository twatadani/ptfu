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

    def read(self, bytesio_or_file):
        ''' BytesIOまたはファイルパスを与え、datadictの形式で読み出す。'''
        ndarray =  DICOMReader.dcm2npy(pydicom.dcmread(bytesio_or_file))
        datadict = {}
        datadict['data'] = ndarray
        return datadict

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

name = 'dicomreader'

