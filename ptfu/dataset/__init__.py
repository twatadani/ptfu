# -*- coding: utf-8 -*-

''' Personal TensorFlow Utility dataset サブパッケージ: データセット作成用ユーティリティ '''

from .datatype import DataType as DataType
from .datatype import StoreType as StoreType
from .srcreader import SrcReader as SrcReader
from .dstwriter import DstWriter as DstWriter
from .datasetcreator import DatasetCreator as DatasetCreator

# DICOMは例外なのでgetextを定義しなおす
DataType.DICOM.getext = (lambda self: 'dcm')

# 各メンバに対してreader関数を設定する
from .srcreader import JPGReader, PNGReader, DICOMReader, NPYReader
# singleton instances
pngreader = PNGReader()
jpgreader = JPGReader()
dicomreader = DICOMReader()
npyreader = NPYReader()

DataType.PNG.reader = (lambda : pngreader)
DataType.JPG.reader = (lambda : jpgreader)
DataType.DICOM.reader = (lambda : dicomreader)
DataType.NPY.reader = (lambda : npyreader)

# DIRは拡張子を持たないので、空文字列を返す
StoreType.DIR.getext = (lambda self: '')

# readerの実体をenumメンバに与える
from .srcreader import DirReader, TarReader, ZipReader
StoreType.DIR.reader = (lambda : DirReader)
StoreType.TAR.reader = (lambda : TarReader)
StoreType.ZIP.reader = (lambda : ZipReader)

# readerfuncの実体をそれぞれのenumメンバに与える
StoreType.DIR.readerfunc = (lambda typereader: typereader.read_from_rawfile)
StoreType.TAR.readerfunc = (lambda typereader: typereader.read_from_tar)
StoreType.ZIP.readerfunc = (lambda typereader: typereader.read_from_zip)

# writerの実体をenumメンバに与える
from .dstwriter import DirWriter, ZipWriter, TarWriter, TFRecordWriter
StoreType.DIR.writer = (lambda: DirWriter)
StoreType.TAR.writer = (lambda: TarWriter)
StoreType.ZIP.writer = (lambda: ZipWriter)
StoreType.TFRECORD.writer = (lambda: TFRecordWriter)

name='ptfu.dataset'
