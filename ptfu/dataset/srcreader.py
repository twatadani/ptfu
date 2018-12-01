# -*- coding: utf-8 -*-

''' SrcReaderモジュール: データセットの元データの読み込み処理を記述する '''

import os
import os.path
import glob
import itertools
from zipfile import ZipFile
from tarfile import TarFile

import pydicom
from PIL import Image
import numpy as np

class SrcReader:
    ''' データソースからの読み出しを担当するクラス '''

    def __init__(self, srcdatatype, srcstoretype, srcpath):
        ''' SrcReaderのイニシャライザ。
        srcdatatype: データセット元データの形式。DataType enumのいずれかを指定する。
        srcstoretype: データセット元データの格納形式。StoreType enumのいずれかを指定する。
        srcpath: データセット元データの場所。strで指定する。 '''

        self.datatype = srcdatatype
        self.storetype = srcstoretype
        self.srcpath = os.path.expanduser(srcpath)

        self.areader = self.storetype.reader()(self.datatype, self.srcpath) # archive reader
        self.treader = self.datatype.reader # type reader
        return

    def iterator(self):
        ''' 個々のデータを読んでゆくイテレータを返すメンバ関数。
        返すデータの型は(データの名前, npy ndarrayで表現された画像データ)のタプル '''
        return self.areader.iterator()

    def datanumber(self):
        ''' 元データのデータ件数を返す。 '''
        return self.areader.datanumber()

class TypeReader:
    ''' 各データ形式に沿ったデータの読み出しを担当するインターフェースを規定する基底クラス '''

    def __init__(self, srcdatatype):
        ''' TypeReader イニシャライザ
        srcdatatype: DataType Enumのメンバ '''
        self.datatype = srcdatatype
        return

    def readtonpy(self, storetype, srcpath, srcname):
        ''' 適切なread関数を選び、データを読み出してndarrayの形式で返す '''
        actual_readfunc = storetype.readerfunc(self.datatype.reader())
        return actual_readfunc(srcpath, srcname)

    def read_from_rawfile(self, srcpath, filename):
        ''' 生のディレクトリ内からデータを読み出し、ndarray形式で返す。
        srcpath: ソースディレクトリのパス
        filename: ファイル名
        srcpath, filenameはos.path.joinで結合するので、srcpathにファイル名まで記述して、filenameが空でも
        よい。zip, tarとの引数の数をそろえるため2引数関数としている。 '''
        raise NotImplementedError

    def read_from_zip(self, zipfile, zipname):
        ''' ZIPアーカイブからデータを読み出し、ndarray形式で返す。
        zipfile: openされたzipfileオブジェクト
        zipname: zipアーカイブ内のメンバ名 '''
        data = zipfile.read(zipname)
        bytesio = BytesIO(data)
        return read_from_bytes(bytesio)

    def read_from_tar(self, tarfile, tarname):
        ''' Tarアーカイブからデータを読み出し、ndarray形式で返す。
        tarfile: openされたtarfileオブジェクト
        tarname: tarアーカイブ内のメンバ名またはTarInfoオブジェクト '''
        stream = tarfile.extractfile(tarname) # stereamはBufferedReader
        stream.seek(0)
        buffer = stream.read()
        return read_from_bytes(BytesIO(buffer))
        
    def read_from_bytes(self, bytesio):
        ''' インメモリに読み込まれたBytesIOオブジェクトからデータを読み出し、ndarray形式で返す。 '''
        raise NotImplementedError
    

class PILReader(TypeReader):
    ''' PILで読み込める形式のTypeReaderの中間的実装 '''

    def __init__(self, datatype):
        super(PILReader, self).__init__(datatype)

    def read_from_rawfile(self, srcpath, filename):
        ''' 生のディレクトリ内からデータを読み出し、ndarray形式で返す。
        srcpath: ソースディレクトリのパス
        filename: ファイル名
        srcpath, filenameはos.path.joinで結合するので、srcpathにファイル名まで記述して、filenameが空でも
        よい。zip, tarとの引数の数をそろえるため2引数関数としている。 '''
        
        # 読み込むファイルのフルパス
        fullpath = os.path.join(srcpath, filename)
        # Image化
        img = Image.open(fullpath)
        return np.asarray(img)

    def read_from_bytes(self, bytesio):
        ''' インメモリに読み込まれたBytesIOオブジェクトからデータを読み出し、ndarray形式で返す。 '''
        img = Image.open(bytesio)
        return np.asarray(img)


class PNGReader(PILReader):
    ''' PNGデータを読み込むリーダー '''

    def __init__(self):
        from .datatype import DataType
        super(PNGReader, self).__init__(DataType.PNG)

class JPGReader(PILReader):

    def __init__(self):
        from .datatype import DataType
        super(JPGReader, self).__init__(DataType.JPG)
        return

class DICOMReader(TypeReader):

    def __init__(self):
        from .datatype import DataType
        super(DICOMReader, self).__init__(DataType.DICOM)

    def read_from_rawfile(self, srcpath, filename):
        ''' 生のディレクトリ内からデータを読み出し、ndarray形式で返す。
        srcpath: ソースディレクトリのパス
        filename: ファイル名
        srcpath, filenameはos.path.joinで結合するので、srcpathにファイル名まで記述して、filenameが空でも
        よい。zip, tarとの引数の数をそろえるため2引数関数としている。 '''
        fullpath = os.path.join(srcpath, filename)
        return dcm2npy(pydicom.dcmread(fullpath))

    def read_from_bytes(self, bytesio):
        ''' インメモリに読み込まれたBytesIOオブジェクトからデータを読み出し、ndarray形式で返す。 '''
        return dcm2npy(pydicom.dcmread(bytesio))

    def dcm2npy(self, dcmdataset, dtype=np.float32):
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
        
        
class NPYReader(TypeReader):

    def __init__(self):
        from .datatype import DataType
        super(NPYReader, self).__init__(DataType.NPY)

    def read_from_rawfile(self, srcpath, filename):
        ''' 生のディレクトリ内からデータを読み出し、ndarray形式で返す。
        srcpath: ソースディレクトリのパス
        filename: ファイル名
        srcpath, filenameはos.path.joinで結合するので、srcpathにファイル名まで記述して、filenameが空でも
        よい。zip, tarとの引数の数をそろえるため2引数関数としている。 '''
        fullpath = os.path.join(srcpath, filename)
        return np.load(fullpath)

    def read_from_bytes(self, bytesio):
        ''' インメモリに読み込まれたBytesIOオブジェクトからデータを読み出し、ndarray形式で返す。 '''
        return np.load(bytesio)

class ArchiveReader:
    ''' 格納形式それぞれからデータを読み出すインターフェースを規定する基底クラス '''

    def __init__(self, srcdatatype, srcpath):
        ''' イニシャライザ
        srcdatatype: データのフォーマット
        srcpath: データ格納元のパス
        '''
        self.datatype = srcdatatype
        self.srcpath = srcpath
        self.typereader = self.datatype.reader()
        return

    def __del__(self):
        ''' デストラクタ '''
        self.close_src()

    def open_src(self):
        ''' アーカイブソースをオープンする デフォルトではなにもしない '''
        return

    def close_src(self):
        ''' アーカイブソースをクローズする デフォルトではなにもしない '''

    def iterator(self):
        ''' 個々のデータを読み出してゆくイテレータを返す
        返り値は(データの名前, ndarray)のタプル
        '''
        self.open_src()
        arclist = self.arclist()
        for i in arclist:
            storetype = self.storetype()
            arcname = self.arcname()
            yield (os.path.splitext(os.path.basename(i))[0], self.typereader.readtonpy(storetype, arcname, i))
        self.close_src()

    def storetype(self):
        ''' このArchiveReaderに対応するStoreTypeを返す 具象クラスで定義する '''
        raise NotImplementedError

    def arcname(self):
        ''' このArchiveReaderのアーカイブ名を返す デフォルトはself.srcpathをそのまま返す'''
        return self.srcpath
        
    def datanumber(self):
        ''' 格納されているデータ件数を返す デフォルトでは一旦arclistを呼び出して数える '''
        count = 0
        for i in self.arclist():
            count += 1
        return count

    def arclist(self):
        ''' 格納されているアーカイブメンバのうち、datatypeにマッチするもののイテレータを返す '''
        ext = '.' + self.datatype.getext()
        upperext = ext.upper()
        namelist = self.alllist()
        smalliter = self._iterext(ext, namelist)
        largeiter = self._iterext(upperext, namelist)
        mergeiter = itertools.chain(smalliter, largeiter)
        return mergeiter
        
    @staticmethod
    def _iterext(ext, namelist):
        ''' namelist内にextを含む集合を返すイテレータを生成するジェネレータ関数 '''
        for i in namelist:
            if ext in i:
                yield i

    def alllist(self):
        ''' 格納されているメンバすべてを返す内部処理用の関数 実際には具象クラスで定義する
        返すのはリストでもイテレータでもよい。
        arclistがオーバーライドされていればalllistは定義しなくともよい '''
        raise NotImplementedError

class DirReader(ArchiveReader):
    ''' ディレクトリ内に生ファイルが格納されている形式のデータを読み出すArchiveReader '''

    def __init__(self, srcdatatype, srcpath):
        super(DirReader, self).__init__(srcdatatype, srcpath)

    def storetype(self):
        ''' このArchiveReaderに対応するStoreTypeを返す '''
        from .datatype import StoreType
        return StoreType.DIR

    def alllist(self):
        ''' このアーカイブ内のすべての要素を返す '''
        globstr = os.path.join(self.srcpath, '**/*')
        globresult = glob.iglob(globstr, recursive=True)
        for i in globresult:
            yield os.path.relpath(i, start=self.srcpath)
        
class ZipReader(ArchiveReader):
    ''' zipアーカイブされたデータを読み込むリーダー '''

    def __init__(self, srcdatatype, srcpath):
        super(ZipReader, self).__init__(srcdatatype, srcpath)
        self.zfp = None

    def open_src(self):
        ''' アーカイブソースをオープンする ZipReaderではZipFileをオープンする '''
        if self.zfp is None:
            self.zfp = ZipFile.open(self.srcpath, mode='r')
        return

    def close_src(self):
        ''' アーカイブソースをクローズする ZipReaderではZipFileをクローズする '''
        if self.zfp is not None:
            self.zfp.close()
            self.zfp = None
        return

    def storetype(self):
        ''' このArchiveReaderに対応するStoreTypeを返す '''
        from .datatype import StoreType
        return StoreType.ZIP

    def alllist(self):
        ''' このアーカイブ内のすべての要素を返す '''
        if self.zfp is None:
            self.open_src()
        return self.zfp.namelist()

class TarReader(ArchiveReader):
    ''' tarアーカイブされたデータを読み込むリーダー '''

    def __init__(self, srcdatatype, srcpath):
        super(TarReader, self).__init__(srcdatatype, srcpath)
        self.tfp = None
        return

    def open_src(self):
        ''' アーカイブソースをオープンする TarReaderではTarFileをオープンする '''
        if self.tfp is None:
            self.tfp = TarFile.open(name=self.srcpath, mode='r')

    def close_src(self):
        ''' アーカイブソースをクローズする TarReaderではTarFileをクローズする '''
        if self.tfp is not None:
            self.tfp.close()
            self.tfp = None

    def storetype(self):
        ''' このArchiveReaderに対応するStoreTypeを返す '''
        from .datatype import StoreType
        return StoreType.TAR

    def alllist(self):
        ''' このアーカイブ内のすべてのメンバをリストアップする '''
        if self.tfp is None:
            self.open_src()
        return self.tfp.getnames()
