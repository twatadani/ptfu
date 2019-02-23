''' TypeReader.py: DataTypeに沿ったデータ読み出しを担当するクラスTypeReaderを記述する '''

#from io import BytesIO

class TypeReader:
    ''' 各データ形式に沿ったデータの読み出しを担当するインターフェースを規定する基底クラス '''

    def __init__(self, srcdatatype):
        ''' TypeReader イニシャライザ
        srcdatatype: DataType Enumのメンバ '''
        self.datatype = srcdatatype
        return

    def read(self, bytesio_or_file):
        ''' BytesIOまたはファイルパスを与え、ndarrayの形式で読み出す。
        実際の動作は具象クラスで定義する。 '''
        raise NotImplementedError

    #def readtonpy(self, storetype, srcpath, srcname, arcobj):
        #''' 適切なread関数を選び、データを読み出してndarrayの形式で返す
        #storetype: StoreTypeメンバ
        #srcpath: アーカイブのパス
        #srcname: アーカイブメンバ名
        #arcobj: オープンされてアーカイブオブジェクト(実際に呼び出される関数依存) '''
        #actual_readfunc = storetype.readerfunc(self.datatype.reader())
        #return actual_readfunc(srcpath, srcname, arcobj)

    #def read_from_rawfile(self, srcpath, filename, arcobj):
        #''' 生のディレクトリ内からデータを読み出し、ndarray形式で返す。
        #srcpath: ソースディレクトリのパス
        #filename: ファイル名
        #srcpath, filenameはos.path.joinで結合するので、srcpathにファイル名まで記述して、filenameが空でも
        #よい。zip, tarとの引数の数をそろえるため2引数としている。
        #arcobj: この関数では使用しない '''
        #raise NotImplementedError

    #def read_from_zip(self, srcpath, zipname, zipobj):
        #''' ZIPアーカイブからデータを読み出し、ndarray形式で返す。
        #srcpath: zipアーカイブファイルのパス
        #zipname: zipアーカイブ内のメンバ名
        #zipobj: openされたzipfileオブジェクト '''

        #data = zipobj.read(zipname)
        #bytesio = BytesIO(data)
        #return self.read_from_bytes(bytesio)

    #def read_from_tar(self, srcpath, tarname, tarfile):
        #''' Tarアーカイブからデータを読み出し、ndarray形式で返す。
        #tarfile: openされたtarfileオブジェクト
        #tarname: tarアーカイブ内のメンバ名またはTarInfoオブジェクト '''
        #stream = tarfile.extractfile(tarname) # stereamはBufferedReader
        #stream.seek(0)
        #buffer = stream.read()
        #return self.read_from_bytes(BytesIO(buffer))
        
    #def read_from_bytes(self, bytesio):
        #''' インメモリに読み込まれたBytesIOオブジェクトからデータを読み出し、ndarray形式で返す。 '''
        #raise NotImplementedError
