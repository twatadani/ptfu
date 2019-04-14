''' TypeReader.py: DataTypeに沿ったデータ読み出しを担当するクラスTypeReaderを記述する '''

class TypeReader:
    ''' 各データ形式に沿ったデータの読み出しを担当するインターフェースを規定する基底クラス '''

    def __init__(self, srcdatatype):
        ''' TypeReader イニシャライザ
        srcdatatype: DataType Enumのメンバ '''
        self.datatype = srcdatatype
        return

    def read(self, bytesio_or_file):
        ''' BytesIOまたはファイルパスを与え、datadictの形式で読み出す。
        実際の動作は具象クラスで定義する。 '''
        raise NotImplementedError

name = 'typereader'

