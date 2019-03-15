''' pklreader.py: Pickle Readerを記述する '''

from .typereader import TypeReader
import pickle

class PKLReader(TypeReader):
    ''' pickle reader '''

    def __init__(self):
        from .datatype import DataType
        super(PKLReader, self).__init__(DataType.PKL)
        return

    def read(self, bytesio_or_file):
        ''' BytesIOまたはファイルパスを与え、datadictの形式で読み出す '''
        from io import BytesIO
        if isinstance(bytesio_or_file, BytesIO):
            return pickle.load(bytesio_or_file)
        else:
            with open(bytesio_or_file, mode='rb') as f:
                return pickle.load(f)
    
