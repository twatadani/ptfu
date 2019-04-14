''' PNGReader.py: TypeReaderの一種であるPNGReaderを記述する。直接的にはPILReaderの子クラスである。 '''

from .pilreader import PILReader

class PNGReader(PILReader):
    ''' PNGデータを読み込むリーダー '''

    def __init__(self):
        from .datatype import DataType
        super(PNGReader, self).__init__(DataType.PNG)
        return

name = 'pngreader'

