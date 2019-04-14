''' JPGReader.py: TypeReaderの一種であるJPGReaderを記述する。直接的にはPILReaderの子クラスである。 '''

from .pilreader import PILReader

class JPGReader(PILReader):

    def __init__(self):
        from .datatype import DataType
        super(JPGReader, self).__init__(DataType.JPG)
        return

name = 'jpgreader'

