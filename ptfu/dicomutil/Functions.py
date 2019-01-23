''' Functions.py DICOMユーティリティにおけるユーティリティー関数群 '''

import numpy as np

def arithmetic_rshift(x, shift_bits):
    ''' 算術右シフト '''
    if x >= 0:
        return x >> shift_bits
    else:
        return x / (2 ** shift_bits)

_shiftcache = {}

def bitconvert(dcmdata, verbose=False):
    ''' BitsStoredが16でないDICOMデータを16ビットにコンバートする 
    与えられたdcmdataには破壊的操作が行われ、dcmdata自体が返り値になる '''
    from ..logger import get_default_logger
    logger = get_default_logger()
    if dcmdata.BitsStored == 16:
        return dcmdata

    littleendian = dcmdata.is_little_endian

    represent = dcmdata.data_element('PixelRepresentation').value # 1: signed 0: unsigned
    signed = (represent == 1)
    if verbose:
        logger.debug('Shape:' +  str(dcmdata.Rows) +  'x' + str(dcmdata.Columns) + ' ' + str(dcmdata.Rows * dcmdata.Columns) + ' pixels')
        t = 'Signed' if represent == 1 else 'Unsigned'
        logger.debug('PixelRepresentation:' +  str(represent) +  ' (' + str(t) + ')')
        logger.debug('BitsAllocated:' + str(dcmdata.BitsAllocated))
        logger.debug('BitsStored:' +  str(dcmdata.BitsStored))
        logger.debug('HighBit:' + str(dcmdata.HighBit))
        e = 'LittleEndian' if littleendian else 'BigEndian'
        logger.debug('Endian: ' +  e)
        p = dcmdata.PixelData
        logger.debug('PixelData: ' +  str(type(p)) +  '; ' +  str(len(p)))

    shift_bits = dcmdata.BitsAllocated - dcmdata.HighBit - 1
    shape = (dcmdata.Rows, dcmdata.Columns)

    expected = shape[0] * shape[1] * (dcmdata.BitsAllocated // 8)
    pixeldata = dcmdata.PixelData
    if(len(pixeldata) > expected):
        if verbose:
            logger.debug('PixelDataの長さが本来よりも長すぎるので、truncateします。 expected = ' +  str(expected) +  ', actual = ' + str(len(pixeldata)))
        dcmdata.PixelData = pixeldata[0:expected] # ok

    if shift_bits > 0:
        if littleendian:
            if signed: # signed little endian
                key = (shift_bits, shape)
                if key in _shiftcache:
                    shift = _shiftcache[key]
                else:
                    if verbose:
                        logger.debug('shift cacheにないシフトが検出されました: ' +  str(key))
                    shift = np.full(shape, shift_bits, dtype=np.int16)
                    _shiftcache[key] = shift
                lefted = np.left_shift(dcmdata.pixel_array, shift)
                shifted = np.frompyfunc(arithmetic_rshift, 2, 1)(lefted, shift_bits)
                if shifted.dtype != np.int16:
                    shifted = np.int16(shifted)
                dcmdata.PixelData = shifted.tobytes()
                dcmdata.BitsStored = 16
                dcmdata.HighBit = 15
            else: #unsigned little endian
                pixarray = dcmdata.pixel_array
                if pixarray.dtype != np.uint16:
                    pixarray = np.uint16(pixarray)
                dcmdata.BitsStored = 16
                dcmdata.HighBit = 15
                dcmdata.PixelData = pixarray.tobytes()
        else: # bigendian
            if signed: # signed big endian
                raise NotImplementedError
            else: # unsigned big endian
                pixelarray = dcmdata.pixel_array
                #if pixelarray.dtype != np.uint16:
                #    pixelarray = np.uint16(pixelarray)
                dcmdata.BitsStored = 16
                dcmdata.HighBits = 15
                dcmdata.PixelData = pixelarray.tobytes()
    return dcmdata
