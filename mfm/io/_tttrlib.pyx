import numpy as np
cimport numpy as np
cimport cython
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t


@cython.boundscheck(False)
def beckerMerged(uint8_t[:] b, uint8_t[:] can, uint32_t[:] tac, uint64_t[:] mt, uint64_t[:] event,
                 unsigned long length):
    cdef unsigned int g, i
    cdef unsigned char b0, b1, b2, b3
    cdef unsigned char inv, mtov
    cdef unsigned long ov, ovfl
    ov, g, mtov, inv = 0, 0, 0, 0
    for i in range(1, length):
        b3 = b[4*i+3]
        inv = (b3 & 128) >> 7
        mtov = (b3 & 64) >> 6
        b0, b1, b2 = b[4*i], b[4*i+1], b[4*i+2]
        if (inv==0) and (mtov==0):
            event[g]=g
            tac[g]=((b3 & 0x0F)<<8 | b2)
            ovfl= ov*4096;
            mt[g]= ((b1 & 15)<<8 | b0) + ovfl
            can[g] = ((b1 & 0xF0) >> 4)
            g += 1
        else:
            if (inv==0) and (mtov==1):
                ov += 1
                event[g]=g
                tac[g]=((b3 & 0x0F)<<8 | b2)
                ovfl= ov*4096
                mt[g]= ((b1 & 15)<<8 | b0) + ovfl
                can[g] = ((b1 & 0xF0) >> 4)
                g += 1
            else:
                if (inv==1) and (mtov==1):
                    ov += ((b3 & 15)<<24) | ((b2<<16) | ((b1<<8) | b0))
    return g

@cython.boundscheck(False)
def ht3(uint8_t[:] b, uint8_t[:] can, uint32_t[:] tac, uint64_t[:] mt, uint64_t[:] event, unsigned long length):

    cdef unsigned long long i, number_of_photon

    cdef unsigned char b0=0, b1=0, b2=0, b3=0
    cdef unsigned char inv=0
    cdef long long ovfl = 0

    ov, number_of_photon, inv = 0, 0, 0
    for i in range(length):
        b3=b[4*i+3]
        inv= (b3 & 254) >> 1

        b0=b[4*i]
        b1=b[4*i+1]
        b2=b[4*i+2]

        if inv == 127:
            ov += 1
        else:
            event[number_of_photon]=number_of_photon
            tac[number_of_photon] = ((b3 & 1)<<14 | b2<<6 |(b1 & 252)>>2)
            ovfl = ov * 1024
            mt[number_of_photon]= (((b1 & 3)<<8 |b0 ) + ovfl)
            can[number_of_photon] = ((b3 & 254) >> 1)
            if can[number_of_photon] > 64:
                can[number_of_photon] -= 64
            number_of_photon += 1
    return number_of_photon


@cython.boundscheck(False)
def iss_16(np.ndarray[np.uint16_t, ndim=1] b, uint8_t[:] can, uint32_t[:] tac, uint64_t[:] mt,
           long length, int step, int phMode, int offset):
    """
    Reading of ISS-photon format (FCS-measurements)

    :param b:
    :return:
    """
    cdef unsigned int i, j, k
    cdef unsigned int ch1, ch2
    cdef unsigned int frequency

    if step == 1:
        k = 1 if phMode else 0
        for i in range(offset, b.shape[0]):
            ch1 = b[i]
            if phMode:
                mt[k] = mt[k-1] + ch1
                can[k] = 0
                k += 1
            else:
                for j in range(ch1):
                    mt[k] = i
                    can[k] = 0
                    k += 1
    elif step == 2:
        k = 0
        for i in range(offset, b.shape[0], 2):
            ch1 = b[i]
            ch2 = b[i + 1]
            if phMode:
                mt[k] = mt[k-1] + ch1
                can[k] = 0
                k += 1
                mt[k] = mt[k-1] + ch2
                can[k] = 1
                k += 1
            else:
                for j in range(ch1):
                    mt[k] = i
                    can[k] = 0
                    k += 1
                for j in range(ch2):
                    mt[k] = i
                    can[k] = 1
                    k += 1
    return k


@cython.boundscheck(False)
def iss_32(np.ndarray[np.uint32_t, ndim=1] b, uint8_t[:] can, uint32_t[:] tac, uint64_t[:] mt,
           long length, int step, int phMode, int offset):
    """
    Reading of ISS-photon format (FCS-measurements)

    :param data:
    :return:
    """
    cdef unsigned int i, j, k
    cdef unsigned int ch1, ch2
    cdef unsigned int frequency

    #Data is saved as 0: 16-bit or 1: 32-bit

    if step == 1:
        k = 1 if phMode else 0
        for i in range(offset, length):
            ch1 = b[i]
            if phMode:
                mt[k] = mt[k-1] + ch1
                can[k] = 0
                k += 1
            else:
                for j in range(ch1):
                    mt[k] = i
                    can[k] = 0
                    k += 1
    elif step == 2:
        k = 0
        for i in range(offset, length, 2):
            ch1 = b[i]
            ch2 = b[i + 1]
            if phMode:
                mt[k] = mt[k-1] + ch1
                can[k] = 0
                k += 1
                mt[k] = mt[k-1] + ch2
                can[k] = 1
                k += 1
            else:
                for j in range(ch1):
                    mt[k] = i
                    can[k] = 0
                    k += 1
                for j in range(ch2):
                    mt[k] = i
                    can[k] = 1
                    k += 1
    return k

