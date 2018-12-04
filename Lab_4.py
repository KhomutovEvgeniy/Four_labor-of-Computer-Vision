import cv2
import numpy as np
from matplotlib import pyplot as plt
import cmath
import timeit


image = cv2.imread('figure.jpg', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(image, (150, 150))

"""Медленное прямое дискретное преобразование Фурье (ДПФ) для одномерного массива"""
def DDFTslow1(x):
    x = np.asarray(x, dtype=complex)
    rows = x.shape[0]
    cols = np.arange(rows)
    row = cols.reshape((rows, 1))
    M = np.exp(-2j * np.pi * row * cols / rows)  # Вычисление матрицы преобразования
    return np.dot(M, x)  # Скалярное произведение

"""Медленное прямое дискретное преобразование Фурье для двумерного массива"""
def DDFTslow2(x):
    cols, rows = x.shape[0:2]
    dft = np.zeros((cols, rows), dtype=complex)

    for col in range(cols):  # computing dft for all columns
        xModel = np.asarray(x[col, :], dtype=float)
        cv2.waitKey(500)
        xDFT = DDFTslow1(xModel)
        dft[col, :] = xDFT

    for row in range(rows):     # computing dft for all rows
        yModel = np.asarray(dft[:, row], dtype=complex)
        yDFT = DDFTslow1(yModel)
        dft[:, row] = yDFT
    return dft

"""Медленное обратное дискретное преобразование Фурье для двумерного массива"""
def IDFTslow2(x):
    cols, rows = x.shape[0:2]
    dft = np.zeros((cols, rows), dtype=complex)

    for col in range(cols):  # Вычисление ДПФ по строкам
        xModel = np.asarray(x[col, :], dtype=float)
        cv2.waitKey(500)
        xDFT = DDFTslow1(xModel)
        dft[col, :] = xDFT

    for row in range(rows):  # Вычисление ДПФ по столбцам
        yModel = np.conj(np.asarray(dft[:, row], dtype=complex))
        yDFT = DDFTslow1(yModel)
        dft[:, row] = yDFT / (cols * rows)

    print(dft)
    return dft


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)

    return wrapped

b= np.array([1, 2, 132, 255],
            [4, 100, 35, 2])



wrapped = wrapper(DDFTslow2, b)
print(timeit.timeit(wrapped, number=100))





def DFFT1(x):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]

    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return DDFTslow1(x)
    else:
        X_even = DFFT1(x[::2])
        X_odd = DFFT1(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N / 2] * X_odd,
                               X_even + factor[N / 2:] * X_odd])

""" Direct Fast Fourier Transmision """
def DFFT2(x):
    cols, rows = x.shape[0:2]
    dft = np.zeros((cols, rows), dtype=complex)

    for col in range(cols):  # computing dft for all columns
        xModel = np.asarray(x[col, :], dtype=float)
        cv2.waitKey(500)
        xDFT = DFFT1(xModel)
        dft[col, :] = xDFT

    for row in range(rows):  # computing dft for all rows
        yModel = np.asarray(dft[:, row], dtype=complex)
        yDFT = abs(DFFT1(yModel))
        dft[:, row] = yDFT
    return dft

""" Indirect Fast Fourier Transmision """
def IFFT2(x):
    cols, rows = x.shape[0:2]
    dft = np.zeros((cols, rows), dtype=complex)

    for col in range(cols):  # computing dft for all columns
        xModel = np.asarray(x[col, :], dtype=float)
        cv2.waitKey(500)
        xDFT = DFFT1(xModel)
        dft[col, :] = xDFT

    for row in range(rows):  # computing dft for all rows
        yModel = np.conj(np.asarray(dft[:, row], dtype=complex))
        yDFT = DFFT1(yModel)
        dft[:, row] = yDFT
    return dft


# Получение амплитуд образов Фурье
def amplitudeFourier(x):
    realPart = np.asarray(abs(x))
    amplitudes = realPart.astype(np.int)
    return amplitudes


x = np.random.random(1024)

a1 = np.array([1, 2, ])
a3 = np.array([132, 255, ])
a2 = np.array([1, 132])
b = np.array([[1, 120, 23, 14], [132, 255, 45, 40]])
# example = np.zeros((2, 4), dtype=np.complex)
example = DDFTslow2(b)

amp = amplitudeFourier(example)
print(amp)
# print(np.allclose(DFFT2(b), np.fft.fft2(b)))











# print(np.allclose(DFTslow2(b), np.fft.fft2(b)))


'''
print(np.allclose(np.fft.fft2(b), cv2.dft(np.float32(b))))
'''

