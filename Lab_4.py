import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import math
import time

# imageGray = cv.imread('figure.jpg', cv.IMREAD_GRAYSCALE)
imageGray = cv.imread('Lenna.png', cv.IMREAD_GRAYSCALE)
# cv.imshow('Lenna', imageGray)
imageGray = cv.resize(imageGray, (512, 512))

sizeX, sizeY = imageGray.shape[0:2]

def getGaussKernel(size, size_y=None):
    """Получение Ядра фильтра Гаусса"""
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    x, y = np.mgrid[-size:size + 1, -size_y:size_y + 1]
    g = np.exp(-(x ** 2 / float(size) + y ** 2 / float(size_y)))
    return g / g.sum()


"""Ядра фильтров"""

# Ядро Собеля по горизонтали
sobelX = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

# Ядро Собеля по вертикали
sobelY = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])

# Ядро Лапласса
laplacian = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]])

# Ядро Гаусса
gaussian = getGaussKernel(5)

# Ядро усредняющего фильтра
boxFilter = np.ones((5, 5))


def timeit(method):
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()

        print('%r %2.2g sec' % (method.__name__, te - ts))
        return result

    return timed


def DDFTslow1(x):
    """Медленное прямое дискретное преобразование Фурье (ДПФ) для одномерного массива"""
    x = np.asarray(x, dtype=complex)
    rows = x.shape[0]
    cols = np.arange(rows)
    row = cols.reshape((rows, 1))
    M = np.exp(-2j * np.pi * row * cols / rows)  # Вычисление матрицы преобразования
    return np.dot(M, x)  # Скалярное произведение


@timeit
def DDFTslow2(x):
    """Медленное прямое дискретное преобразование Фурье для двумерного массива"""
    cols, rows = x.shape[0:2]
    dft = np.zeros((cols, rows), dtype=complex)

    for col in range(cols):  # computing dft for all columns
        xModel = np.asarray(x[col, :], dtype=float)
        xDFT = DDFTslow1(xModel)
        dft[col, :] = xDFT

    for row in range(rows):  # computing dft for all rows
        yModel = np.asarray(dft[:, row], dtype=complex)
        yDFT = DDFTslow1(yModel)
        dft[:, row] = yDFT
    return dft


@timeit
def IDFTslow2(x):
    """Медленное обратное дискретное преобразование Фурье для двумерного массива"""
    cols, rows = x.shape[0:2]
    dft = np.zeros((cols, rows), dtype=complex)

    for col in range(cols):  # Вычисление ДПФ по строкам
        xModel = np.asarray(x[col, :], dtype=float)
        xDFT = DDFTslow1(xModel)
        dft[col, :] = xDFT

    for row in range(rows):  # Вычисление ДПФ по столбцам
        yModel = np.conj(np.asarray(dft[:, row], dtype=complex))
        yDFT = DDFTslow1(yModel)
        dft[:, row] = yDFT / (cols * rows)
    return dft


def DFFT1(x):
    """Быстрое преобразование Фурье по алгоритму Cooley-Tukey дл одномерного массива"""
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]

    if N % 2 is not 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return DDFTslow1(x)
    else:
        xEven = DFFT1(x[::2])
        xOdd = DFFT1(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([xEven + factor[:N / 2] * xOdd,
                               xEven + factor[N / 2:] * xOdd])


@timeit
def DFFT2(x):
    """ Прямое быстрое преобразование Фурье для двумерного массива"""
    cols, rows = x.shape[0:2]
    dft = np.zeros((cols, rows), dtype=complex)

    for col in range(cols):  # Вычисление БДПФ по колонкам
        xModel = np.asarray(x[col, :], dtype=float)
        xDFT = DFFT1(xModel)
        dft[col, :] = xDFT

    for row in range(rows):  # Вычисление БДПФ по строкам
        yModel = np.asarray(dft[:, row], dtype=complex)
        yDFT = DFFT1(yModel)
        dft[:, row] = yDFT
    return dft


@timeit
def IFFT2(x):
    """ Обратное быстрое преобразование Фурье """
    cols, rows = x.shape[0:2]
    dft = np.zeros((cols, rows), dtype=complex)

    for col in range(cols):  # Вычисление БДПФ по колонкам
        xModel = np.asarray(x[col, :], dtype=float)
        xDFT = DFFT1(xModel)
        dft[col, :] = xDFT

    for row in range(rows):  # Вычисление БДПФ по строкам
        yModel = np.conj(np.asarray(dft[:, row], dtype=complex))
        yDFT = DFFT1(yModel)
        dft[:, row] = yDFT / (cols * rows)
    return dft


def magnitudeFourier(x):
    """Получение амплитуд Фурье-образов"""
    realPart = abs(x)
    # print(realPart)
    # magnitudes = realPart.astype(np.uint16)
    return realPart
    # return magnitudes


def convSpectrum(src):
    """Смещение низких частот Фурье изображентя в центр изображения, реплейс квадрантов по диагонали"""
    cx = src.shape[0] // 2
    cy = src.shape[1] // 2

    beautifulSpectrum = np.zeros((src.shape[0], src.shape[1]), dtype=complex)

    q0 = src[0:cx, 0:cy] # Верхний левый квадрант
    q1 = src[cx:, 0:cy]  # Верхний правый квадрант
    q2 = src[0:cx, cy:]  # Нижний левый квадрант
    q3 = src[cx:, cy:]   # Нижний правый квадрант

    beautifulSpectrum[0:cx, 0:cy] = q3
    beautifulSpectrum[cx:, 0:cy] = q2
    beautifulSpectrum[0:cx, cy:] = q1
    beautifulSpectrum[cx:, cy:] = q0

    return beautifulSpectrum


def fourierConvolution(x, kernel):
    """Свертка изображения с ядром с помощью Фурье"""
    rows, cols = x.shape[0:2]
    kernelSize = kernel.shape[1]
    mask = np.zeros((rows, cols), np.float32)
    mask[0:kernelSize, 0:kernelSize] = kernel

    fourierImage = np.fft.fft2(x)

    fourierMask = np.fft.fft2(mask)

    conv = fourierImage * fourierMask
    convBack = np.fft.ifft2(conv)
    convBack = np.abs(convBack)

    convOneChan = np.zeros((rows, cols))
    cv.normalize(convBack, convOneChan, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

    return convOneChan


def toLog(src):
    rows, cols = src.shape[0:2]
    res = np.asarray(src)

    for row in range(rows):
        for col in range(cols):
            res[row, col] = math.log(src[row, col] + 1)

    return res


def freqFilter(src, filter):
    """Фильтр высоких/низких частот, в зависимости от ядра"""
    srcX, srcY = src.shape[0:2]
    filterX, filterY = filter.shape[0:2]

    zeros = np.zeros((srcX, srcY))
    ones = np.ones((srcX, srcY))
    res = np.asarray(src)

    if filter[0, 0] == 0:
        ones[(srcX - filterX) // 2:(srcX + filterX) // 2, (srcY - filterY) // 2:(srcY + filterY) // 2] = filter[:, :]

        res = src * ones
    else:
        zeros[(srcX - filterX) // 2:(srcX + filterX) // 2, (srcY - filterY) // 2:(srcY + filterY) // 2] = filter[:, :]

        res = src * zeros

    return res


def correlation(src, symbol):

    srcMean, srcDev = cv.meanStdDev(src)
    symbolMean, symbolDev = cv.meanStdDev(symbol)

    srcNorm = (src - srcMean) / srcDev
    symbolNorm = (symbol - symbolMean) / symbolDev

    srcRows, srcCols = src.shape
    symbolRows, sumbolCols = symbol.shape

    mask = np.zeros((srcRows, srcCols), np.float32)
    mask[0:symbolRows, 0:sumbolCols] = symbolNorm[:, :]

    symbolFourier = np.fft.fft2(mask)
    fshiftSymbol = np.fft.fftshift(symbolFourier)

    srcFourier = np.fft.fft2(srcNorm)
    fshiftSrc = np.fft.fftshift(srcFourier)

    conv = fshiftSrc*np.conj(fshiftSymbol)

    conv = np.fft.ifftshift(conv)

    convBack = np.fft.ifft2(conv)
    magnitudeSpectrum = toLog(np.abs(convBack))

    res = np.zeros((magnitudeSpectrum.shape[0], magnitudeSpectrum.shape[1]))
    cv.normalize(magnitudeSpectrum, res, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    max = np.max(res)

    res = cv.inRange(res, (max - 0.02), 1)

    return res


"""Проверка на верность"""
b = np.array([[1, 15, 145, 30], [241, 255, 7, 79]])

# print(np.allclose(DDFTslow2(b), np.fft.fft2(b)))
# print(np.allclose(IDFTslow2(b), np.fft.ifft2(b)))
#
# print(np.allclose(DFFT2(b), np.fft.fft2(b)))
# print(np.allclose(IFFT2(b), np.fft.ifft2(b)))

"""Фурье-образы ядер фильтров"""
sobelXfourier = np.fft.fft2(sobelX)
sobelYfourier = np.fft.fft2(sobelY)
laplasianFourier = np.fft.fft2(laplacian)
gaussianFourier = np.fft.fft2(gaussian)
boxFilterFourier = np.fft.fft2(boxFilter)

cv.imshow('Original in Grayscale', imageGray)

imageFourier = np.fft.fft2(imageGray)
imageFourierMagn = magnitudeFourier(imageFourier)

imageFourierMagnLog = toLog(imageFourierMagn)

# Спектры Фурье образов исходного изображения до и после смены местами квадрантов
imageFourierMagnLogNorm = np.zeros((sizeX, sizeY))
cv.normalize(imageFourierMagnLog, imageFourierMagnLogNorm, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
# cv.imshow('image DFT after log and normalize', imageFourierMagnLogNorm)

imageFourierMagnLogNormBeautSpectr = abs(convSpectrum(imageFourierMagnLogNorm))
print(imageFourierMagnLogNormBeautSpectr)
# cv.imshow('image DFT beautiful spectrum', imageFourierMagnLogNormBeautSpectr)

"""п.2"""
"""Свертка изображения с ядрами"""
"""
convWithSobelX = np.zeros((sizeX, sizeY))
convWithSobelX = fourierConvolution(imageGray, sobelX)
cv.imshow('Convolution with SobelX', convWithSobelX)

convWithSobelY = np.zeros((sizeX, sizeY))
convWithSobelY = fourierConvolution(imageGray, sobelY)
cv.imshow('Convolution with SobelY', convWithSobelY)

convWithLaplacian = np.zeros((sizeX, sizeY))
convWithLaplacian = fourierConvolution(imageGray, laplacian)
cv.imshow('Convolution with Laplacian', convWithLaplacian)

convWithGauss = np.zeros((sizeX, sizeY))
convWithGauss = fourierConvolution(imageGray, gaussian)
cv.imshow('Convolution with Gaussian', convWithSobelY)
"""

"""п.3"""
"""
filterHighFreq = np.zeros((sizeX - 100, sizeY - 100))   # Фильтры частот
filterLowFreq = np.ones((80, 80))

imageFourierInv = convSpectrum(imageFourier)    # Смена квадрантов местами в исходном спектре

filterHigh = freqFilter(imageFourierInv, filterHighFreq)   # Обрезание высоких частот
filterLow = freqFilter(imageFourierInv, filterLowFreq)     # Обрезание низких частот

filterHighMagn = magnitudeFourier(filterHigh)
filterLowMagn = magnitudeFourier(filterLow)

filterHighMagnLog = toLog(filterHighMagn)
filterLowMagnLog = toLog(filterLowMagn)

cv.normalize(filterHighMagnLog, filterHighMagnLog, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
cv.normalize(filterLowMagnLog, filterLowMagnLog, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

cv.imshow("filterHighMagnLog", filterHighMagnLog)   # Фильтры частот и вывод магнитуд после фильтрации
cv.imshow("filterLowMagnLog", filterLowMagnLog)

# filterHighMagnLog = convSpectrum(filterHighMagnLog)   # Обратная смена квадрантов местами
# filterLowMagnLog = convSpectrum(filterLowMagnLog)

filterHighBack = np.fft.ifft2(filterHigh)   # Обратное преобразование Фурье
filterLowBack = np.fft.ifft2(filterLow)

filterHighBack = np.abs(filterHighBack)
filterLowBack = np.abs(filterLowBack)

filterHighBackEnd = np.zeros((sizeX, sizeY))
filterLowBackEnd = np.zeros((sizeX, sizeY))

cv.normalize(filterHighBack, filterHighBackEnd, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)  # Вывод обр пробр-ий
cv.normalize(filterLowBack, filterLowBackEnd, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

cv.imshow('Filter High Frequency', filterHighBackEnd)
cv.imshow('Filter Low Frequency', filterLowBackEnd)
"""

"""п.4"""
"""
autoNumb = cv.imread('autoNumber.jpg', cv.IMREAD_GRAYSCALE)
charA = cv.imread('a.jpg', cv.IMREAD_GRAYSCALE)

cv.imshow('Auto numbers', autoNumb)
cv.imshow('Char A', charA)

corr = correlation(autoNumb, charA)
cv.imshow('Correlation', corr)
"""
# convSpectrum(sobelXfourier)
# convSpectrum(sobelYfourier)
# convSpectrum(laplasianFourier)
# convSpectrum(gaussianFourier)
# convSpectrum(boxFilterFourier)

cv.waitKey(0)
cv.destroyAllWindows()

