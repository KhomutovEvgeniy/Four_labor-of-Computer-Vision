import numpy as np
import cv2
from matplotlib import pyplot as plt



mean_filter = np.ones((3,3))
box_blur = np.ones((5,5))

# creating a guassian filter
x = cv2.getGaussianKernel(5,10)
gaussian = x*x.T

sharpness =np.array([[-1, -1, -1, -1, -1],
                     [-1, -1, -1, -1, -1],
                     [-1, -1, 75, -1, -1],
                     [-1, -1, -1, -1, -1],
                     [-1, -1, -1, -1, -1]])

# different edge detecting filters
# scharr in x-direction
scharr = np.array([[-3, 0, 3],
                   [-10,0,10],
                   [-3, 0, 3]])
# sobel in x direction
sobel_y= np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
# sobel in y direction
sobel_x= np.array([[-1,-2,-1],
                   [0, 0, 0],
                   [1, 2, 1]])
# laplacian
laplacian=np.array([[0, 1, 0],
                    [1,-4, 1],
                    [0, 1, 0]])



img = cv2.imread('cat_lick_square.png',0)
cat_eye = cv2.imread('cat_eye.png', 0)

def MDFT1(x):    #My Discrete Fourier Transform 1-dimensional
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    n = np.arange(N) #column of numbers
    k = n.reshape((N, 1))# let's reshape column into a row
    M = np.exp(-2j * np.pi * k * n / N) # calculating the matrix of transformation using column n and row k
    return np.dot(M, x)  #return matrix multiplicaation




def MDFT2(x):
    N_col = x.shape[0]
    N_row = x.shape[1]
    Res=np.zeros( (N_col,N_row), dtype=complex)

    for col in range(N_col): #calculating dft for all columns
        x_sample = np.asarray(x[col, :], dtype=float)
        x_dft = MDFT1(x_sample)
        Res[col, :] = x_dft

    for row in range(N_row):# then for a rows using transformed columns
        y_sample = np.asarray(Res[:, row], dtype=complex)
        y_dft = MDFT1(y_sample)
        Res[:, row] = y_dft

    return Res


def MIDFT2(x):
    N_col = x.shape[0]
    N_row = x.shape[1]
    Res=np.zeros( (N_col,N_row), dtype=complex)

    for col in range(N_col): #calculating dft for all columns
        x_sample = np.asarray(x[col, :], dtype=float)
        x_dft = MDFT1(x_sample)
        Res[col, :] = x_dft

    for row in range(N_row):# then for a rows using transformed columns
        y_sample = np.conj( np.asarray(Res[:, row], dtype=complex) )
        y_dft = MDFT1(y_sample)
        Res[:, row] = y_dft/(N_col*N_row)

    return Res


def MFFT1(x):
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]

    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return MDFT1(x)
    else:
        X_even = MFFT1(x[::2]) #Slice from start to end with the step two (big steppy:D)
        X_odd = MFFT1(x[1::2]) #Slice from start+1 to end with the step two
        turn = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + turn[:N // 2] * X_odd,
                               X_even + turn[N // 2:] * X_odd]) #concatenate arrays to make sequence^
                                                            # X_even, turn[:N / 2] * X_odd,  X_even, turn[N / 2:] * X_odd]

def MFFT2(x):
    N_col = x.shape[0]
    N_row = x.shape[1]
    Res=np.zeros( (N_col,N_row), dtype=complex)

    for col in range(N_col): #calculating dft for all columns
        x_sample = np.asarray(x[col, :], dtype=float)
        x_dft = MFFT1(x_sample)
        Res[col, :] = x_dft

    for row in range(N_row):# then for a rows using transformed columns
        y_sample = np.asarray(Res[:, row], dtype=complex)
        y_dft = MFFT1(y_sample)
        Res[:, row] = y_dft

    return Res


def MIFFT2(x):
    N_col = x.shape[0]
    N_row = x.shape[1]
    Res=np.zeros( (N_col,N_row), dtype=complex)

    for col in range(N_col): #calculating dft for all columns
        x_sample = np.conj(np.asarray(x[col, :], dtype=complex))
        x_dft = MFFT1(x_sample)
        Res[col, :] = x_dft

    for row in range(N_row):# then for a rows using transformed columns
        y_sample = (np.asarray(Res[:, row], dtype=complex))
        y_dft = MFFT1(y_sample)
        Res[:, row] = y_dft/(N_col*N_row)

    return Res


def fourier_conv(img, kernel):
    rows, cols = img.shape
    k_size = kernel.shape[1]
    mask = np.zeros((rows, cols), np.float32)
    mask[0:k_size, 0:k_size] = kernel

    f = MFFT2(img)
    fshift = np.fft.fftshift(f)

    f_np_mask = MFFT2(mask)
    fshift_np_mask = np.fft.fftshift(f_np_mask)

    conv = fshift * fshift_np_mask
    conv = np.fft.ifftshift(conv)
    conv_back = MIFFT2(conv)
    conv_back = np.abs(conv_back)

    return conv_back

def LPF (img, rect_size):
    rs = rect_size // 2
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    mask[crow - rs:crow + rs, ccol - rs:ccol + rs] = 1

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return  img_back



def HPF (img, rect_size):
    rs= rect_size//2
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    #mask = np.zeros((rows, cols), np.uint8)
    #mask[crow - 10:crow + 10, ccol - 10:ccol + 10] = 1

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    #fshift = fshift * mask
    fshift[crow - rs:crow + rs, ccol - rs:ccol + rs] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return  img_back



def correlation (img, kernel):
    rows, cols = img.shape
    k_size = kernel.shape[1]

    mask = np.zeros((rows, cols), np.float32)
    mask[0:k_size, 0:k_size] = kernel

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    f_np_mask = np.fft.fft2(mask)
    fshift_np_mask = np.fft.fftshift(f_np_mask)

    conv = fshift * fshift_np_mask

    #conv = np.fft.ifftshift(conv)
    magnitude_spectrum = np.abs(conv)
    #magnitude_spectrum = 20 * np.log(np.abs(conv))
    conv_back = np.fft.ifft2(conv)
    conv_back = np.abs(conv_back)

    return magnitude_spectrum



#
f_np = np.fft.fft2(img)
fshift_np = np.fft.fftshift(f_np)
img_back_np = np.fft.ifft2(fshift_np)
img_back_np = np.abs(img_back_np)
#

rows, cols = img.shape
crow, ccol = rows//2 , cols//2     # center

# create a mask first, center square is 1, remaining all zeros
mask = np.zeros((rows, cols), np.float32)
#mask[crow-30:crow+30, ccol-30:ccol+30] = 1
#for i in range(3):
  #  for j in range(3):
   #   mask [i, j] = sobel_x[i,j]
mask[0:3, 0:3] = sobel_y


f_np_mask = np.fft.fft2(mask)
fshift_np_mask = np.fft.fftshift(f_np_mask)


#mas=np.float32(mask)




conv= fshift_np_mask * fshift_np
conv = np.fft.ifftshift(conv)
conv_back = np.fft.ifft2(conv)
conv_back = np.abs(conv_back)

plt.subplot(331),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])

plt.subplot(332),plt.imshow(fourier_conv(img, sharpness), cmap ='gray')
plt.title('Sharpness'), plt.xticks([]), plt.yticks([])

plt.subplot(333),plt.imshow(fourier_conv(img, gaussian), cmap ='gray')
plt.title('Gauss blur'), plt.xticks([]), plt.yticks([])

plt.subplot(334),plt.imshow(fourier_conv(img, sobel_x), cmap ='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])

plt.subplot(335),plt.imshow(fourier_conv(img, sobel_y), cmap ='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.subplot(336),plt.imshow(fourier_conv(img, laplacian), cmap ='gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])

plt.figure()
plt.subplot(121),plt.imshow(LPF(img, 20), cmap ='gray')
plt.title('LPF'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(HPF(img, 20), cmap ='gray')
plt.title('HPF'), plt.xticks([]), plt.yticks([])

'''
plt.figure()
plt.subplot(121),plt.imshow(correlation(img,cat_eye ), cmap ='gray')
plt.title('LPF'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(HPF(img, 20), cmap ='gray')
plt.title('HPF'), plt.xticks([]), plt.yticks([])

#plt.subplot(144),plt.imshow(img_back_np, cmap = 'gray')
#plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
'''
plt.show()