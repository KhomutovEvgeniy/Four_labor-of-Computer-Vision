import cv2
import numpy as np
from matplotlib import pyplot as plt
import cmath
import time

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

img = cv2.imread('cat_lick.jpg',0)


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
        x_sample = np.asarray(x[col, :], dtype=float)
        x_dft = MFFT1(x_sample)
        Res[col, :] = x_dft

    for row in range(N_row):# then for a rows using transformed columns
        y_sample = np.conj( np.asarray(Res[:, row], dtype=complex) )
        y_dft = MFFT1(y_sample)
        Res[:, row] = y_dft/(N_col*N_row)

    return Res

def fourier_conv(img, kernel):
    rows, cols = img.shape
    k_size = kernel.shape[1]
    mask = np.zeros((rows, cols), np.float32)
    mask[0:k_size, 0:k_size] = kernel

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    f_np_mask = np.fft.fft2(mask)
    fshift_np_mask = np.fft.fftshift(f_np_mask)

    conv = fshift * fshift_np_mask
    conv = np.fft.ifftshift(conv)
    conv_back = np.fft.ifft2(conv)
    conv_back = np.abs(conv_back)

    return conv_back


'''
b= np.array([1,2, 132,255 ])
print("results are equal: ",np.allclose(MDFT1(b), np.fft.fft(b)))
print (MDFT1(b))
print (np.fft.fft(b))
'''


x = np.random.random(1024)
a1=np.array( [1, 2,])
a3=np.array( [132, 255,])
a2=np.array( [1, 132 ])
b= np.array([ [1,120, 23, 14], [132,255,45, 40] ])



print(np.allclose(MIFFT2(b), np.fft.ifft2(b)))
print (MIFFT2(b))
print (np.fft.ifft2(b))

print(np.allclose(MFFT1(x), np.fft.fft(x)) )
#print (DFT2D(b))

img = cv2.imread('cat_lick_square.png',0)

plt.subplot(111),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])


mytime = cv2.TickMeter()
mytime.start()
dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
mytime.stop()
cv2_sec=cv2.TickMeter.getTimeSec(mytime)
cv2_tick=cv2.TickMeter.getTimeTicks(mytime)

mytime = cv2.TickMeter()
mytime.start()
f = MDFT2(img)
fshift = np.fft.fftshift(f)
mdft = 20*np.log(np.abs(fshift))
mytime.stop()
mdft_sec=cv2.TickMeter.getTimeSec(mytime)
mdft_tick=cv2.TickMeter.getTimeTicks(mytime)
mytime.stop()

mytime = cv2.TickMeter()
mytime.start()
ff = MFFT2(img)
fshift = np.fft.fftshift(ff)
mfft = 20*np.log(np.abs(fshift))
mytime.stop()
mfft_sec=cv2.TickMeter.getTimeSec(mytime)
mfft_tick=cv2.TickMeter.getTimeTicks(mytime)


plt.subplot(221),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'),plt.suptitle("time comparison chart for input image with 512x512 pixel size" ),
plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(magnitude_spectrum, cmap = 'gray'), plt.xlabel('cv2.dft: time(sec) %.2f, time(ticks) %.2d' % (cv2_sec, cv2_tick)),
plt.title('cv2.dft (actually a fft)'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(mdft, cmap = 'gray'),  plt.xlabel('MDFT2: time(sec) %.2f,  my time(ticks) %.2d' % (mdft_sec, mdft_tick)),
plt.title('MDFT2'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(mfft, cmap = 'gray'),  plt.xlabel('MFFT2: time(sec): %.2f, time(ticks): %.2d' % (mfft_sec, mfft_tick)),
plt.title('MFFT2'), plt.xticks([]), plt.yticks([])
plt.show()





'''
print( MDFT1(a1))
print (np.fft.fft(a1))

print( MDFT1(a2))
print (np.fft.fft(a2))

print( MDFT1(a3))
print (np.fft.fft(a3))
print(np.allclose(np.fft.fft2(b), cv2.dft(np.float32(b))))
'''

