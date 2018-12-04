import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('cat_lick_square.png',0)
cat_eye = cv2.imread('cat_eye2.png', 0)



def correlation (img, template):
    rows, cols = img.shape
    t_size = template.shape[1]
    t_rows, t_cols = template.shape


    t_sigma = np.sqrt(np.sum(template)/(t_rows*t_cols)) #treating a template
    t_mean = np.mean(template)
    t_norm = np.zeros((t_rows, t_cols), dtype=np.float32)
    for row in range(t_rows):
        for col in range(t_cols):
            t_norm[row, col] = (template[row, col] - t_mean)/t_sigma

    cv2.imwrite('t_norm.png', np.abs(t_norm))
    mask = np.zeros((rows, cols), np.float32)
    mask[0:t_rows, 0:t_cols] = t_norm


    i_sigma = np.sqrt(np.sum(img)/(rows*cols))
    i_mean = np.mean(img)
    i_norm = np.zeros((rows, cols), dtype=np.float32)
    for row in range(rows):
        for col in range(cols):
            i_norm[row, col] = (img[row, col] - i_mean)/i_sigma
    cv2.imwrite('i_norm.png', np.abs(i_norm))

    t_f = np.fft.fft2(mask)
    fshift_t = np.fft.fftshift(t_f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift_t))
    cv2.imwrite('t_fft.png', magnitude_spectrum)

    i_f = np.fft.fft2(i_norm)
    fshift_i = np.fft.fftshift(i_f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift_i))
    cv2.imwrite('i_fft.png', magnitude_spectrum)
    #cv2.imshow('image2', magnitude_spectrum)
    #cv2.imshow('image1', magnitude_spectrum)

    conv = fshift_i*np.conj(fshift_t)

    conv = np.fft.ifftshift(conv)
    magnitude_spectrum = np.abs(conv)
    #magnitude_spectrum = 20 * np.log(np.abs(conv))
    cv2.imwrite('conv.png', magnitude_spectrum)

    conv_back = np.fft.ifft2(conv)
    #conv_back = np.fft.ifftshift(conv_back)
    magnitude_spectrum = 20 * np.log(np.abs(conv_back))
    #magnitude_spectrum = np.abs(conv_back)
    cv2.imwrite('conv_ifft.png', magnitude_spectrum)

    result = cv2.inRange(magnitude_spectrum, 217, 255)
    cv2.imwrite('result.png', result)


    return magnitude_spectrum



plt.figure()
plt.subplot(121),plt.imshow(correlation(img, cat_eye), cmap ='gray')
plt.title('conv ifft'), plt.xticks([]), plt.yticks([])
result = cv2.imread('result.png',0)
plt.subplot(122),plt.imshow(result, cmap ='gray')
plt.title('result'), plt.xticks([]), plt.yticks([])



plt.show()