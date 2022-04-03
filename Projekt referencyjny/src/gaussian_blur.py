import numpy as np
from time import time
import cv2


def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g


def gaussian_blur(input, size, add_zero_frames=True):
    if size not in [3, 5, 7]:
        print("Niepoprawny rozmiar filtra. Zastosowano rozmiar 5x5")
        size = 5

    d = size // 2

    if add_zero_frames:
        I = np.zeros((input.shape[0]+d*2, input.shape[1]+d*2))
        I[d:-d, d:-d] = input.astype('float64')
    else:
        I = input.astype('float64')

    (YY, XX) = I.shape
    I_gauss = I.astype('float64')

    kernel_3x3 = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
    kernel_5x5 = [[1, 4, 6, 4, 1], [4, 16, 24, 16, 4],
                  [6, 24, 36, 24, 6], [4, 16, 24, 16, 4], [1, 4, 6, 4, 1]]
    kernel_7x7 = [[0, 0, 1, 2, 1, 0, 0], [0, 3, 13, 22, 13, 3, 0], [1, 13, 59, 97, 59, 13, 1],
                  [2, 22, 97, 159, 97, 22, 2], [1, 13, 59, 97, 59, 13, 1], [0, 3, 13, 22, 13, 3, 0],
                  [0, 0, 1, 2, 1, 0, 0]]


    if size == 3:
        print("Operacja filtracji Gaussa dla rozmiaru 3x3")
        for y in range(d, YY - d):
            for x in range(d, XX - d):
                I_gauss[y, x] = np.sum(I[y - d:y + d + 1, x - d:x + d + 1] * kernel_3x3) / 16
    elif size == 5:
        print("Operacja filtracji Gaussa dla rozmiaru 5x5")
        for y in range(d, YY - d):
            for x in range(d, XX - d):
                I_gauss[y, x] = np.sum(I[y - d:y + d + 1, x - d:x + d + 1] * kernel_5x5) / 273
    elif size == 7:
        print("Operacja filtracji Gaussa dla rozmiaru 7x7")
        for y in range(d, YY - d):
            for x in range(d, XX - d):
                I_gauss[y, x] = np.sum(I[y - d:y + d + 1, x - d:x + d + 1] * kernel_7x7) / 1003

    if add_zero_frames:
        output = I_gauss[d:-d, d:-d]
    else:
        output = I_gauss

    return output
