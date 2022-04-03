import numpy as np
import cv2


def sobel_gradient(input, add_zero_frames=True):
    Kx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    Ky = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

    if add_zero_frames:
        I = np.zeros((input.shape[0]+2, input.shape[1]+2))
        I[1:-1, 1:-1] = input.astype('float64')
    else:
        I = input.astype('float64')

    (YY, XX) = I.shape
    mag = np.zeros(I.shape, dtype='float64')
    angle = np.zeros(I.shape, dtype='float64')

    for y in range(1, YY-1):
        for x in range(1, XX-1):
            Ix = np.sum(I[y - 1:y + 2, x - 1:x + 2] * Kx)
            Iy = np.sum(I[y - 1:y + 2, x - 1:x + 2] * Ky)

            mag[y, x] = np.sqrt(Ix ** 2 + Iy ** 2)
            angle[y, x] = np.arctan2(Iy, Ix)
            angle[y, x] = angle[y, x] * 180. / np.pi

            if angle[y, x] < 0:
                angle[y, x] += 180

    if add_zero_frames:
        mag_out = mag[1:-1, 1:-1]
        angle_out = angle[1:-1, 1:-1]
    else:
        mag_out = mag
        angle_out = angle

    return mag_out, angle_out


def non_max_suppression(mag, angle):
    YY, XX = mag.shape
    Z = np.zeros((YY, XX), dtype=np.int32)
    max_Z = 0
    for y in range(1, YY-1):
        for x in range(1, XX-1):
            q = 255
            r = 255
            if (0 <= angle[y, x] < 22.5) or (157.5 <= angle[y, x] <= 180):
                q = mag[y, x + 1]
                r = mag[y, x - 1]
                # angle 45
            elif (22.5 <= angle[y, x] < 67.5):
                q = mag[y + 1, x - 1]
                r = mag[y - 1, x + 1]
                # angle 90
            elif (67.5 <= angle[y, x] < 112.5):
                q = mag[y + 1, x]
                r = mag[y - 1, x]
                # angle 135
            elif (112.5 <= angle[y, x] < 157.5):
                q = mag[y - 1, x - 1]
                r = mag[y + 1, x + 1]

            if (mag[y, x] >= q) and (mag[y, x] >= r):
                Z[y, x] = mag[y, x]
            else:
                Z[y, x] = 0

            if max_Z < Z[y, x]:
                max_Z = Z[y, x]
    return Z, max_Z


def threshold(mag, max_mag, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    highThreshold = max_mag * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    YY, XX = mag.shape
    res = np.zeros((YY,XX), dtype=np.int32)

    for y in range(YY):
        for x in range(XX):

            grad_mag = mag[y, x]

            if grad_mag < lowThreshold:
                res[y, x] = 0
            elif highThreshold > grad_mag >= lowThreshold:
                res[y, x] = 100
            else:
                res[y, x] = 255

    return res


def hysteresis(res):
    YY, XX = res.shape
    for y in range(1, YY - 1):
        for x in range(1, XX - 1):
            if res[y, x] == 100:
                if ((res[y + 1, x - 1] == 255) or (res[y + 1, x] == 255) or (res[y + 1, x + 1] == 255)
                        or (res[y, x - 1] == 255) or (res[y, x + 1] == 255)
                        or (res[y - 1, x - 1] == 255) or (res[y - 1, x] == 255)
                        or (res[y - 1, x + 1] == 255)):
                    res[y, x] = 255
                else:
                    res[y, x] = 0

    return res