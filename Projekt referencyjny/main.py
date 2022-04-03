from src.gaussian_blur import gaussian_blur
from src.canny import sobel_gradient, non_max_suppression, threshold, hysteresis
from src.hough import hough_cicrle
import cv2
import numpy as np


I = cv2.imread("Input/puszki.jpg")
I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

I_blur = gaussian_blur(I, 5, True)
mag, angle = sobel_gradient(I_blur, True)
Z, max_Z = non_max_suppression(mag, angle)
res = threshold(Z, max_Z)
output = hysteresis(res)
img = hough_cicrle(res, 200, 100, 100, 70)

cv2.imshow("Obraz", output.astype('uint8'))
cv2.imwrite("After Canny.jpg", img.astype('uint8'))
cv2.waitKey(0)
