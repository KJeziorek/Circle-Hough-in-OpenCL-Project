import numpy as np
import matplotlib.pyplot as plt


def hough_cicrle(img,N,aN,bN,thr,r_min = 20, r_max = 300):
    # img - eadge image
    # N - number of accumulator for diffrent radius
    # aN - boxes for a
    # bN - boxes for b
    # thr - circle threshold
    # Initialize accumulator
    Y, X = img.shape
    r_max = min(np.sqrt(Y**2 + X**2),r_max)
    circles = []

    for r in np.linspace(r_min, r_max, N):
        H = np.zeros((bN,aN))
        for j in range(0,Y):
            for i in range(0,X):
                if img[j][i] != 0:
                    for theta in np.linspace(0,2*np.pi):
                        a = i - r*np.cos(theta)
                        b = j - r*np.sin(theta)
                        jj = int(max(b//(Y/bN),0))
                        ii = int(max(a//(X/aN),0))
                        if (ii < aN and jj < bN) and (ii != 0 and jj != 0):
                            H[jj][ii] =  H[jj][ii] + 1
        plt.figure()
        plt.imshow(H)
        plt.show()
        # Find values of a and b
        for jj in range(0,bN):
            for ii in range(0,aN):
                if H[jj][ii] > thr:
                    circles.append((ii,jj,r,H[jj][ii]))
    return circles