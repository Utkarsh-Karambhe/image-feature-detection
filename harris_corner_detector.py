import cv2
import numpy as np

img = cv2.imread('images/image3.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray, blockSize=3, ksize=3, k=0.04)

dst = cv2.dilate(dst, None)

img[dst > 0.05 * dst.max()] = [0, 255, 0]

cv2.imshow('Harris Corners', img)
cv2.waitKey(0)
cv2.imwrite('harris_corners.jpg', img)