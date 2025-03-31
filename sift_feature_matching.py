import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('images/image1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('images/image2.jpg', cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()

# Find keypoints and descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m,n in matches:
    if m.distance < 0.72*n.distance:
        good.append([m])

# Custom parameters for drawing matches
match_thickness = 7  
match_color = (255, 255, 0) 

img_matches = cv2.drawMatchesKnn(
    img1, kp1, 
    img2, kp2, 
    good, 
    None, 
    matchColor=match_color,
    singlePointColor=None,
    matchesMask=None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# Convert to RGB for matplotlib
img_matches_rgb = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 10))
plt.imshow(img_matches_rgb)
plt.title('SIFT Feature Matches')
plt.tight_layout()
plt.show()

cv2.imwrite('sift_matches.jpg', img_matches)