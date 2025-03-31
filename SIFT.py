import cv2
import numpy as np
import matplotlib.pyplot as plt

def sift_feature_matching(img1_path, img2_path, max_features=None):
    # Read images
    img1 = cv2.imread('images/image1.jpg')
    img2 = cv2.imread('images/image2.jpg')
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create(nfeatures=max_features)
    
    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    # Draw matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, 
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return img_matches, len(keypoints1), len(keypoints2), len(good_matches)

# Test with sample images
img1_path = 'images/building1.jpg'
img2_path = 'images/building2.jpg'

# Default parameters
result_img, kp1_count, kp2_count, good_matches_count = sift_feature_matching(img1_path, img2_path)

plt.figure(figsize=(15, 10))
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.title(f'SIFT Matching: {good_matches_count} matches out of {kp1_count}/{kp2_count} keypoints')
plt.axis('off')
plt.savefig('results/sift_default.png', bbox_inches='tight')
plt.show()

# With reduced max features
result_img_limited, kp1_count_limited, kp2_count_limited, good_matches_count_limited = sift_feature_matching(img1_path, img2_path, max_features=100)

plt.figure(figsize=(15, 10))
plt.imshow(cv2.cvtColor(result_img_limited, cv2.COLOR_BGR2RGB))
plt.title(f'SIFT Matching (Limited Features): {good_matches_count_limited} matches out of {kp1_count_limited}/{kp2_count_limited} keypoints')
plt.axis('off')
plt.savefig('results/sift_limited.png', bbox_inches='tight')
plt.show()