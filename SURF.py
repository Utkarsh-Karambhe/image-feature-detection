import cv2
import numpy as np
import matplotlib.pyplot as plt

def surf_feature_matching(img1_path, img2_path, max_features=500):
    # Read images
    img1 = cv2.imread('images/image1.jpg')
    img2 = cv2.imread('images/image2.jpg')
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Initialize SURF detector
    surf = cv2.ORB_create(nfeatures=max_features)
    
    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = surf.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = surf.detectAndCompute(gray2, None)
    
    # Use BFMatcher with Hamming distance for SURF descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply ratio test
    good_matches = []
    for pair in matches:
        if len(pair) == 2:  # Handle case where fewer than k matches are returned
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    # Draw matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return img_matches, len(keypoints1), len(keypoints2), len(good_matches)

# Test with sample images
img1_path = 'images/landmark1.jpg'
img2_path = 'images/landmark2.jpg'

# Default features
result_img, kp1_count, kp2_count, good_matches_count = surf_feature_matching(img1_path, img2_path)

plt.figure(figsize=(15, 10))
plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
plt.title(f'SURF Matching: {good_matches_count} matches out of {kp1_count}/{kp2_count} keypoints')
plt.axis('off')
plt.savefig('results/surf_default.png', bbox_inches='tight')
plt.show()

# More features
result_img_high, kp1_count_high, kp2_count_high, good_matches_count_high = surf_feature_matching(img1_path, img2_path, max_features=1000)

plt.figure(figsize=(15, 10))
plt.imshow(cv2.cvtColor(result_img_high, cv2.COLOR_BGR2RGB))
plt.title(f'SURF Matching (More Features): {good_matches_count_high} matches out of {kp1_count_high}/{kp2_count_high} keypoints')
plt.axis('off')
plt.savefig('results/surf_more_features.png', bbox_inches='tight')
plt.show()