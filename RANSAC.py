import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import time

def ransac_outlier_removal(img1_path, img2_path, ransac_threshold=3.0):
    # Read images
    img1 = cv2.imread('images/image1.jpg')
    img2 = cv2.imread('images/image2.jpg')
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
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
    
    # Draw all matches before RANSAC
    img_all_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, 
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Extract location of good matches
    if len(good_matches) >= 4:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography using RANSAC
        start_time = time()
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
        end_time = time()
        
        # Calculate inliers and outliers
        inlier_count = np.sum(mask)
        outlier_count = len(good_matches) - inlier_count
        
        # Select only inlier points
        ransac_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i][0] == 1]
        
        # Draw matches after RANSAC
        img_ransac_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, ransac_matches, None, 
                                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Create warped image using the homography
        h, w = img1.shape[:2]
        warped_img = cv2.warpPerspective(img1, M, (img2.shape[1], img2.shape[0]))
        
        return img_all_matches, img_ransac_matches, warped_img, len(good_matches), inlier_count, end_time - start_time
    else:
        print("Not enough matches found!")
        return None, None, None, 0, 0, 0

# Test with sample images
img1_path = 'images/scene1.jpg'
img2_path = 'images/scene2.jpg'

# Default RANSAC threshold
all_matches, ransac_matches, warped_img, total_matches, inlier_count, ransac_time = ransac_outlier_removal(img1_path, img2_path)

if all_matches is not None:
    # Plot before RANSAC
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(all_matches, cv2.COLOR_BGR2RGB))
    plt.title(f'Before RANSAC: {total_matches} matches')
    plt.axis('off')
    plt.savefig('results/before_ransac.png', bbox_inches='tight')
    plt.show()
    
    # Plot after RANSAC
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(ransac_matches, cv2.COLOR_BGR2RGB))
    plt.title(f'After RANSAC: {inlier_count} inliers, {total_matches - inlier_count} outliers removed')
    plt.axis('off')
    plt.savefig('results/after_ransac.png', bbox_inches='tight')
    plt.show()
    
    # Plot warped image
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB))
    plt.title('Warped Image using Homography')
    plt.axis('off')
    plt.savefig('results/warped_image.png', bbox_inches='tight')
    plt.show()

# Stricter RANSAC threshold
all_matches_strict, ransac_matches_strict, warped_img_strict, total_matches_strict, inlier_count_strict, ransac_time_strict = ransac_outlier_removal(img1_path, img2_path, ransac_threshold=1.0)

if all_matches_strict is not None:
    # Plot after stricter RANSAC
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(ransac_matches_strict, cv2.COLOR_BGR2RGB))
    plt.title(f'After Stricter RANSAC: {inlier_count_strict} inliers, {total_matches_strict - inlier_count_strict} outliers removed')
    plt.axis('off')
    plt.savefig('results/after_strict_ransac.png', bbox_inches='tight')
    plt.show()