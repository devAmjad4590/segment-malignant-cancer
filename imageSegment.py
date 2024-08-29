import cv2
import numpy as np
import os

def segmentMalignantRegions(image):
    if image is None:
        raise ValueError("Image not loaded correctly. Please check the file path and ensure the image exists.")
    
    # Ensure the image is in grayscale
    if len(image.shape) != 2:
        raise ValueError("Input image must be a single-channel grayscale image.")
    
    # Processing Image
    blurred_image = cv2.GaussianBlur(image, (3, 3), 1)  # Apply Gaussian Blur
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(2, 2))  # Adjust clipLimit and tileGridSize as needed
    enhanced_image = clahe.apply(blurred_image)  # Apply CLAHE

    # Apply adaptive thresholding
    adaptive_thresh_image = cv2.adaptiveThreshold(
        enhanced_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 797, 30
    )

    # Use morphological operations to refine the segmentation
    kernel = np.ones((12,12), np.uint8)
    mask = cv2.morphologyEx(adaptive_thresh_image, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Optional: Smoothing the boundaries of the mask
    mask = cv2.GaussianBlur(mask, (21, 21), 1)

    mask = cv2.bitwise_not(mask)

    # Ensure the mask is binary (0 and 1)
    mask = mask // 255

    # Connectivity analysis
    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)

    # Optionally, filter out small components
    min_size = 2000  # Minimum size of components to keep
    new_mask = np.zeros_like(mask)
    for label in range(1, num_labels):  # Start from 1 to ignore the background
        component = (labels == label)
        if np.sum(component) >= min_size:
            new_mask[component] = 1

    return new_mask