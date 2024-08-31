import cv2
import numpy as np
import os
from evaluateSegmentation import compute_metrics
import matplotlib.pyplot as plt

def segmentMalignantRegions(image):
    if image is None:
        raise ValueError("Image not loaded correctly. Please check the file path and ensure the image exists.")
    
    # Ensure the image is in grayscale
    if len(image.shape) != 2:
        raise ValueError("Input image must be a single-channel grayscale image.")
    
    # Processing Image
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1)  # Apply Gaussian Blur
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(2, 2))  # Adjust clipLimit and tileGridSize as needed
    enhanced_image = clahe.apply(blurred_image)  # Apply CLAHE

    _, thresholded_image = cv2.threshold(enhanced_image, 111, 255, cv2.THRESH_BINARY)
    # Use morphological operations to refine the segmentation
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Optional: Smoothing the boundaries of the mask
    mask = cv2.GaussianBlur(mask, (5, 5), 1)

    mask = cv2.bitwise_not(mask)

    # Ensure the mask is binary (0 and 1)
    mask = mask // 255

    return mask, enhanced_image, thresholded_image

image_index = 3

mask_image, enhanced_image, thresholded_image = segmentMalignantRegions(cv2.imread(f'./Dataset/malignant/malignant ({image_index}).png', cv2.IMREAD_GRAYSCALE))
predict_image = cv2.imread(f'./Dataset/groundtruth/malignant ({image_index})_mask.png', cv2.IMREAD_GRAYSCALE)

metrics = compute_metrics(mask_image, predict_image)
print(metrics)

# Plotting the images
plt.figure(figsize=(20, 5))

plt.subplot(1, 4, 1)
plt.title('Enhanced Image')
plt.imshow(enhanced_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title('Thresholded Image')
plt.imshow(thresholded_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title('Mask Image')
plt.imshow(mask_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title('Predict Image')
plt.imshow(predict_image, cmap='gray')
plt.axis('off')

plt.show()