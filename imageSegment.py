import cv2
import numpy as np
from matplotlib import pyplot as plt

def segmentMalignantRegions(image):
    # Processing Image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred_image = cv2.GaussianBlur(image, (3, 3), 1)  # Apply Gaussian Blur
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(2, 2))  # Adjust clipLimit and tileGridSize as needed
    enhanced_image = clahe.apply(blurred_image)  # Apply CLAHE

    _, thresholded_image = cv2.threshold(enhanced_image, 77, 255, cv2.THRESH_BINARY)

    # Use morphological operations to refine the segmentation
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Post-Processing: Removing small regions
    num_labels, labels_im = cv2.connectedComponents(mask)
    min_region_size = 5000  # Increase this value to remove more smaller regions
    for i in range(1, num_labels):
        if np.sum(labels_im == i) < min_region_size:
            mask[labels_im == i] = 0

    # Optional: Smoothing the boundaries of the mask
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    mask = cv2.bitwise_not(mask)


    # Ensure the mask is binary (0 and 1)
    mask = mask // 255

    # Display original, enhanced, and mask images
    plt.figure(figsize=(15, 5))

    # Original image
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')

    # Enhanced image
    plt.subplot(1, 3, 2)
    plt.title('Enhanced Image')
    plt.imshow(enhanced_image, cmap='gray')

    # Mask image
    plt.subplot(1, 3, 3)
    plt.title('Mask Image')
    plt.imshow(mask, cmap='gray')

    plt.show()

# Read the image
image = cv2.imread('./Dataset/malignant/malignant (1).png')
segmentMalignantRegions(image)
