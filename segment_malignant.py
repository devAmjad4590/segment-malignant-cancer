import cv2
import numpy as np
from matplotlib import pyplot as plt

def segmentMalignantRegions(image):
    # Processing Image
    blurred_image = cv2.GaussianBlur(image, (5, 5), 1)  # Apply Gaussian Blur
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(2, 2))  # Adjust clipLimit and tileGridSize as needed
    enhanced_image = clahe.apply(blurred_image)  # Apply CLAHE

    _, thresholded_image = cv2.threshold(enhanced_image, 55, 255, cv2.THRESH_BINARY)



    # Use morphological operations to refine the segmentation
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

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

# Read the image in grayscale mode
image = cv2.imread('./Dataset//malignant/malignant (26).png', cv2.IMREAD_GRAYSCALE)
segmentMalignantRegions(image)