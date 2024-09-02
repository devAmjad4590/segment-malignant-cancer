import cv2
import numpy as np

def segmentMalignantRegions(image):
    # Processing Image
    blurred_image = cv2.GaussianBlur(image, (3, 3), 1)  # Apply Gaussian Blur to reduce noise and smooth the image
    clahe = cv2.createCLAHE(clipLimit=9.0, tileGridSize=(3, 3))  # Create CLAHE object with specified clip limit and tile grid size
    enhanced_image = clahe.apply(blurred_image)  # Apply CLAHE to enhance contrast of the image

    # Apply adaptive thresholding to the enhanced image
    adaptive_thresh_image = cv2.adaptiveThreshold(
        enhanced_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 797, 30
    )

    # Use morphological operations to refine the segmentation
    kernel = np.ones((12, 12), np.uint8)  # Create a large kernel for morphological operations
    mask = cv2.morphologyEx(adaptive_thresh_image, cv2.MORPH_CLOSE, kernel, iterations=1)  # Close small holes within the objects
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)  # Remove small noise outside the objects
    kernel = np.ones((2, 2), np.uint8)  # Create a smaller kernel for dilation
    mask = cv2.dilate(mask, kernel, iterations=2)  # Dilate the mask to ensure object boundaries are more pronounced

    # Optional: Smoothing the boundaries of the mask
    mask = cv2.GaussianBlur(mask, (13, 13), 1)  # Apply Gaussian Blur to smooth the edges of the segmented regions

    mask = cv2.bitwise_not(mask)  # Invert the mask so that the segmented regions are in white

    # Ensure the mask is binary (0 and 1)
    mask = mask // 255  # Convert the mask to a binary image

    # Connectivity analysis to label connected components
    num_labels, labels = cv2.connectedComponents(mask, connectivity=4)  # Find connected components in the binary mask

    # Optionally, filter out small components
    min_size = 2000  # Minimum size of components to retain
    new_mask = np.zeros_like(mask)  # Create a new mask to store the filtered components
    for label in range(2, num_labels):  # Start from 2 to ignore the background and small components
        component = (labels == label)  # Isolate each component by label
        if np.sum(component) >= min_size:  # Check if the component size meets the minimum size requirement
            new_mask[component] = 1  # Add the component to the new mask if it meets the size criteria

    return new_mask  # Return the final mask with segmented malignant regions
