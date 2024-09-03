import cv2
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, jaccard_score, adjusted_rand_score
import matplotlib.pyplot as plt

def segment(image):
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)  # Apply Gaussian Blur
    clahe = cv2.createCLAHE(clipLimit=9.0, tileGridSize=(1, 1))  # Adjust clipLimit and tileGridSize as needed
    enhanced_image = clahe.apply(blurred_image)  # Apply CLAHE
    thresholded_image = cv2.adaptiveThreshold(enhanced_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 801, 30)  # Apply adaptive thresholding
    segmented_image = cv2.bitwise_and(image, image, mask=thresholded_image)  # Apply mask to original image

    cv2.imshow("Original Image", image)
    cv2.imshow("Enhanced Image", enhanced_image)
    cv2.imshow("Segmented Image", segmented_image)
    cv2.imshow("Thresholded Image", thresholded_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # thresholded_image = cv2.adaptiveThreshold(enhanced_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 601, 65)

    # # Use morphological operations to refine the segmentation
    # kernel = np.ones((5, 5), np.uint8)
    
    # mask = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel, iterations=1)
    # # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # mask = cv2.bitwise_not(mask)

    # mask = mask // 255

    # # Connectivity analysis
    # num_labels, labels = cv2.connectedComponents(mask, connectivity=8)

    # # Optionally, filter out small components
    # min_size = 1300  # Minimum size of components to keep
    # new_mask = np.zeros_like(mask)
    # for label in range(1, num_labels):  # Start from 1 to ignore the background
    #     component = (labels == label)
    #     if np.sum(component) >= min_size:
    #         new_mask[component] = 1
    
    # return new_mask, enhanced_image

def compute_metrics(pred_mask, true_mask):
    true_mask = (true_mask > 0).astype(np.uint8)

    # Flatten the masks to 1D for pixel-wise comparison
    pred_mask_flat = pred_mask.flatten()
    true_mask_flat = true_mask.flatten()
    
    # Adapted Rand Error (ARE)
    are = adjusted_rand_score(true_mask_flat, pred_mask_flat)    
    # Precision and Recall
    precision = precision_score(true_mask_flat, pred_mask_flat, average='binary')
    recall = recall_score(true_mask_flat, pred_mask_flat, average='binary')
    
    # Intersection over Union (IoU)
    iou = jaccard_score(true_mask_flat, pred_mask_flat, average='binary')
    
    return {"ARE": are, "Precision": precision, "Recall": recall, "IoU": iou}

def evaluate_segmentation(image_path, ground_truth_path):
    # Load the original image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Failed to load image from {image_path}")
        return

    # Generate predicted mask using the segmentation function
    pred_mask, enhanced_image = segment(image)
    
    # Load the ground truth mask
    true_mask = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)
    if true_mask is None:
        print(f"Error: Failed to load ground truth mask from {ground_truth_path}")
        return
    
    # Binarize the true mask
    _, true_mask = cv2.threshold(true_mask, 127, 1, cv2.THRESH_BINARY)
    
    # Compute metrics
    metrics = compute_metrics(pred_mask, true_mask)
    
    # Print the results
    print("#### DETAILED RESULTS ####")
    print(f"Image: {os.path.basename(image_path)}")
    print(f"ARE: {metrics['ARE']}")
    print(f"Precision: {metrics['Precision']}")
    print(f"Recall: {metrics['Recall']}")
    print(f"IoU: {metrics['IoU']}")
    
    # Display images and masks
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].set_title("Original Image")
    ax[0].imshow(image, cmap='gray')
    ax[1].set_title("Enhanced Image")
    ax[1].imshow(enhanced_image, cmap='gray')
    ax[2].set_title(f"Predicted Mask (Precision: {metrics['Precision']:.2f})")
    ax[2].imshow(pred_mask, cmap='gray')
    ax[3].set_title("Ground Truth Mask")
    ax[3].imshow(true_mask, cmap='gray')
    plt.show()

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Evaluate segmentation results.")
#     parser.add_argument("-i", "--image_path", required=True, help="Path to the original image.")
#     parser.add_argument("-g", "--ground_truth_path", required=True, help="Path to the ground truth mask.")
#     args = parser.parse_args()

#     evaluate_segmentation(args.image_path, args.ground_truth_path)

segment(cv2.imread('./Dataset/malignant/malignant (1).png', cv2.IMREAD_GRAYSCALE))