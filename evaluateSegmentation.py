import cv2
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, jaccard_score, adjusted_rand_score
from prettytable import PrettyTable
from imageSegment import segmentMalignantRegions  # Import your segmentation function
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def compute_metrics(pred_mask, true_mask):
    """
    Compute evaluation metrics for a single predicted mask.
    
    Parameters:
        pred_mask (numpy.ndarray): The predicted binary mask.
        true_mask (numpy.ndarray): The ground truth binary mask.
    
    Returns:
        dict: A dictionary containing the metrics (ARE, Precision, Recall, IoU).
    """
    # Flatten the masks to 1D for pixel-wise comparison
    pred_mask_flat = pred_mask.flatten()
    true_mask_flat = true_mask.flatten()
    
    # Adapted Rand Error (ARE)
    are = adjusted_rand_score(true_mask_flat, pred_mask_flat)    
    # Precision and Recall
    precision = precision_score(true_mask_flat, pred_mask_flat, zero_division=1)
    recall = recall_score(true_mask_flat, pred_mask_flat)
    
    # Intersection over Union (IoU)
    iou = jaccard_score(true_mask_flat, pred_mask_flat)
    
    return {"ARE": are, "Precision": precision, "Recall": recall, "IoU": iou}

def evaluate_segmentation(images_dir, ground_truth_dir):
    """
    Evaluate the segmentation results against the ground truth.
    
    Parameters:
        images_dir (str): Directory containing the original images.
        ground_truth_dir (str): Directory containing ground truth masks.
    
    Returns:
        None
    """
    image_files = sorted(os.listdir(images_dir))
    ground_truth_files = sorted(os.listdir(ground_truth_dir))
    
    table = PrettyTable()
    table.field_names = ["Image", "ARE", "Precision", "Recall", "IoU"]
    
    metrics_sum = {"ARE": 0, "Precision": 0, "Recall": 0, "IoU": 0}
    total_images = len(image_files)
    
    images = []
    pred_masks = []
    true_masks = []
    
    for i, file_name in enumerate(image_files):
        print(images_dir, file_name)
        # Load the original image in grayscale mode
        image = cv2.imread(os.path.join(images_dir, file_name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error: Failed to load image from {os.path.join(images_dir, file_name)}")
            continue

        # Generate predicted mask using the segmentation function
        pred_mask = segmentMalignantRegions(image)
        
        # Load the ground truth mask
        true_mask = cv2.imread(os.path.join(ground_truth_dir, ground_truth_files[i]), cv2.IMREAD_GRAYSCALE)
        if true_mask is None:
            print(f"Error: Failed to load ground truth mask from {os.path.join(ground_truth_dir, ground_truth_files[i])}")
            continue
        
        # Binarize the true mask
        _, true_mask = cv2.threshold(true_mask, 127, 1, cv2.THRESH_BINARY)
        
        # Compute metrics
        metrics = compute_metrics(pred_mask, true_mask)
        
        # Accumulate metrics
        for key in metrics:
            metrics_sum[key] += metrics[key]
        
        # Add results to the table
        table.add_row([file_name, metrics["ARE"], metrics["Precision"], metrics["Recall"], metrics["IoU"]])
        
        # Store images and masks for later display
        images.append(image)
        pred_masks.append(pred_mask)
        true_masks.append(true_mask)
    
    # Calculate average metrics
    avg_metrics = {key: metrics_sum[key] / total_images for key in metrics_sum}
    table.add_row(["Average", avg_metrics["ARE"], avg_metrics["Precision"], avg_metrics["Recall"], avg_metrics["IoU"]])
    
    # Print the results
    print("#### DETAILED RESULTS ####")
    print(table)
    
    # Display images and masks with navigation
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plt.subplots_adjust(bottom=0.2)
    current_index = [0]

    def display_image(index):
        ax[0].clear()
        ax[1].clear()
        ax[0].set_title(f"Predicted Mask - {image_files[index]}")
        ax[0].imshow(pred_masks[index], cmap='gray')
        ax[1].set_title(f"Ground Truth Mask - {image_files[index]}")
        ax[1].imshow(true_masks[index], cmap='gray')
        plt.draw()

    def next_image(event):
        current_index[0] = (current_index[0] + 1) % total_images
        display_image(current_index[0])

    display_image(current_index[0])

    ax_next = plt.axes([0.8, 0.05, 0.1, 0.075])
    btn_next = Button(ax_next, 'Next')
    btn_next.on_clicked(next_image)

    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate segmentation results.")
    parser.add_argument("-i", "--images_dir", required=True, help="Directory of original images.")
    parser.add_argument("-g", "--ground_truth_dir", required=True, help="Directory of ground truth masks.")
    args = parser.parse_args()

    evaluate_segmentation(args.images_dir, args.ground_truth_dir)