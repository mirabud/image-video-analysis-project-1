from scipy.spatial import KDTree
import numpy as np

def calculate_metrics_and_mse(predicted_labels, true_labels, threshold=5):
    """
    Calculates the confusion metrics (TP, FP, FN) and the Mean Squared Error (MSE)
    for the correctly labeled points using KDTree for efficient matching.

    Parameters:
        predicted_labels (list of dict): List of predictions with 'label_x' and 'label_y' keys.
        true_labels (list of dict): List of true labels with 'label_x' and 'label_y' keys.
        threshold (float): The maximum Euclidean distance to consider a prediction as correct (True Positive).

    Returns:
        tuple: (TP, FP, FN, average_mse)
            - TP: True Positives (correct predictions)
            - FP: False Positives (incorrect predictions)
            - FN: False Negatives (missed ground-truth labels)
            - average_mse: The average Mean Squared Error of the correctly labeled points.
    """
    
    # Handle edge cases: no labels or no predictions
    if not true_labels or not predicted_labels:
        return 0, 0, 0, 0

    # Convert labels and predictions into NumPy arrays
    true_coords = np.array([[label["label_x"], label["label_y"]] for label in true_labels])
    predicted_coords = np.array([[pred["label_x"], pred["label_y"]] for pred in predicted_labels])

    # Build a KDTree with the predicted coordinates
    tree = KDTree(predicted_coords)

    tp = 0
    fp = 0
    fn = 0
    mse_list = []  # To store MSE for correct predictions
    used_indices = set()  # To track used predictions

    # Evaluate each ground-truth label
    for coord in true_coords:
        # Query the nearest prediction using KDTree
        distance, index = tree.query(coord)

        if distance <= threshold and index not in used_indices:
            # True Positive: Predicted correctly and not used before
            tp += 1
            used_indices.add(index)
            # Calculate the squared error for the matched pair and store it
            mse = np.sum((coord - predicted_coords[index])**2)
            mse_list.append(mse)
        else:
            # False Negative: Ground-truth without matching prediction
            fn += 1

    # False Positives: Predictions that weren't used
    fp = len(predicted_labels) - len(used_indices)

    # Calculate the average MSE for correct predictions (TP)
    average_mse = np.mean(mse_list) if mse_list else 0

    return tp, fp, fn, average_mse

def calculate_precision(tp, fp):
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)

def calculate_recall(tp, fn):
    if tp + fn == 0:
        return 0
    return tp / (tp + fn)

def calculate_f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def calculate_accuracy(tp, fp, fn):
    return tp / (tp + fp + fn)

def calculate_metrics(predicted_labels, imgs_with_labels, img_name, threshold=5):
    """
    Calculates evaluation metrics for object detection predictions.

    Parameters:
    - predicted_labels (list): A list of predicted labels (or bounding boxes) for the given image.
    - imgs_with_labels (dict): A dictionary where keys are image names and values are lists of true labels.
    - img_name (str): The name of the image for which metrics are being calculated.
    - threshold (int, optional): The distance threshold to determine whether a predicted label is a true positive. Default is 5.

    Returns:
    - precision (float): The precision of the predictions (TP / (TP + FP)).
    - recall (float): The recall of the predictions (TP / (TP + FN)).
    - f1_score (float): The F1 score, which is the harmonic mean of precision and recall.
    - accuracy (float): The accuracy of the predictions.
    - mse (float): The mean squared error between predicted and true labels.

    Functionality:
    1. Retrieves the true labels (ground truth) for the specified image (`img_name`) from the `imgs_with_labels` dictionary.
    2. Calls `calculate_metrics_and_mse` to compute:
       - True positives (TP)
       - False positives (FP)
       - False negatives (FN)
       - Mean Squared Error (MSE) between the predicted and true labels.
    3. Prints the TP, FP, FN, and MSE values.
    4. Calculates the following metrics using helper functions:
       - Precision: Measures how many of the predicted positives are true positives.
       - Recall: Measures how many of the true positives were identified by the predictions.
       - F1 Score: Combines precision and recall into a single metric.
       - Accuracy: Measures overall correctness of the predictions.
    5. Returns all the calculated metrics as a tuple.
    """
    true_labels = imgs_with_labels.get(img_name, [])
    tp, fp, fn, mse = calculate_metrics_and_mse(predicted_labels, true_labels, threshold)

    precision = calculate_precision(tp, fp)
    recall = calculate_recall(tp, fn)
    f1_score = calculate_f1_score(precision, recall)
    accuracy = calculate_accuracy(tp, fp, fn)

    return precision, recall, f1_score, accuracy, mse