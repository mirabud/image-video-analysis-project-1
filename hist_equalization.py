import cv2

def hist_equalization(img):
    # Apply Histogram equalization
    equalized = cv2.equalizeHist(img)

    # Get binary threshold 
    _, binary_thresh = cv2.threshold(equalized, 50, 200, cv2.THRESH_BINARY)

    return binary_thresh
