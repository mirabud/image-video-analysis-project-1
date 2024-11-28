import cv2

def laplacian(img):
    # Apply High-pass filtering
    laplacian = cv2.Laplacian(img, cv2.CV_64F)

    # Normalization
    laplacian = cv2.convertScaleAbs(laplacian)
    # Threshold
    _, laplacian_thresh = cv2.threshold(laplacian, 50, 200, cv2.THRESH_BINARY)

    return laplacian_thresh
