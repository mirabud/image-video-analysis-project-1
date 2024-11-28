import cv2

def hist_equalization(img):
    equalized = cv2.equalizeHist(img)

    _, binary_thresh = cv2.threshold(equalized, 50, 200, cv2.THRESH_BINARY)

    return binary_thresh