import cv2

def laplacian(img):
    laplacian = cv2.Laplacian(img, cv2.CV_64F)

    laplacian = cv2.convertScaleAbs(laplacian)
    _, laplacian_thresh = cv2.threshold(laplacian, 50, 200, cv2.THRESH_BINARY)

    return laplacian_thresh