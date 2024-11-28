import cv2

def gaussian_canny(img):
        # Apply Gaussian Blur
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)

        # Apply the detector of edges Canny
        edges = cv2.Canny(img_blur, 50, 200)
        
        return edges
