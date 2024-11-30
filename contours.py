import cv2

def find_contours(img, min_area=10, max_area=100):
    """
    Finds and filters contours in a grayscale image based on their area.
    Divides the image into three vertical sections and adjusts the coordinates 
    of contours from the second and third sections to match their original position.

    Parameters:
    - img (numpy.ndarray): Grayscale image where contours will be detected.
    - min_area (int, optional): Minimum area of the contours to be considered valid (default is 10).
    - max_area (int, optional): Maximum area of the contours to be considered valid (default is 100).

    Returns:
    - filtered_contours (list): A list of filtered contours, where each contour is a numpy array of points.
      Only contours with an area in the specified ranges are included.
    """
    
    # Create a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Apply dilation followed by erosion (morphological closing)
    morph_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    morph_img = img


    # Depending of the X coordenate, we will take bigger or smaller areas.
    x1 = 580
    x2 = 650

    contours, _ = cv2.findContours(morph_img[:x1][:], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = [cnt for cnt in contours if min_area <= cv2.contourArea(cnt) <= max_area]

    contours, _ = cv2.findContours(morph_img[x1:x2][:], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        for point in cnt:
            point[0][1] += x1


    filtered_contours_aux = [cnt for cnt in contours if min_area*15 <= cv2.contourArea(cnt) <= max_area]

    filtered_contours.extend(filtered_contours_aux)


    contours, _ = cv2.findContours(morph_img[x2:][:], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        for point in cnt:
            point[0][1] += x2
    
    filtered_contours_aux = [cnt for cnt in contours if min_area*30 <= cv2.contourArea(cnt) <= max_area]

    filtered_contours.extend(filtered_contours_aux)
    return filtered_contours

def count_contours(contours, area_threshold=20):
    """
    Counts the number of contours with an area above a given threshold and
    calculates the center of mass (centroid) for each contour. It also creates 
    a label for each detected contour.

    Parameters:
    - contours (list): List of contours detected in an image.
    - img (numpy.ndarray): The original image where the contours are drawn.
    - area_threshold (int, optional): Minimum area for a contour to be considered valid (default is 10).

    Returns:
    - predicted_labels (list): A list of dictionaries, each containing the label "Person" and the coordinates
      (x, y) of the centroid of the contour.
    - people_count (int): The total number of valid contours (i.e., contours with an area above the threshold).
    """
    predicted_labels = []
    people_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > area_threshold:
            people_count += 1

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        label = {
            "label_name": "Person",
            "label_x": cX,
            "label_y": cY
        }
        predicted_labels.append(label)
    
    return predicted_labels, people_count