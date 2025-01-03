"""
This script performs color thresholding on an image using OpenCV. 
It supports multiple color spaces (RGB, HSV, and LUV) and allows users to adjust thresholds 
via trackbars in a GUI window. Detected components are highlighted with bounding boxes, 
and the results are saved to a file.
"""

import cv2
import numpy as np
import os

# Initialize paths and load the image
home_dir = os.path.expanduser("~")
image_path = os.path.join(home_dir, "bazant_ws/.images/saved_image.jpg")
image = cv2.imread(image_path)
if image is None:
    print("Error: Could not load image from path:", image_path)
    exit()

# Set the default color space
current_color_space = "RGB"


def update_binarization(x):
    """
    Updates the binarization output based on trackbar positions.
    Includes color space conversions and thresholding logic.
    """
    global current_color_space, display_image

    lower_red = cv2.getTrackbarPos("Lower Red", "Binarization")
    upper_red = cv2.getTrackbarPos("Upper Red", "Binarization")
    lower_green = cv2.getTrackbarPos("Lower Green", "Binarization")
    upper_green = cv2.getTrackbarPos("Upper Green", "Binarization")
    lower_yellow = cv2.getTrackbarPos("Lower Yellow", "Binarization")
    upper_yellow = cv2.getTrackbarPos("Upper Yellow", "Binarization")

    if current_color_space == "HSV":
        color_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif current_color_space == "LUV":
        color_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
    else:
        color_image = image

    lower_red_bound = np.array([lower_red, 0, 0])
    upper_red_bound = np.array([upper_red, 255, 255])
    lower_green_bound = np.array([0, lower_green, 0])
    upper_green_bound = np.array([255, upper_green, 255])
    lower_yellow_bound = np.array([0, 0, lower_yellow])
    upper_yellow_bound = np.array([255, 255, upper_yellow])

    red_mask = cv2.inRange(color_image, lower_red_bound, upper_red_bound)
    green_mask = cv2.inRange(color_image, lower_green_bound, upper_green_bound)
    yellow_mask = cv2.inRange(color_image, lower_yellow_bound, upper_yellow_bound)

    combined_mask = cv2.bitwise_and(red_mask, green_mask)
    combined_mask = cv2.bitwise_and(combined_mask, yellow_mask)
    combined_mask = cv2.medianBlur(combined_mask, 13)

    binarized_image = cv2.bitwise_and(image, image, mask=combined_mask)
    binarized_image[binarized_image == 0] = 255

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        combined_mask, connectivity=8
    )

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cv2.rectangle(binarized_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    display_image = binarized_image
    cv2.imshow("Binarization", binarized_image)


def switch_color_space():
    """
    Cycles through RGB, HSV, and LUV color spaces.
    """
    global current_color_space
    if current_color_space == "RGB":
        current_color_space = "HSV"
    elif current_color_space == "HSV":
        current_color_space = "LUV"
    else:
        current_color_space = "RGB"


cv2.namedWindow("Binarization")
cv2.createTrackbar("Lower Red", "Binarization", 0, 255, update_binarization)
cv2.createTrackbar("Upper Red", "Binarization", 0, 255, update_binarization)
cv2.createTrackbar("Lower Green", "Binarization", 0, 255, update_binarization)
cv2.createTrackbar("Upper Green", "Binarization", 0, 255, update_binarization)
cv2.createTrackbar("Lower Yellow", "Binarization", 0, 255, update_binarization)
cv2.createTrackbar("Upper Yellow", "Binarization", 0, 255, update_binarization)

update_binarization(0)
print("Press 's' to switch between RGB, HSV, and LUV color spaces.")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        switch_color_space()
        update_binarization(0)
    elif key == 27:
        break

binarized_image_path = os.path.join(home_dir, "bazant_ws/.images/binarized_image.jpg")
cv2.imwrite(binarized_image_path, display_image)
cv2.destroyAllWindows()
