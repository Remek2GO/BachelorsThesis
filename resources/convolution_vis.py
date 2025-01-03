"""
This script performs edge detection on an input image using a convolution operation with a specified kernel. 
It accepts the image path as a command-line argument, applies the convolution, and saves the processed image.
"""

import cv2
import numpy as np
import argparse


def convolve_edges(image_path, convolution_matrix):
    """
    Applies a convolution operation to detect edges in the input image.

    :param image_path: Path to the input image.
    :param convolution_matrix: Convolution matrix (kernel) for edge detection.
    :return: Processed image with detected edges.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to load image from path: {image_path}")

    convolved_image = cv2.filter2D(image, -1, convolution_matrix)
    return convolved_image


def main():
    """
    Parses command-line arguments, processes the image, and saves the result.
    """
    parser = argparse.ArgumentParser(
        description="Image convolution for edge detection."
    )
    parser.add_argument("image_path", help="Path to the input image (e.g., apple.jpg).")
    args = parser.parse_args()

    image_path = args.image_path

    edge_detector = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    try:
        result = convolve_edges(image_path, edge_detector)
        output_path = "edges_result.jpg"
        cv2.imwrite(output_path, result)
        print(f"Processed image saved as '{output_path}'.")
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()
