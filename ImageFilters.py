##############################################################################
# FILE: ImageFilters.py
# WRITER: Itai Muntner
# DESCRIPTION: A helper file that applies filters to images
##############################################################################

##############################################################################
#                                   Imports                                  #
##############################################################################
import numpy as np
from typing import List, Union
from MatrixConvolution import convolve

##############################################################################
#                                   Typing                                   #
##############################################################################
SingleChannelImage = List[List[int]]
ColoredImage = List[List[List[int]]]
Image = Union[ColoredImage, SingleChannelImage, np.ndarray]

##############################################################################
#                                 CONSTANTS                                  #
##############################################################################
EDGE_DETECTION_COMMAND = "sobel"
BOX_BLUR_COMMAND = "box"
SHARPEN_COMMAND = "sharpen"
IDENTITY_KERNEL_5X5 = [[0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0]]
BOX_BLUR_KERNEL_5X5 = [[1 / 25 for _ in range(5)] for _ in range(5)]
SOBEL_KERNEL_X = [[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]]
SOBEL_KERNEL_Y = [[1, 2, 1],
                  [0, 0, 0],
                  [-1, -2, -1]]

##############################################################################
#                              Helper Functions                              #
##############################################################################

def apply_filter(rgb_image: Image, filter_type: str,
                 box_blur_height: int=3, box_blur_width: int=3, sharpening_factor: float=1.5) -> Image:
    """
    Apply a filter to the given image.

    Parameters:
    ------------
    image (list of lists of a list of int): The input image to be filtered.
    filter_type (str): The type of filter to apply. Can be 'sharpen', 'box', or 'sobel'.
    box_blur_height (int): The height of the filter kernel. Default is 3.
    box_blur_width (int): The width of the filter kernel. Default is 3.
    sharpening_factor (float): The factor to sharpen the image. Default is 1.5.

    Returns:
    ------------
    list of lists of a list of int: The filtered image.
    """
    # Check if the image is empty
    if rgb_image is None or len(rgb_image) == 0 or rgb_image[0] is None or len(rgb_image[0]) == 0:
        return []

    image_layers_num = len(rgb_image[0][0])

    # Check if the image is grayscale or colored
    if image_layers_num == 1:
        # If the image is grayscale, we can apply the filter directly
        single_layer_image = [[pixel[0] for pixel in row] for row in rgb_image]
        return apply_filter_to_single_layer(single_layer_image, filter_type,
                                            box_blur_height, box_blur_width, sharpening_factor)

    else:
        # If the image is colored, we need to apply the filter to each layer separately
        filtered_image = []
        for layer in range(image_layers_num):
            single_layer_image =\
                [[rgb_image[i][j][layer] for j in range(len(rgb_image[0]))] for i in range(len(rgb_image))]
            filtered_layer =(
                apply_filter_to_single_layer(
                    single_layer_image, filter_type, box_blur_height, box_blur_width, sharpening_factor))
            filtered_image.append(filtered_layer)

        # Combine the filtered layers back into a single image
        combined_filtered_image = [[[filtered_image[layer][i][j] for layer in range(image_layers_num)]
                                     for j in range(len(filtered_image[0][0]))]
                                    for i in range(len(filtered_image[0]))]

        return combined_filtered_image


def apply_filter_to_single_layer(single_layer_image: Image, filter_type: str,
                                 box_blur_height: int=3, box_blur_width: int=3, sharpening_factor: float=1.5) -> Image:
    """
    Apply a filter to a single layer of the image.

    Parameters:
    ------------
    single_layer_image (list of lists of int): The input image to be filtered.
    filter_type (str): The type of filter to apply. Can be 'sharpen', 'box', or 'sobel'.
    box_blur_height (int): The height of the filter kernel. Default is 3.
    box_blur_width (int): The width of the filter kernel. Default is 3.
    sharpening_factor (float): The factor to sharpen the image. Default is 1.5.

    Returns:
    ------------
    list of lists of int: The filtered image.
    """
    if filter_type == SHARPEN_COMMAND:
        return sharpen_image(single_layer_image, sharpening_factor)
    elif filter_type == BOX_BLUR_COMMAND:
        return blur_image(single_layer_image, box_blur_height, box_blur_width)
    elif filter_type == EDGE_DETECTION_COMMAND:
        return edge_detection(single_layer_image)
    else:
        raise ValueError(f"Invalid filter type. {filter_type} is not supported.")


def sharpen_image(single_layer_image: Image, sharpening_factor: float=1.5) -> Image:
    """
    Apply a sharpening filter to the given image.

    Parameters:
    ------------
    image (list of lists of a list of int): The input image to be sharpened.
    sharpening_factor (float): The factor to sharpen the image. Default is 1.5.

    Returns:
    ------------
    list of lists of a list of int: The sharpened image.
    """
    # Calculate the 5x5 sharpening kernel by the formula:
    # sharpen_kernel = identity_kernel + sharpening_factor * (identity_kernel - box_blur_kernel)
    sharpen_kernel =\
            [[IDENTITY_KERNEL_5X5[i][j] + sharpening_factor * (IDENTITY_KERNEL_5X5[i][j] - BOX_BLUR_KERNEL_5X5[i][j])
              for j in range(5)] for i in range(5)]

    # Convolve the image with the sharpening kernel
    return convolve(single_layer_image, sharpen_kernel)



def blur_image(single_layer_image: Image, box_blur_height: int=3, box_blur_width: int=3) -> Image:
    """
    Apply a blurring filter to the given image.

    Parameters:
    ------------
    image (list of lists of a list of int): The input image to be blurred.
    box_blur_height (int): The height of the filter kernel. Default is 3.
    box_blur_width (int): The width of the filter kernel. Default is 3.

    Returns:
    ------------
    list of lists of a list of int: The blurred image.
    """
    # Define the blurring kernel
    kernel = [[1 / (box_blur_height * box_blur_width) for _ in range(box_blur_width)] for _ in range(box_blur_height)]

    # Convolve the image with the kernel
    return convolve(single_layer_image, kernel)

def edge_detection(single_layer_image: Image) -> Image:
    """
    Apply an edge detection filter using Sobel operator to the given image.

    Parameters:
    ------------
    image (list of lists of a list of int): The input image to be edge detected.

    Returns:
    ------------
    list of lists of a list of int: The edge detected image.
    """
    # Convolve the image with the Sobel kernels
    gradient_x = convolve(single_layer_image, SOBEL_KERNEL_X)
    gradient_y = convolve(single_layer_image, SOBEL_KERNEL_Y)

    # Calculate the magnitude of the gradients
    edge_image = [[round((gradient_x[i][j]**2 + gradient_y[i][j]**2)**0.5) for j in range(len(gradient_x[0]))]
                  for i in range(len(gradient_x))]

    return edge_image