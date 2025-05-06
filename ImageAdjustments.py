##############################################################################
# FILE: ImageAdjustments.py
# WRITER: Itai Muntner
# DESCRIPTION: A helper file that applies adjustments to images
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
Kernel = Union[List[List[float]], np.ndarray]
Vector = Union[List[float], np.ndarray]


##############################################################################
#                                 CONSTANTS                                  #
##############################################################################
BRIGHTNESS_COMMAND = "brightness"
CONTRAST_COMMAND = "contrast"
SATURATION_COMMAND = "saturation"

##############################################################################
#                              Helper Functions                              #
##############################################################################

def apply_adjustment(rgb_image: Image, adjustment_type: str, adjustment_value: float=1.0) -> Image:
    """
    Apply an adjustment to the given image.

    Parameters:
    ------------
    image (list of lists of a list of int): The input image to be adjusted.
    adjustment_type (str): The type of adjustment to apply. Can be 'brightness', 'contrast', or 'gamma'.

    Returns:
    ------------
    list of lists of a list of int: The adjusted image.
    """
    # Check if the image is empty
    if not rgb_image or not rgb_image[0]:
        return []

    image_layers_num = len(rgb_image[0][0]) if isinstance(rgb_image[0], list) else 1

    # Check if the image is grayscale or colored
    if image_layers_num == 1:
        # If the image is grayscale, we can apply the adjustment directly
        single_layer_image = [[pixel[0] for pixel in row] for row in rgb_image]
        return apply_adjustment_to_single_layer(single_layer_image, adjustment_type, adjustment_value)

    else:
        # If the image is colored, we need to apply the adjustment to each layer separately
        adjusted_image = []
        for layer in range(image_layers_num):
            single_layer_image =\
                [[rgb_image[i][j][layer] for j in range(len(rgb_image[0]))] for i in range(len(rgb_image))]
            adjusted_layer = apply_adjustment_to_single_layer(single_layer_image, adjustment_type, adjustment_value)
            adjusted_image.append(adjusted_layer)

        # Combine the adjusted layers back into a single image
        combined_adjusted_image = [[[adjusted_image[layer][i][j] for layer in range(image_layers_num)]
                                     for j in range(len(adjusted_image[0]))]
                                    for i in range(len(adjusted_image[0][0]))]

        return combined_adjusted_image


def apply_adjustment_to_single_layer(single_layer_image: Image,
                                     adjustment_type: str, adjustment_value: float=1.0) -> Image:
    """
    Apply an adjustment to a single layer of the image.

    Parameters:
    ------------
    single_layer_image (list of lists of int): The input image to be adjusted.
    adjustment_type (str): The type of adjustment to apply. Can be 'brightness', 'contrast', or 'gamma'.

    Returns:
    ------------
    list of lists of int: The adjusted image.
    """
    if adjustment_type == BRIGHTNESS_COMMAND:
        return adjust_brightness(single_layer_image, 1.2)
    elif adjustment_type == CONTRAST_COMMAND:
        return adjust_contrast(single_layer_image, 1.5)
    elif adjustment_type == SATURATION_COMMAND:
        return adjust_saturation(single_layer_image, 1.5)
    else:
        raise ValueError(f"Unknown adjustment type: {adjustment_type}")


def adjust_brightness(single_layer_image, brightness_factor):
    """
    Adjust the brightness of a single layer of the image.

    Parameters:
    ------------
    single_layer_image (list of lists of int): The input image to be adjusted.
    brightness_factor (float): The factor to adjust the brightness by.

    Returns:
    ------------
    list of lists of int: The adjusted image.
    """
    # Check if the image is empty
    if not single_layer_image or not single_layer_image[0]:
        return []

    # Create a new image with the same dimensions
    adjusted_image = np.zeros_like(single_layer_image)

    # Adjust the brightness
    for i in range(len(single_layer_image)):
        for j in range(len(single_layer_image[0])):
            adjusted_image[i][j] = np.clip(single_layer_image[i][j] * brightness_factor, 0, 255)

    return adjusted_image.tolist()
