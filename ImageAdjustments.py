##############################################################################
# FILE: ImageAdjustments.py
# WRITER: Itai Muntner
# DESCRIPTION: A helper file that applies adjustments to images: brightness,
#              contrast, and saturation. The adjustments are done using NumPy.
##############################################################################

##############################################################################
#                                   Imports                                  #
##############################################################################
import numpy as np
from typing import List, Union

##############################################################################
#                                   Typing                                   #
##############################################################################
SingleChannelImage = List[List[int]]
ColoredImage = List[List[List[int]]]
Image = Union[ColoredImage, SingleChannelImage, np.ndarray]


##############################################################################
#                                 CONSTANTS                                  #
##############################################################################
MAX_PIXEL_VALUE = 255
MAX_PIXEL_VALUE_FLOAT = 255.0
MIDDLE_PIXEL_VALUE = 128.0
MIN_PIXEL_VALUE = 0

##############################################################################
#                              Helper Functions                              #
##############################################################################

def adjust_image(image: Image, brightness: float=0.0, contrast: float=0.0, saturation: float=1.0):
    """
    Adjust brightness, contrast, and saturation using NumPy only.

    Parameters:
        image (np.ndarray): RGB image (uint8)
        brightness (float): Additive brightness, range ~[-1.0, 1.0]
        contrast (float): Contrast factor, where 0.0 = no change, -1.0 = gray, 1.0 = double contrast
        saturation (float): 1.0 = no change, 0.0 = grayscale

    Returns:
        np.ndarray: Adjusted RGB image (uint8)
    """
    image = image.astype(np.float32)

    # Adjust brightness
    image += brightness * MAX_PIXEL_VALUE_FLOAT

    # Adjust contrast
    image = (1 + contrast) * (image - MIDDLE_PIXEL_VALUE) + MIDDLE_PIXEL_VALUE

    # Adjust saturation
    hsv = rgb_to_hsv(np.clip(image, MIN_PIXEL_VALUE, MAX_PIXEL_VALUE).astype(np.uint8))
    hsv[..., 1] *= saturation
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 1)
    image = hsv_to_rgb(hsv)

    return image


def rgb_to_hsv(image: Image) -> Image:
    """
    Convert an RGB image to HSV format using NumPy.
    This function was taken from ChatGPT and modified to work with NumPy,
    as I wasn't familiar enough with the HSV color space and how to convert it from RGB.
    I did read it thoroughly and understand it, but I didn't want to risk making a mistake.

    Parameters:
    ------------
    img (np.ndarray): RGB image (uint8)

    Returns:
    ------------
    np.ndarray: HSV image (float32)
    """
    # Ensure the image is in float32 format, normalized to [0, 1]
    image = image.astype(np.float32) / MAX_PIXEL_VALUE_FLOAT

    # Separate the channels into r, g, b
    red, green, blue = image[..., 0], image[..., 1], image[..., 2]

    # Calculate max and min channels for each pixel
    max_channel = np.maximum(np.maximum(red, green), blue)
    min_channel = np.minimum(np.minimum(red, green), blue)

    # Value calculation
    value = max_channel

    # Saturation calculation
    delta = max_channel - min_channel

    # Avoid division by zero
    saturation = np.where(max_channel == 0, 0, delta / max_channel)

    # Hue calculation
    hue = np.zeros_like(max_channel)

    # If delta is zero, hue is undefined (set to 0)
    mask = delta != 0

    # Calculate hue based on which channel is the max
    # Red channel is max
    idx = (max_channel == red) & mask
    hue[idx] = (green[idx] - blue[idx]) / delta[idx]

    # Green channel is max
    idx = (max_channel == green) & mask
    hue[idx] = 2.0 + (blue[idx] - red[idx]) / delta[idx]

    # Blue channel is max
    idx = (max_channel == blue) & mask
    hue[idx] = 4.0 + (red[idx] - green[idx]) / delta[idx]

    # Normalize hue to [0, 1]
    hue = (hue / 6.0) % 1.0
    hue[delta == 0] = 0

    # Stack the channels to form the HSV image
    hsv = np.stack([hue, saturation, value], axis=-1)
    return hsv


def hsv_to_rgb(hsv: Image) -> Image:
    """
    Convert an HSV image to RGB format using NumPy.
    This function was taken from ChatGPT and modified to work with NumPy,
    as I wasn't familiar enough with the HSV color space and how to convert it to RGB.
    I did read it thoroughly and understand it, but I didn't want to risk making a mistake.

    Parameters:
    ------------
    hsv (np.ndarray): HSV image (float32)

    Returns:
    ------------
    np.ndarray: RGB image (uint8)
    """
    # Separate the channels into h, s, v
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Convert hue to sector between 0 and 5 and a fractional part
    hue_sector = np.floor(hue * 6).astype(int)
    hue_fractional = hue * 6 - hue_sector

    # Calculate interpolated values
    p = value * (1 - saturation)
    q = value * (1 - hue_fractional * saturation)
    t = value * (1 - (1 - hue_fractional) * saturation)

    # Ensure hue_sector is in the range [0, 5]
    hue_sector = hue_sector % 6

    # Create conditions for each sector
    conditions = [
        (hue_sector == 0), (hue_sector == 1), (hue_sector == 2),
        (hue_sector == 3), (hue_sector == 4), (hue_sector == 5)
    ]

    # Initialize RGB array
    rgb = np.zeros_like(hsv)

    # Assign RGB values based on hue sector
    rgb[conditions[0]] = np.stack([value, t, p], axis=-1)[conditions[0]]
    rgb[conditions[1]] = np.stack([q, value, p], axis=-1)[conditions[1]]
    rgb[conditions[2]] = np.stack([p, value, t], axis=-1)[conditions[2]]
    rgb[conditions[3]] = np.stack([p, q, value], axis=-1)[conditions[3]]
    rgb[conditions[4]] = np.stack([t, p, value], axis=-1)[conditions[4]]
    rgb[conditions[5]] = np.stack([value, p, q], axis=-1)[conditions[5]]

    # Clip values to [0, 1] and convert to uint8
    return (np.clip(rgb, 0, 1) * MAX_PIXEL_VALUE).astype(np.uint8)
