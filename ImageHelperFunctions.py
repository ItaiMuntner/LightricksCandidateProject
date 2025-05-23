##############################################################################
# FILE: ImageHelperFunctions.py
# WRITER: Itai Muntner
# DESCRIPTION: A helper file that handles image loading, saving, and displaying.
#              The PIL library is used to handle the images, as NumPy is not
#              suitable for displaying images. I also used the PIL library to
#              convert the images to lists for easier manipulation.
#              The lists are then converted back to images for saving and displaying.
##############################################################################

##############################################################################
#                                   Imports                                  #
##############################################################################
from PIL import Image as PILImage
from typing import List, Union
import numpy as np

##############################################################################
#                                   Typing                                   #
##############################################################################
SingleChannelImage = List[List[int]]
ColoredImage = List[List[List[int]]]
Image = Union[ColoredImage, SingleChannelImage]


##############################################################################
#                                 CONSTANTS                                  #
##############################################################################
GRAYSCALE_CODE = "L"
RGB_CODE = "RGB"

##############################################################################
#                              Helper Functions                              #
##############################################################################

def load_image(image_filename: str, mode: str = RGB_CODE) -> Image:
    """
    Loads the image stored in the path image_filename and return it as a list
    of lists.

    Parameters
    ------------
    image_filename (string): a path to an image file. If path doesn't exist an
    exception will be thrown.
    mode (string): use GRAYSCALE_CODE = "L" for grayscale images.

    Returns
    ------------
    A multi-dimensional list representing the image in the format
    rows X cols X channels. The list is 2D in case of a grayscale image and 3D
    in case it's colored.
    """
    img = PILImage.open(image_filename)
    # Convert palette-based images with transparency to RGBA
    if img.mode == "P" and "transparency" in img.info:
        img = img.convert("RGBA")
    else:
        img = img.convert(mode)
    image = __lists_from_pil_image(img)
    return image


def show_image(image: Image) -> None:
    """
    Displays an image.

    Parameters
    ------------
    image: an image represented as a multi-dimensional list of the
    format rows X cols X channels.

    Returns
    ------------
    None
    """
    __pil_image_from_lists(image).show()


def save_image(image: Image, filename: str) -> None:
    """
    Converts an image represented as lists to an Image object and saves it as
    an image file at the path specified by filename.

    Parameters
    ------------
    image: an image represented as a multi-dimensional list.
    filename: a path in which to save the image file. If the path is
    incorrect, an exception will be thrown.

    Returns
    ------------
    None
    """
    if not filename.endswith('.png'):
        filename = f'{filename.split(".")[0]}.png'

    __pil_image_from_lists(image).save(filename)


def __lists_from_pil_image(image: PILImage) -> Image:
    """
    Converts an Image object to an image represented as lists.

    Parameters
    ------------
    image: a PIL Image object

    Returns
    ------------
    The same image represented as multi-dimensional list.
    """
    width, height = image.size
    pixels = list(image.getdata())
    pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
    if type(pixels[0][0]) == tuple:
        for i in range(height):
            for j in range(width):
                pixels[i][j] = list(pixels[i][j])
    return pixels

def __pil_image_from_lists(image_as_lists: Image) -> PILImage:
    """
    Creates an Image object out of an image represented as lists.

    Parameters
    ------------
    image_as_lists: an image represented as multi-dimensional list.

    Returns
    ------------
    The same image as a PIL Image object.
    """
    height = len(image_as_lists)
    width = len(image_as_lists[0])
    mode = "RGB" if isinstance(image_as_lists[0][0], (list, tuple)) else "L"  # Determine mode based on pixel format
    im = PILImage.new(mode, (width, height))

    for j in range(height):
        for i in range(width):
            pixel = image_as_lists[j][i]
            if mode == "RGB" and isinstance(pixel, list):
                pixel = tuple(pixel)  # Convert list to tuple for RGB
            elif mode == "L" and isinstance(pixel, (list, tuple)):
                pixel = int(pixel[0])  # Extract single value for grayscale and ensure it's an int
            elif mode == "L" and isinstance(pixel, (np.ndarray, list)):
                pixel = int(pixel[0])  # Handle NumPy arrays or lists for grayscale
            elif mode == "L":
                pixel = int(pixel)  # Ensure grayscale pixel is an int
            im.putpixel((i, j), pixel)

    return im