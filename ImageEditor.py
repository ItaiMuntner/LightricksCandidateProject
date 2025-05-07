##############################################################################
# FILE: ImageEditor.py
# WRITER: Itai Muntner
# DESCRIPTION: The main file that handles the image editing process. It reads the JSON
#              file, applies the specified operations to the image, and saves the
#              resulting image. The file uses the ImageHelperFunctions, ImageFilters,
#              and ImageAdjustments modules to perform the operations. The file
##############################################################################

##############################################################################
#                                   Imports                                  #
##############################################################################
import ImageHelperFunctions
import ImageAdjustments
import ImageFilters
import ParserJSON
import argparse
import json
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
INPUT_KEY = "input"
OUTPUT_KEY = "output"
DISPLAY_KEY = "display"
OPERATION_KEY = "operations"
TYPE_KEY = "type"
VALUE_KEY = "value"
BRIGHTNESS_KEY = "brightness"
CONTRAST_KEY = "contrast"
SATURATION_KEY = "saturation"
SHARPEN_KEY = "sharpen"
BOX_BLUR_KEY = "box"
EDGE_DETECTION_KEY = "sobel"
HEIGHT_KEY = "height"
WIDTH_KEY = "width"
AMOUNT_KEY = "amount"

##############################################################################
#                                  Functions                                 #
##############################################################################

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Parameters:
    ------------
    None

    Returns:
    ------------
    argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Edit an image using JSON config.")
    parser.add_argument("--config", required=True,
                        help="Path to JSON config file that specifies image operations.")
    return parser.parse_args()


def apply_operation(image: Image, operation: dict) -> Image:
    """
    Apply the specified operation to the image.

    Parameters:
    ------------
    image (Image): The input image to be processed.
    operation (dict): The operation to be applied.

    Returns:
    ------------
    Image: The processed image.
    """
    if operation[TYPE_KEY] == BRIGHTNESS_KEY:
        return ImageAdjustments.adjust_image(image, brightness=operation[VALUE_KEY])
    elif operation[TYPE_KEY] == CONTRAST_KEY:
        return ImageAdjustments.adjust_image(image, contrast=operation[VALUE_KEY])
    elif operation[TYPE_KEY] == SATURATION_KEY:
        return ImageAdjustments.adjust_image(image, saturation=operation[VALUE_KEY])
    elif operation[TYPE_KEY] == SHARPEN_KEY:
        return ImageFilters.apply_filter(image, filter_type=SHARPEN_KEY, sharpening_factor=operation[AMOUNT_KEY])
    elif operation[TYPE_KEY] == BOX_BLUR_KEY:
        return ImageFilters.apply_filter(image, filter_type=BOX_BLUR_KEY,
                                         box_blur_height=operation[HEIGHT_KEY], box_blur_width=operation[WIDTH_KEY])
    elif operation[TYPE_KEY] == EDGE_DETECTION_KEY:
        return ImageFilters.apply_filter(image, filter_type=EDGE_DETECTION_KEY)
    else:
        raise ValueError(f"Unknown operation type: {operation[TYPE_KEY]}.")


def main() -> None:
    """
    Main function to handle the image editing process.
    It reads the JSON file, applies the specified operations to the image,
    and saves the resulting image if specified.

    Parameters:
    ------------
    None

    Returns:
    ------------
    None
    """
    # Parse command line arguments and load the JSON configuration file
    args = parse_args()
    config = args.config
    try:
        with open(config, 'r') as json_file:
            config = json.load(json_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {json_file} was not found.")

    # Check the JSON configuration file for validity
    ParserJSON.check_json(config)
    image_data = ParserJSON.extract_image_data(config)

    # Load the image
    try:
        image = ImageHelperFunctions.load_image(image_data[INPUT_KEY])
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {image_data[INPUT_KEY]} was not found.")

    # Apply operations to the image
    for operation in image_data[OPERATION_KEY]:
        image = apply_operation(image, operation)

    if image_data[OUTPUT_KEY]:
        ImageHelperFunctions.save_image(image, image_data[OUTPUT_KEY])

    if image_data[DISPLAY_KEY]:
        ImageHelperFunctions.show_image(image)

if __name__ == "__main__":
    main()