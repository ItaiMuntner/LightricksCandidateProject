##############################################################################
# FILE: ParserJSON.py
# WRITER: Itai Muntner
# DESCRIPTION: The ParserJSON module is responsible for parsing the JSON input
#              file and extracting the relevant information for image processing.
##############################################################################

##############################################################################
#                                   Imports                                  #
##############################################################################

##############################################################################
#                                   Typing                                   #
##############################################################################


##############################################################################
#                                 CONSTANTS                                  #
##############################################################################
REQUIRED_KEYS = ["input"]
OPTIONAL_KEYS = ["output", "display", "operations"]
OPERATIONS_KEYS = ["type"]
ADJUSTMENTS_TYPES = ["brightness", "contrast", "saturation"]
FILTERS_TYPES = ["sharpen", "box", "sobel"]


##############################################################################
#                              Helper Functions                              #
##############################################################################


def parse_json(json_file: str) -> dict:
    """
    Parses the JSON file and returns the data as a dictionary.

    Parameters
    ------------
    json_file (str): The path to the JSON file.

    Returns
    ------------
    dict: The parsed data from the JSON file.
    """
    import json

    with open(json_file, 'r') as file:
        data = json.load(file)

    # Check if the JSON data is empty
    if not data:
        raise ValueError("The JSON file is empty.")

    # Check if the JSON data contains the required keys
    for key in REQUIRED_KEYS:
        if key not in data:
            raise KeyError(f"The JSON file is missing the required key: {key}")
    # Check if the input key is a string
    if "input" in data and not isinstance(data["input"], str):
        raise ValueError("The 'input' key must be a string.")

    # Check if the JSON data contains at least one of output or display keys
    if "output" not in data and "display" not in data:
        raise KeyError("The JSON file must contain at least one of the keys: 'output' or 'display'.")
    # Check if the output key is a string
    if "output" in data and not isinstance(data["output"], str):
        raise ValueError("The 'output' key must be a string.")
    # Check if the display key is a boolean
    if "display" in data and not isinstance(data["display"], bool):
        raise ValueError("The 'display' key must be a boolean.")

    # Check if the operations' key contains the required keys
    if "operations" in data:
        # Passing over each operation in the operations' list
        for operation in data["operations"]:
            # Check if the operation is a dictionary
            if not isinstance(operation, dict):
                raise ValueError("Each operation in the 'operations' key must be a dictionary.")
            # Check if the operation contains the type key
            if "type" not in operation:
                raise KeyError(f"The operation is missing the required key: 'type'")
            # Check if the operation type is valid
            if operation["type"] not in ADJUSTMENTS_TYPES + FILTERS_TYPES:
                raise ValueError(f"Invalid operation type: {operation['type']}."
                                 f"Must be one of {ADJUSTMENTS_TYPES + FILTERS_TYPES}.")

            if check_operation_values(operation):
                raise ValueError(f"Invalid operation values: {operation}.")

    return data


def check_operation_values(operation):
    """
    Checks if the operation values are valid.

    Parameters
    ------------
    operation (dict): The operation to check.

    Returns
    ------------
    bool: True if the operation values are valid, False otherwise.
    """
    # Check if the operation contains the required values
    if operation["type"] in ADJUSTMENTS_TYPES:
        # Check if the adjustment value is a number
        if "value" not in operation or not isinstance(operation["value"], (int, float)):
            return False
        # Check if the adjustment value is in the valid range
        if operation["type"] == "brightness":

    elif operation["type"] in FILTERS_TYPES:
        if operation["type"] == "box":
            # Check if the box blur size is a positive integer
            if "height" not in operation or not isinstance(operation["height"], int) or operation["height"] <= 0:
                return False
            if "width" not in operation or not isinstance(operation["width"], int) or operation["width"] <= 0:
                return False
            return True
        elif operation["type"] == "sharpen":
            # Check if the sharpening factor is a positive number
            if "factor" not in operation or not isinstance(operation["factor"], (int, float)) or operation[
                "factor"] <= 0:
                return False
            return True
        elif operation["type"] == "sobel":
            if len(operation) != 1:
                return False
            return True
        # If the operation type is not in the list of valid types, return False
        return False
    else:
        # If the operation type is not in the list of valid types, return False
        return False


def extract_image_data(data: dict) -> dict:
    """
    Extracts the image data from the parsed JSON data.

    Parameters
    ------------
    data (dict): The parsed JSON data.

    Returns
    ------------
    dict: The extracted image data.
    """
    image_data = {
        "image": data.get("image"),
        "filters": data.get("filters"),
        "adjustments": data.get("adjustments")
    }

    return image_data