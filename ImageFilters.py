from MatrixConvolution import convolve

def apply_filter(rgb_image, filter_type):
    """
    Apply a filter to the given image.

    Parameters:
    ------------
    image (list of lists of a list of int): The input image to be filtered.
    filter_type (str): The type of filter to apply. Can be 'sharpen', 'blur', or 'edge'.

    Returns:
    ------------
    list of lists of a list of int: The filtered image.
    """
    # Check if the image is empty
    if not rgb_image or not rgb_image[0] or not rgb_image[0][0]:
        return []

    image_layers_num = len(rgb_image[0][0])

    # Check if the image is grayscale or colored
    if image_layers_num == 1:
        # If the image is grayscale, we can apply the filter directly
        single_layer_image = [[pixel[0] for pixel in row] for row in rgb_image]
        return apply_filter_to_single_layer(single_layer_image, filter_type)

    else:
        # If the image is colored, we need to apply the filter to each layer separately
        filtered_image = []
        for layer in range(image_layers_num):
            single_layer_image = [[rgb_image[i][j][layer] for j in range(len(rgb_image[0]))] for i in range(len(rgb_image))]
            filtered_layer = apply_filter_to_single_layer(single_layer_image, filter_type)
            filtered_image.append(filtered_layer)

        # Combine the filtered layers back into a single image
        combined_filtered_image = [[[filtered_image[layer][i][j] for layer in range(image_layers_num)]
                                     for j in range(len(filtered_image[0]))]
                                    for i in range(len(filtered_image[0][0]))]

        return combined_filtered_image


def apply_filter_to_single_layer(single_layer_image, filter_type):
    """
    Apply a filter to a single layer of the image.

    Parameters:
    ------------
    single_layer_image (list of lists of int): The input image to be filtered.
    filter_type (str): The type of filter to apply. Can be 'sharpen', 'blur', or 'edge'.

    Returns:
    ------------
    list of lists of int: The filtered image.
    """
    if filter_type == 'sharpen':
        return sharpen_image(single_layer_image)
    elif filter_type == 'blur':
        return blur_image(single_layer_image)
    elif filter_type == 'edge':
        return edge_detection(single_layer_image)
    else:
        raise ValueError("Invalid filter type. Choose from 'sharpen', 'blur', or 'edge'.")


def sharpen_image(single_layer_image):
    """
    Apply a sharpening filter to the given image.

    Parameters:
    ------------
    image (list of lists of a list of int): The input image to be sharpened.

    Returns:
    ------------
    list of lists of a list of int: The sharpened image.
    """
    # Define the sharpening kernel
    kernel = [[0, -1, 0],
              [-1, 5, -1],
              [0, -1, 0]]

    # Convolve the image with the kernel
    return convolve(single_layer_image, kernel)


def blur_image(single_layer_image):
    """
    Apply a blurring filter to the given image.

    Parameters:
    ------------
    image (list of lists of a list of int): The input image to be blurred.

    Returns:
    ------------
    list of lists of a list of int: The blurred image.
    """
    # Define the blurring kernel
    kernel = [[1/9, 1/9, 1/9],
              [1/9, 1/9, 1/9],
              [1/9, 1/9, 1/9]]

    # Convolve the image with the kernel
    return convolve(single_layer_image, kernel)

def edge_detection(single_layer_image):
    """
    Apply an edge detection filter using Sobel operator to the given image.

    Parameters:
    ------------
    image (list of lists of a list of int): The input image to be edge detected.

    Returns:
    ------------
    list of lists of a list of int: The edge detected image.
    """
    # Define the Sobel kernels
    kernel_x = [[-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]]

    kernel_y = [[1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]]

    # Convolve the image with the Sobel kernels
    gradient_x = convolve(single_layer_image, kernel_x)
    gradient_y = convolve(single_layer_image, kernel_y)

    # Calculate the magnitude of the gradients
    edge_image = [[round((gradient_x[i][j]**2 + gradient_y[i][j]**2)**0.5) for j in range(len(gradient_x[0]))]
                  for i in range(len(gradient_x))]

    return edge_image