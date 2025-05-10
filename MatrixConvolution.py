##############################################################################
# FILE: MatrixConvolution.py
# WRITER: Itai Muntner
# DESCRIPTION: A helper file that computes the convolution of a matrix with a kernel
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
Kernel = Union[List[List[float]], np.ndarray]
Vector = Union[List[float], np.ndarray]


##############################################################################
#                                 CONSTANTS                                  #
##############################################################################
KERNEL_SIZE_THRESHOLD = 15

##############################################################################
#                              Helper Functions                              #
##############################################################################

def convolve(matrix: Image, kernel: Kernel) -> Image:
    """
    This function convolves a matrix with a given kernel. The convolution is done with an extended matrix
    approach, where the input matrix is padded with the closest values from the edges.

    Parameters:
    ------------
    matrix (list of lists of int): The input matrix to be convolved.
    kernel (list of lists of float): The kernel to convolve with.

    Returns:
    ------------
    list of lists of int: The convolved matrix.
    """
    # Using numpy, so converting the input lists to numpy arrays
    matrix = np.array(matrix)
    kernel = np.array(kernel)

    # Getting the size of the kernel
    kernel_shape = (kernel.shape[0], kernel.shape[1])

    # Padding the image with the closest values from the edges, so no information is lost
    padded_matrix = np.pad(matrix, ((kernel_shape[0] // 2, kernel_shape[0] // 2),
                                    (kernel_shape[1] // 2, kernel_shape[1] // 2)),
                           mode='edge')

    convolved_matrix = np.zeros_like(padded_matrix)

    # If the kernel is small, the naive approach is a lot faster
    if kernel_shape[0] <= KERNEL_SIZE_THRESHOLD and kernel_shape[1] <= KERNEL_SIZE_THRESHOLD:
        convolved_matrix = naive_convolution(kernel, kernel_shape, matrix, padded_matrix)

    else:
        # Check if the kernel is separable
        if is_approximately_separable(kernel):
            # If the kernel is separable, decompose it into two 1D kernels
            decomposed_kernel1, decomposed_kernel2 = separate_Kernel(kernel)

            # Convolve the matrix with the two 1D kernels
            convolved_matrix = convolve_1D_fft(padded_matrix, decomposed_kernel1, decomposed_kernel2)
        else:
            # If the kernel is not separable, use the 2D convolution
            convolved_matrix = convolve_2D_fft(padded_matrix, kernel)

    # Returning the convolved matrix after trimming it to the original size
    return np.array(convolved_matrix)[kernel_shape[0] // 2: -kernel_shape[0] // 2,
           kernel_shape[1] // 2: -kernel_shape[1] // 2].tolist()


def naive_convolution(kernel: Kernel, kernel_shape: tuple[int, int], matrix: Image, padded_matrix: Image) -> Image:
    """
    This function convolves a matrix with a given kernel using the naive approach.

    Parameters:
    ------------
    kernel (list of lists of float): The kernel to convolve with.
    kernel_shape (tuple): The shape of the kernel.
    matrix (list of lists of int): The input matrix to be convolved.
    padded_matrix (list of lists of int): The padded input matrix.

    Returns:
    ------------
    list of lists of int: The convolved matrix.
    """
    convolved_matrix = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            # Using numpy to do the convolution
            convolved_matrix[i, j] = (
                np.sum(padded_matrix[i:i + kernel_shape[0], j:j + kernel_shape[1]] * kernel).round())
    return convolved_matrix.astype(int).tolist()


def convolve_1D_fft(padded_matrix: Image, decomposed_kernel1: Vector, decomposed_kernel2: Vector) -> Image:
    """
    This function convolves a 2D matrix with two 1D kernels.
    It is done by doing an FFT convolution, which is faster than the naive approach.

    Parameters:
    ------------
    padded_matrix (list of lists of int): The input matrix to be convolved after padding.
    decomposed_kernel1 (list of float): The first 1D kernel to convolve with.
    decomposed_kernel2 (list of float): The second 1D kernel to convolve with.

    Returns:
    ------------
    list of lists of int: The convolved matrix.
    """
    # Using FFT convolution for faster computation in O(n log n) time- useful for large matrices/kernels
    # First convolve the matrix with the first 1D kernel
    convolved_matrix1 = np.fft.ifft(np.fft.fft(padded_matrix, axis=0) * np.fft.fft(decomposed_kernel1)).real

    # Second convolve the result with the second 1D kernel
    convolved_matrix2 = np.fft.ifft(np.fft.fft(convolved_matrix1, axis=1) * np.fft.fft(decomposed_kernel2)).real

    # Finally, we convert the result to a list of lists
    return np.round(convolved_matrix2).astype(int).tolist()


def convolve_2D_fft(padded_matrix: Image, kernel: Kernel) -> Image:
    """
    This function convolves a 2D matrix with a given kernel.
    It is done by doing an FFT convolution, which is faster than the naive approach.

    Parameters:
    ------------
    padded_matrix (list of list of int): The input matrix to be convolved after padding.
    kernel (list of list of int): The kernel to convolve with.

    Returns:
    ------------
    list of list of int: The convolved matrix.
    """
    # Using FFT convolution for faster computation in O(n log n) time- useful for large matrices/kernels
    convolved_matrix = np.fft.ifft2(np.fft.fft2(padded_matrix) * np.fft.fft2(kernel, s=padded_matrix.shape)).real

    # Returning the matrix to being a list of lists and returning it
    return np.round(convolved_matrix).astype(int).tolist()


def is_approximately_separable(kernel: Kernel, threshold: float=0.9) -> bool:
    """
    This function checks if a kernel is approximately separable. A kernel is approximately separable if it can be
    decomposed into two 1D vectors, such that the product of the two vectors is close to the original kernel
    within a certain threshold. This is done by using Singular Value Decomposition (SVD).

    Parameters:
    ------------
    kernel (list of lists of float): The kernel to check.
    threshold (float): The threshold for the approximation. Default is 0.9.

    Returns:
    ------------
    bool: True if the kernel is approximately separable, False otherwise.
    """
    # Using SVD to decompose the kernel into 3 matrices
    u, s, vt = np.linalg.svd(kernel)

    # Calculating separability ratio
    separability_ratio = s[0] / np.sum(s)

    # Checking if the separability ratio is greater than the threshold
    return separability_ratio > threshold


def separate_Kernel(kernel: Kernel) -> tuple[Vector, Vector]:
    """
    This function separates a 2D kernel into two 1D kernels.
    The separation is done by using Singular Value Decomposition (SVD).
    It is assumed that the kernel is approximately separable.

    Parameters:
    ------------
    kernel (list of lists of float): The kernel to be separated.

    Returns:
    ------------
    tuple: A tuple containing the two 1D kernels.
    """
    # Using SVD to decompose the kernel into 3 matrices
    u, s, vt = np.linalg.svd(kernel)

    # Getting the first singular value and its corresponding vectors
    singular_value = s[0]
    vector1 = u[:, 0] * singular_value
    vector2 = vt[0, :]

    # Returning the two 1D kernels
    return vector1, vector2