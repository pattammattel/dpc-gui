import numpy as np

def mad(image, axis=None):
    """
    Calculate the Median Absolute Deviation (MAD) of the image along the specified axis.
    """
    median = np.median(image, axis=axis)
    return np.median(np.abs(image - median), axis=axis)

def find_hot_pixels_mad(image, threshold_factor=3):
    """
    Identify hot pixels in an image using the Median Absolute Deviation (MAD).
    
    Args:
    - image (numpy array): The 2D image array (grayscale).
    - threshold_factor (float): The number of MADs above the median to consider as a hot pixel.
    
    Returns:
    - hot_pixels (numpy array): A boolean array where True represents a hot pixel.
    """
    # Calculate the median and MAD of the image
    median = np.median(image)
    mad_value = mad(image)

    # Identify hot pixels: pixels that are more than threshold_factor MAD above the median
    hot_pixels = np.abs(image - median) > threshold_factor * mad_value

    return hot_pixels  # Boolean array with the same shape as the input image