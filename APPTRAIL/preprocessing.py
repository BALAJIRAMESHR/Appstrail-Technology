import pywt  # pywt stands for PyWavelets, a Python library for discrete wavelet transform
import numpy as np  # numpy is a fundamental package for scientific computing in Python.
import cv2  # cv2 is the OpenCV library for computer vision
from skimage import exposure
from PIL import (
    Image,
    ImageOps,
)  # PIL is the Python Imaging Library, which provides image processing capabilities.


def load_image(image_path):
    return cv2.imread(cv2.samples.findFile(image_path), cv2.IMREAD_GRAYSCALE)

"wavelet_denoising"
""" 
img: The input image to be denoised. It is expected to be a 2D numpy array (grayscale image).
->wavelet: The type of wavelet to use for the transform. The default is "db1" (Daubechies wavelet with one vanishing moment).
->threshold: The threshold value for denoising. Coefficients below this value are set to zero.
->pywt.wavedec2: Performs a 2D discrete wavelet transform on the input image img. The result is a list of wavelet coefficients.
->coeffs[0]: The approximation coefficients (low-frequency components) at the coarsest scale.
->coeffs[1:]: The detail coefficients (high-frequency components) at each scale, stored as tuples of arrays
->coeffs_thresholded: A copy of the original coefficients list.
->coeffs_thresholded[1:]: The detail coefficients are processed.
->pywt.threshold(c, threshold, mode="soft"): Applies soft thresholding to each detail coefficient c.
->Soft Thresholding: Reduces the magnitude of coefficients by the threshold value. If a coefficient is smaller than the threshold, it is set to zero. This helps in reducing noise by eliminating small coefficients that are likely to be noise.
->pywt.waverec2: Reconstructs the denoised image from the thresholded wavelet coefficients."""

def wavelet_denoising(img, wavelet="db1", threshold=0.1): 
    coeffs = pywt.wavedec2(
        img, wavelet, level=2
    )  
    coeffs_thresholded = list(coeffs)
    coeffs_thresholded[1:] = [
        tuple(pywt.threshold(c, threshold, mode="soft") for c in detail)
        for detail in coeffs[1:]
    ]
    denoised_img = pywt.waverec2(coeffs_thresholded, wavelet)
    return denoised_img

"adaptive_histogram_equalization"

"""
->Check Direction: The function checks if the direction is "left".
->Rotate Image: If the direction is "left", the image is rotated 90 degrees counterclockwise.
->img.rotate(90, expand=True): This rotates the image by 90 degrees. The expand=True parameter ensures that the output image size is adjusted to accommodate the new orientation.
->Return Image: The rotated image is returned if the direction was "left". If the direction was not "left", the original image is returned.."""

def adaptive_histogram_equalization(
    img,
): # skimage is a library for image processing in Python. The exposure module is specifically used for adjusting image contrast using techniques like adaptive histogram equalization.
    return exposure.equalize_adapthist(img, clip_limit=0.03)


def rotate_image_if_needed(img, direction):
    if direction == "left":
        return img.rotate(90, expand=True)
    return img

"Preprocess_image"

"""
->Load and preprocess the image: Depending on the input type, load the image from a path or use the provided PIL image, then perform wavelet denoising and convert it to a PIL image.
->Enhance and finalize the image: Rotate the image if needed, convert it to grayscale, apply adaptive histogram equalization, and convert it back to a PIL image for the final result."""

def preprocess_image(image_input, rotate_direction="none"):
    if isinstance(image_input, str):
        # Load image from path
        image = load_image(image_input)
    else:
        # Use provided PIL image
        image = np.array(image_input.convert("L"))

    # Perform wavelet denoising
    denoised_image = wavelet_denoising(image)

    # Convert denoised image to PIL Image
    pil_image = Image.fromarray(np.uint8(denoised_image))

    # Rotate image if needed
    rotated_image = rotate_image_if_needed(pil_image, rotate_direction)

    # Convert rotated image to grayscale
    gray_image = ImageOps.grayscale(rotated_image)

    # Apply adaptive histogram equalization
    ahe_image = adaptive_histogram_equalization(np.array(gray_image))

    # Convert back to PIL Image
    final_image = Image.fromarray(np.uint8(ahe_image * 255))

    return final_image
