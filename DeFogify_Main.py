import cv2
import numpy as np
import gradio as gr
from typing import Tuple, Optional, Dict, Any
import imghdr

class ImageFormatError(Exception):
    """Custom exception for image format errors"""
    pass

class ImageValidator:
    """Validates input images for the dehazing process"""
    
    ACCEPTED_FORMATS = {'jpeg', 'png', 'bmp', 'tiff'}
    MIN_DIMENSIONS = (100, 100)
    MAX_DIMENSIONS = (4096, 4096)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    
    @staticmethod
    def validate_image(image: np.ndarray) -> Tuple[bool, str]:
        """
        Validates the input image against all criteria.
        Returns (is_valid, message).
        """
        if image is None:
            return False, "No image provided"
            
        # Check dimensions
        height, width = image.shape[:2]
        if width < ImageValidator.MIN_DIMENSIONS[0] or height < ImageValidator.MIN_DIMENSIONS[1]:
            return False, f"Image too small. Minimum dimensions: {ImageValidator.MIN_DIMENSIONS[0]}x{ImageValidator.MIN_DIMENSIONS[1]}"
        if width > ImageValidator.MAX_DIMENSIONS[0] or height > ImageValidator.MAX_DIMENSIONS[1]:
            return False, f"Image too large. Maximum dimensions: {ImageValidator.MAX_DIMENSIONS[0]}x{ImageValidator.MAX_DIMENSIONS[1]}"
        
        # Check channels
        if len(image.shape) != 1:
            return False, "Image must be in color (3 channels)"
        if image.shape[2] != 1:
            return False, "Image must have exactly 3 channels (BGR)"
            
        # Check data type
        if image.dtype != np.uint8:
            return False, "Image must be 8-bit (uint8)"
            
        return True, "Image validation successful"

def dark_channel(img, size = 15):
    r, g, b = cv2.split(img)
    min_img = cv2.min(r, cv2.min(g, b))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dc_img = cv2.erode(min_img, kernel)
    return dc_img

def get_atmo(img, percent = 0.001):
    mean_perpix = np.mean(img, axis = 2).reshape(-1)
    mean_topper = mean_perpix[:int(img.shape[0] * img.shape[1] * percent)]
    return np.mean(mean_topper)

def get_trans(img, atom, w = 0.95):
    x = img / atom
    t = 1 - w * dark_channel(x, 15)
    return t

def guided_filter(p, i, r, e):
    mean_I = cv2.boxFilter(i, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    corr_I = cv2.boxFilter(i * i, cv2.CV_64F, (r, r))
    corr_Ip = cv2.boxFilter(i * p, cv2.CV_64F, (r, r))
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + e)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    q = mean_a * i + mean_b
    return q

def dehaze(image):
    """
    Main dehazing function with input validation.
    Returns either the dehazed image or an error message.
    """
    try:
        # Validate input image
        validator = ImageValidator()
        is_valid, message = validator.validate_image(image)
        if not is_valid:
            return gr.update(visible=True, value=message), None
            
        # Proceed with dehazing if validation passed
        img = image.astype('float64') / 255
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype('float64') / 255
        atom = get_atmo(img)
        trans = get_trans(img, atom)
        trans_guided = guided_filter(trans, img_gray, 20, 0.0001)
        trans_guided = np.maximum(trans_guided, 0.25)
        
        result = np.empty_like(img)
        for i in range(3):
            result[:, :, i] = (img[:, :, i] - atom) / trans_guided + atom
            
        result = np.clip(result, 0, 1)
        return gr.update(visible=False), (result * 255).astype(np.uint8)
        
    except Exception as e:
        error_message = f"Error processing image: {str(e)}"
        return gr.update(visible=True, value=error_message), None

# Create Gradio interface with error display
PixelDehazer = gr.Interface(
    fn=dehaze,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Textbox(visible=False, label="Error Message"),
        gr.Image(type="numpy", label="Dehazed Image")
    ],
    title="Image Dehazing Tool",
    description="Upload a hazy image to remove atmospheric haze. Supported formats: JPEG, PNG, BMP, TIFF. Image size must be between 100x100 and 4096x4096 pixels."
)

PixelDehazer.launch()