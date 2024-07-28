import google.generativeai as genai  # Used to interact with Google's Generative AI models for generating content from prompts
import io  # Provides tools for working with streams (used for in-memory binary streams)
import pywt  # PyWavelets, a library for performing discrete wavelet transform (used for wavelet denoising)
import numpy as np  # A library for numerical operations on arrays (used for image processing)
from PIL import Image, ImageOps  # Python Imaging Library (PIL) for image manipulation
import streamlit as st  # A framework for creating interactive web applications, used to build the app's frontend

# API key for accessing Google Generative AI services
GOOGLE_API_KEY = "AIzaSyCN8bK-8lFUKTxMd2dBEgSSIPBsHEbnYig"
# Configuring the Google Generative AI client with the provided API key
genai.configure(api_key=GOOGLE_API_KEY)

# Configuration settings for the generative model
MODEL_CONFIG = {
    "temperature": 0.2,  # Controls the randomness of the output (lower values make output more deterministic)
    "top_p": 1,  # Cumulative probability for token sampling (1 means no restriction)
    "top_k": 32,  # Number of highest probability vocabulary tokens to keep for sampling
    "max_output_tokens": 4096,  # Maximum number of tokens in the generated output
}

# Safety settings to block certain types of harmful content
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
]

# Creating an instance of the GenerativeModel with the specified model name, configuration, and safety settings
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=MODEL_CONFIG,
    safety_settings=safety_settings,
)


def wavelet_denoising(image):
    image_array = np.array(image)
    coeffs = pywt.wavedec2(image_array, "db1", level=2)
    threshold = 0.2
    coeffs_thresholded = [
        tuple(pywt.threshold(c, threshold, mode="soft") for c in coeff)
        for coeff in coeffs
    ]
    denoised_image_array = pywt.waverec2(coeffs_thresholded, "db1")
    denoised_image = Image.fromarray(np.uint8(denoised_image_array))
    return denoised_image


def preprocess_image(image):
    denoised_image = wavelet_denoising(image)
    gray_image = ImageOps.grayscale(denoised_image)
    return gray_image


def image_format(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    image_parts = [{"mime_type": "image/jpeg", "data": img_bytes.getvalue()}]
    return image_parts


def gemini_output(image, system_prompt, user_prompt):
    try:
        image_info = image_format(image) # Extract information from the image using the image_format function
        input_prompt = [
            system_prompt,
            image_info[0],
            user_prompt,
        ]  # Combine the system prompt, image information, and user prompt into a single input prompt list
        response = model.generate_content(
            input_prompt
        )  # Generate content using the model with the provided input prompt
        return (
            response.text.strip()
        )  # Return the generated content as a stripped string (removing leading and trailing whitespace)
    except (
        Exception
    ) as e:  # If an error occurs, display an error message in the Streamlit app
        st.error(f"Error generating content: {e}")
        return ""


system_prompt = """
You are a specialist in comprehending electricity meter readings.
Input images in the form of electricity meter readings will be provided to you,
and your task is to respond to questions based on the content of the input image.
"""
