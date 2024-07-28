import streamlit as st
import cv2
import numpy as np
from PIL import Image
from preprocessing import preprocess_image
from model import gemini_output, system_prompt
from translate import Translator
from gtts import gTTS
import tempfile
import os
import base64


# Function to convert text to speech and return audio file path
def text_to_speech(text, lang="en"):
    tts = gTTS(text=text, lang=lang, slow=False)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        tts.save(temp_file.name)
        return temp_file.name


# Function to play audio automatically in the browser
def play_audio(file_path):
    audio_base64 = base64.b64encode(open(file_path, "rb").read()).decode()
    audio_html = f"""
    <audio autoplay>
        <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mpeg">
        Your browser does not support the audio element.
    </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)


# Function to translate text
def translate_text(text, target_language):
    translator = Translator(to_lang=target_language)
    translation = translator.translate(text)
    return translation


# Function to translate specific terms
def translate_meter_reading(value, target_language):
    translations = {
        "en": {"day": "Day", "night": "Night"},
        "ta": {"day": "நாள்", "night": "இரவு"},
        "kn": {"day": "ದಿನ", "night": "ರಾತ್ರಿ"},
        "ml": {"day": "ദിനം", "night": "രാത്രി"},
        "hi": {"day": "दिन", "night": "रात"},
        "ur": {"day": "دن", "night": "رات"},
        "te": {"day": "రోజు", "night": "రాత్రి"},
    }
    return translations.get(target_language, {}).get(value.lower(), value)


# Function to guide the user through the process
def guide_user_step(step_text, lang="en"):
    guide_audio_path = text_to_speech(step_text, lang)
    play_audio(guide_audio_path)
    st.write(step_text)


st.title("Electricity Meter Reading Extractor")

option = st.selectbox("Choose an option", ("Upload Image", "Use Live Video"))
languages = {
    "Tamil": "ta",
    "English": "en",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Hindi": "hi",
    "Urdu": "ur",
    "Telugu": "te",
}
language_choice = st.selectbox(
    "Choose a language for translation", list(languages.keys())
)

if option == "Upload Image":
    guide_user_step(
        "Please upload an image of the electricity meter.", languages[language_choice]
    )

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        guide_user_step("Processing your image.", languages[language_choice])

        preprocessed_image = preprocess_image(image)
        st.image(
            preprocessed_image, caption="Preprocessed Image", use_column_width=True
        )

        user_prompt = "What is the meter number in the image?"
        meter_number_output = gemini_output(
            preprocessed_image, system_prompt, user_prompt
        )
        if meter_number_output == "":
            result_text = "Meter Number: None"
        else:
            translated_meter_number = translate_text(
                meter_number_output, languages[language_choice]
            )
            result_text = f"Meter Number: {translated_meter_number}"

            user_prompt = "What is the electricity meter reading in the image?"
            reading_output = gemini_output(
                preprocessed_image, system_prompt, user_prompt
            )
            if reading_output == meter_number_output:
                result_text += "\nMeter Reading: None"
            else:
                translated_reading = translate_text(
                    reading_output, languages[language_choice]
                )
                translated_reading_day = translate_meter_reading(
                    "day", languages[language_choice]
                )
                translated_reading_night = translate_meter_reading(
                    "night", languages[language_choice]
                )

                if "day" in reading_output.lower():
                    result_text += f"\nMeter Reading ({translated_reading_day}): {translated_reading} kWh"
                elif "night" in reading_output.lower():
                    result_text += f"\nMeter Reading ({translated_reading_night}): {translated_reading} kWh"
                else:
                    result_text += f"\nMeter Reading: {translated_reading} kWh"

        st.write(result_text)

        # Convert result text to speech and play it automatically
        audio_path = text_to_speech(result_text, languages[language_choice])
        play_audio(audio_path)

elif option == "Use Live Video":
    guide_user_step(
        "Starting live video stream. Please point the camera at the electricity meter.",
        languages[language_choice],
    )

    video_placeholder = st.empty()
    result_placeholder = st.empty()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not open video stream.")
    else:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(
                frame_rgb, channels="RGB", caption="Live Video", use_column_width=True
            )

            frame_pil = Image.fromarray(frame_rgb)
            preprocessed_image = preprocess_image(frame_pil)

            video_placeholder.image(
                preprocessed_image, caption="Preprocessed Image", use_column_width=True
            )

            user_prompt = "What is the meter number in the image?"
            meter_number_output = gemini_output(
                preprocessed_image, system_prompt, user_prompt
            )
            if meter_number_output == "":
                result_text = "Meter Number: None"
            else:
                translated_meter_number = translate_text(
                    meter_number_output, languages[language_choice]
                )
                result_text = f"Meter Number: {translated_meter_number}"

                user_prompt = "What is the electricity meter reading in the image?"
                reading_output = gemini_output(
                    preprocessed_image, system_prompt, user_prompt
                )
                if reading_output == meter_number_output:
                    result_text += "\nMeter Reading: None"
                else:
                    translated_reading = translate_text(
                        reading_output, languages[language_choice]
                    )
                    translated_reading_day = translate_meter_reading(
                        "day", languages[language_choice]
                    )
                    translated_reading_night = translate_meter_reading(
                        "night", languages[language_choice]
                    )

                    if "day" in reading_output.lower():
                        result_text += f"\nMeter Reading ({translated_reading_day}): {translated_reading} kWh"
                    elif "night" in reading_output.lower():
                        result_text += f"\nMeter Reading ({translated_reading_night}): {translated_reading} kWh"
                    else:
                        result_text += f"\nMeter Reading: {translated_reading} kWh"

            result_placeholder.text(result_text)

          #   # Convert result text to speech and play it automatically
          #   audio_path = text_to_speech(result_text, languages[language_choice])
          #   play_audio(audio_path)

          #   if st.button("Stop Live Video"):
          #       cap.release()
          #       cv2.destroyAllWindows()
          #       st.write("Live video stopped.")
          #       break
