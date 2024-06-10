import streamlit as st
import cv2
import pandas as pd
import numpy as np
from PIL import Image

def variance_of_laplacian(image):
    """
    Compute the Laplacian of the image and return the variance
    which indicates the blurriness of the image.
    """
    return cv2.Laplacian(image, cv2.CV_64F).var()

def evaluate_blurriness(image):
    """
    Convert the image to grayscale and compute the blurriness score.
    """
    # Convert PIL Image to NumPy array
    image = np.array(image)

    # Ensure the image has 3 channels (RGB) or 4 channels (RGBA)
    if len(image.shape) == 2:
        gray = image
    elif len(image.shape) == 3 and image.shape[2] in [3, 4]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Invalid number of channels in input image")

    blur_score = variance_of_laplacian(gray)
    return blur_score

def evaluate_images(uploaded_files):
    """
    Evaluate the blurriness of all uploaded images and return the results as a DataFrame.
    """
    results = []

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        blur_score = evaluate_blurriness(image)
        results.append((uploaded_file.name, blur_score))

    # Create a DataFrame from the results and sort by blurriness score
    df_results = pd.DataFrame(results, columns=['Filename', 'Blurriness Score'])
    df_results = df_results.sort_values(by='Blurriness Score')

    return df_results

# Streamlit app
st.title('Image Blurriness Analysis')

uploaded_files = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg", "tiff", "bmp", "gif"], accept_multiple_files=True)

if st.button('Analyze'):
    if uploaded_files:
        try:
            results_df = evaluate_images(uploaded_files)
            st.write('Blurriness analysis completed.')
            st.dataframe(results_df)
        except Exception as e:
            st.error(f'Error: {e}')
    else:
        st.error('Please upload some images.')

if st.button('Clear Selection and Browse New Files'):
    st.experimental_rerun()
