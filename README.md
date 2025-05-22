# 📸 Image Analyzer Web App

This project is a Streamlit web application that allows users to upload two images and compare their quality based on calculated sharpness and noise levels. It also highlights the sharpest region within each uploaded image.

## Features

* Upload two images (JPG, JPEG, PNG).
* Displays uploaded images.
* Calculates and displays overall sharpness scores for each image.
* Calculates and displays overall noise level estimates for each image.
* Highlights the N sharpest region(s) in each image with a red rectangle on a grayscale version.
* Provides a comparative summary of which image is "better" based on these metrics.
* Built with Python, Streamlit, OpenCV, and Pillow.

## Setup and Installation

1.  **Clone the repository (or download the files):**
    ```bash
    # If you've pushed to GitHub:
    # git clone [https://github.com/your_username/your_repository_name.git](https://github.com/your_username/your_repository_name.git)
    # cd your_repository_name
    ```

2.  **Create and activate a Python virtual environment:**
    (Ensuring you have Python 3.8+ installed)
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On macOS/Linux
    # .venv\Scripts\activate   # On Windows
    ```

3.  **Install dependencies:**
    Make sure you are in the project directory where `requirements.txt` is located.
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  Ensure your virtual environment is activated.
2.  Navigate to the project directory in your terminal.
3.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
4.  The application will open in your default web browser (usually at `http://localhost:8501`).

## Project Structure