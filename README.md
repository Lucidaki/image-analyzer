# Image Analyzer 🖼️

Image Analyzer is a Python-based web application built with Streamlit that provides tools for evaluating and managing image quality. Users can upload images to analyze sharpness, estimate noise levels, identify and highlight the sharpest "focus points," compare images, rank multiple images by quality, and find duplicate or visually similar images within a batch.

## ✨ Features

* **Multi-Mode Analysis**:
    * **Single Image Analysis**: Upload one image to get its overall sharpness score, noise level estimation, and see its sharpest region(s) highlighted.
    * **Compare Two Images**: Upload two images to get a side-by-side comparison of their quality metrics, highlighted sharp regions, and a summary conclusion.
    * **Analyze/Rank Multiple Images**: Upload a batch of images to view their quality scores in a sortable table and select individual images for a detailed view (including highlighted sharp regions).
    * **Find Duplicate/Similar Images**: Upload a batch of images to identify and group visually similar or duplicate images using perceptual hashing.
* **Focus Point Detection**:
    * Identifies and highlights the N sharpest regions in an image on a grayscale version for clarity.
    * Users can adjust parameters for detection:
        * Number of sharpest regions to highlight.
        * Tile size for sharpness analysis.
        * Minimum sharpness score threshold for highlighting.
    * Optional toggle to display a pointer (arrow) to the primary sharpest region for better visibility.
* **Duplicate/Similarity Detection**:
    * Uses perceptual hashing to find visually similar images.
    * Users can configure:
        * Hashing algorithm (pHash, dHash, aHash, wHash).
        * Hash size.
        * Similarity threshold (maximum hash difference).
* **User-Friendly Interface**:
    * Interactive web application built with Streamlit.
    * Easy file uploading.
    * Clear presentation of results and highlighted images.
    * Adjustable settings via a sidebar for fine-tuning analysis.

## 🛠️ Technologies Used

* Python 3.8+
* Streamlit
* OpenCV (cv2)
* Pillow (PIL)
* NumPy
* Pandas
* ImageHash

## 🚀 Setup and Installation

1.  **Prerequisites**:
    * Ensure you have Python 3.8 or newer installed on your system. You can download it from [python.org](https://www.python.org/).
    * Git for cloning the repository.

2.  **Clone the Repository**:
    Open your terminal or command prompt and run:
    ```bash
    git clone [https://github.com/Lucidaki/image-analyzer.git](https://github.com/Lucidaki/image-analyzer.git)
    cd image-analyzer
    ```

3.  **Create and Activate a Python Virtual Environment**:
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    # For macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate

    # For Windows
    # python -m venv .venv
    # .venv\Scripts\activate
    ```

4.  **Install Dependencies**:
    With your virtual environment activated, install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 How to Run

1.  Ensure your virtual environment is activated (see step 3 in Setup).
2.  Navigate to the project's root directory (`image-analyzer`) in your terminal.
3.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```
4.  The application should automatically open in your default web browser. If not, navigate to the local URL displayed in your terminal (usually `http://localhost:8501`).

## 📝 Using the Application

1.  **Choose an Analysis Mode**: Use the radio buttons at the top to select one of the four modes:
    * Analyze Single Image
    * Compare Two Images
    * Analyze/Rank Multiple Images
    * Find Duplicate/Similar Images
2.  **Adjust Settings (Optional)**: Use the sidebar on the left to:
    * Configure "Focus Point Settings" (number of regions, tile size, sharpness threshold, pointer visibility). These settings apply when images with highlighted focus points are generated.
    * Configure "Duplicate Detection Settings" (hashing algorithm, hash size, similarity threshold). These settings apply only to the "Find Duplicate/Similar Images" mode.
3.  **Upload Images**: Follow the prompts in the selected mode to upload your image(s).
4.  **View Results**: The application will process the images and display the analysis, comparisons, rankings, or duplicate groups based on the selected mode and settings.

## 📁 Project Structure (Key Files)
image-analyzer/
├── app.py              # Main Streamlit application script
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── .gitignore          # Specifies intentionally untracked files for Git

---
This project serves as a practical application of image processing techniques and web application development with Streamlit.