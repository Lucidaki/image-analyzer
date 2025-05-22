import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps

# --- Core Analysis and Highlighting Functions ---

def calculate_sharpness(cv_image):
    """Calculates sharpness from an OpenCV image (handles color or grayscale)."""
    if cv_image is None:
        return 0.0
    
    if len(cv_image.shape) == 3: # Color image
        cv_image_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    else: # Already grayscale
        cv_image_gray = cv_image
        
    laplacian = cv2.Laplacian(cv_image_gray, cv2.CV_64F)
    sharpness_score = np.var(laplacian)
    return sharpness_score

def calculate_noise_level(cv_image):
    """Estimates noise from an OpenCV image (handles color or grayscale)."""
    if cv_image is None:
        return float('inf') # Return infinity for non-existent image (worst noise)
    
    if len(cv_image.shape) == 3: # Color image
        cv_image_gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    else: # Already grayscale
        cv_image_gray = cv_image

    noise_score = np.std(cv_image_gray)
    return noise_score

def find_and_highlight_sharpest_regions(cv_image, num_regions=1, tile_size_wh=(64, 64)):
    """
    Finds the N sharpest regions in an image by analyzing non-overlapping tiles
    and highlights them with a red rectangle on a grayscale version of the image.

    Args:
        cv_image (numpy.ndarray): The input OpenCV image (assumed BGR if color).
        num_regions (int): The number of sharpest regions to highlight.
        tile_size_wh (tuple): A tuple (width, height) for the analysis tiles.

    Returns:
        numpy.ndarray: A new BGR image (appearing grayscale) with sharpest regions highlighted.
                       Returns None or a copy of input if processing fails.
    """
    if cv_image is None:
        print("Warning: Input image for find_and_highlight_sharpest_regions is None.")
        return None

    # Create a grayscale version for analysis
    if len(cv_image.shape) == 3 and cv_image.shape[2] == 3: # Color image
        gray_image_for_analysis = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    elif len(cv_image.shape) == 2: # Already grayscale
        gray_image_for_analysis = cv_image.copy() 
    else: 
        print(f"Warning: Unexpected image format for highlighting: shape {cv_image.shape}")
        return cv_image.copy() if cv_image is not None else None

    # Prepare the output image: Convert the grayscale image to BGR format
    # to allow drawing color rectangles on an image that looks grayscale.
    output_image_to_draw_on = cv2.cvtColor(gray_image_for_analysis, cv2.COLOR_GRAY2BGR)

    img_h, img_w = gray_image_for_analysis.shape[:2]
    tile_w, tile_h = tile_size_wh

    if img_h < tile_h or img_w < tile_w:
        # Image is smaller than tile size, return the grayscale BGR version without highlights
        return output_image_to_draw_on 

    tile_data = [] 

    for y in range(0, img_h - tile_h + 1, tile_h): # Non-overlapping tiles
        for x in range(0, img_w - tile_w + 1, tile_w): # Non-overlapping tiles
            tile = gray_image_for_analysis[y:y+tile_h, x:x+tile_w]
            if tile.size == 0: # Should not happen with current loop logic for full tiles
                continue
            
            laplacian = cv2.Laplacian(tile, cv2.CV_64F) # CV_64F for precision
            sharpness = np.var(laplacian)
            
            if np.isnan(sharpness) or np.isinf(sharpness): # Handle potential invalid math results
                sharpness = 0.0

            tile_data.append({'score': sharpness, 'x': x, 'y': y, 'w': tile_w, 'h': tile_h})

    if not tile_data:
        # No tiles were processed or analyzed
        return output_image_to_draw_on

    # Sort tiles by sharpness score in descending order
    sorted_tiles = sorted(tile_data, key=lambda item: item['score'], reverse=True)

    # Draw highlights on the "grayscale BGR" image
    for i in range(min(num_regions, len(sorted_tiles))):
        top_tile = sorted_tiles[i]
        # Only highlight if sharpness score is above a certain threshold (tune as needed)
        if top_tile['score'] > 10: 
            tx, ty, tw, th = top_tile['x'], top_tile['y'], top_tile['w'], top_tile['h']
            cv2.rectangle(output_image_to_draw_on, (tx, ty), (tx + tw, ty + th), (0, 0, 255), 2) # Red (BGR), thickness 2
    
    return output_image_to_draw_on

# --- Streamlit Application ---
st.set_page_config(layout="wide")
st.title("🖼️ Image Analyzer") # Updated title
st.write("Upload two images to compare their overall quality and see highlighted sharp regions.")

# Setup columns for file uploaders
col1, col2 = st.columns(2)
uploaded_file1 = None
uploaded_file2 = None

with col1:
    st.subheader("Image 1")
    uploaded_file1 = st.file_uploader("Upload Image 1", type=["jpg", "jpeg", "png"], key="file_uploader1")

with col2:
    st.subheader("Image 2")
    uploaded_file2 = st.file_uploader("Upload Image 2", type=["jpg", "jpeg", "png"], key="file_uploader2")


if uploaded_file1 is not None and uploaded_file2 is not None:
    try:
        # Process Image 1 from upload
        img1_pil = Image.open(uploaded_file1)
        img1_pil_oriented = ImageOps.exif_transpose(img1_pil) # Correct EXIF orientation
        img1_pil_rgb = img1_pil_oriented.convert('RGB') # Ensure RGB for consistency
        img1_cv_bgr = cv2.cvtColor(np.array(img1_pil_rgb), cv2.COLOR_RGB2BGR) # Convert to OpenCV BGR

        # Process Image 2 from upload
        img2_pil = Image.open(uploaded_file2)
        img2_pil_oriented = ImageOps.exif_transpose(img2_pil) # Correct EXIF orientation
        img2_pil_rgb = img2_pil_oriented.convert('RGB') # Ensure RGB for consistency
        img2_cv_bgr = cv2.cvtColor(np.array(img2_pil_rgb), cv2.COLOR_RGB2BGR) # Convert to OpenCV BGR

        st.divider() # Visual separator

        # Get images with highlighted sharpest regions
        img1_highlighted = find_and_highlight_sharpest_regions(img1_cv_bgr, num_regions=1, tile_size_wh=(64,64))
        img2_highlighted = find_and_highlight_sharpest_regions(img2_cv_bgr, num_regions=1, tile_size_wh=(64,64))
        
        # Display original and highlighted images
        display_col1, display_col2 = st.columns(2)
        with display_col1:
            st.image(img1_pil_rgb, caption=f"Original: {uploaded_file1.name}", use_container_width=True)
            if img1_highlighted is not None: 
                st.image(img1_highlighted, channels="BGR", caption=f"Sharpest Region (Grayscale): {uploaded_file1.name}", use_container_width=True)
        
        with display_col2:
            st.image(img2_pil_rgb, caption=f"Original: {uploaded_file2.name}", use_container_width=True)
            if img2_highlighted is not None: 
                st.image(img2_highlighted, channels="BGR", caption=f"Sharpest Region (Grayscale): {uploaded_file2.name}", use_container_width=True)

        st.divider() 
        st.header("📊 Overall Analysis Results")

        # Perform overall analysis on original BGR OpenCV images
        sharpness1 = calculate_sharpness(img1_cv_bgr)
        noise1 = calculate_noise_level(img1_cv_bgr)
        sharpness2 = calculate_sharpness(img2_cv_bgr)
        noise2 = calculate_noise_level(img2_cv_bgr)

        # Display overall analysis metrics
        results_col1, results_col2 = st.columns(2)
        with results_col1:
            st.subheader(f"Image 1: {uploaded_file1.name}")
            st.metric(label="Overall Sharpness", value=f"{sharpness1:.2f}")
            st.metric(label="Overall Noise Level", value=f"{noise1:.2f}")
        
        with results_col2:
            st.subheader(f"Image 2: {uploaded_file2.name}")
            st.metric(label="Overall Sharpness", value=f"{sharpness2:.2f}")
            st.metric(label="Overall Noise Level", value=f"{noise2:.2f}")
        
        st.divider()
        st.subheader("💬 Comparison Summary")

        # Overall comparison logic
        if sharpness1 > sharpness2:
            st.write(f"**Overall Sharpness**: Image 1 ('{uploaded_file1.name}') appears sharper.")
        elif sharpness2 > sharpness1:
            st.write(f"**Overall Sharpness**: Image 2 ('{uploaded_file2.name}') appears sharper.")
        else:
            st.write("**Overall Sharpness**: Both images have similar overall sharpness scores.")

        if noise1 < noise2:
            st.write(f"**Overall Noise Level**: Image 1 ('{uploaded_file1.name}') appears to have less noise/texture.")
        elif noise2 < noise1:
            st.write(f"**Overall Noise Level**: Image 2 ('{uploaded_file2.name}') appears to have less noise/texture.")
        else:
            st.write("**Overall Noise Level**: Both images have similar overall noise/texture levels.")
        
        st.divider()
        st.subheader("🏆 Conclusion")
        # Simplified conclusion logic
        if sharpness1 > sharpness2 and noise1 < noise2: # Img1 sharper and less noisy
            st.success(f"**Image 1 ('{uploaded_file1.name}') is likely better overall.**")
        elif sharpness2 > sharpness1 and noise2 < noise1: # Img2 sharper and less noisy
            st.success(f"**Image 2 ('{uploaded_file2.name}') is likely better overall.**")
        elif sharpness1 > sharpness2: # Img1 is sharper
            if noise1 <= noise2 * 1.1: # Img1 is sharper and not significantly noisier (e.g., noise within 10% of other)
                st.info(f"**Image 1 ('{uploaded_file1.name}') is preferred (significantly sharper, comparable noise).**")
            else: # Img1 is sharper but also significantly noisier
                st.warning(f"**Image 1 ('{uploaded_file1.name}') is sharper, but Image 2 ('{uploaded_file2.name}') has less noise. Preference depends on specific needs.**")
        elif sharpness2 > sharpness1: # Img2 is sharper
            if noise2 <= noise1 * 1.1: # Img2 is sharper and not significantly noisier
                st.info(f"**Image 2 ('{uploaded_file2.name}') is preferred (significantly sharper, comparable noise).**")
            else: # Img2 is sharper but also significantly noisier
                st.warning(f"**Image 2 ('{uploaded_file2.name}') is sharper, but Image 1 ('{uploaded_file1.name}') has less noise. Preference depends on specific needs.**")
        elif noise1 < noise2: # Sharpness is similar, Img1 less noisy
            st.info(f"**Sharpness is similar. Image 1 ('{uploaded_file1.name}') is preferred due to lower noise.**")
        elif noise2 < noise1: # Sharpness is similar, Img2 less noisy
            st.info(f"**Sharpness is similar. Image 2 ('{uploaded_file2.name}') is preferred due to lower noise.**")
        else: # Both sharpness and noise are very similar
            st.info("**Both images have very similar quality characteristics based on these metrics.**")

    except Exception as e:
        st.error(f"An error occurred during image processing: {e}")
        # For more detailed debugging during development, uncomment the next two lines:
        # import traceback 
        # st.error(traceback.format_exc())
else:
    st.info("☝️ Please upload two images to begin analysis.")