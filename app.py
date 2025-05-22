import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
import imagehash

# --- Core Analysis and Highlighting Functions ---

def calculate_sharpness(cv_image):
    """Calculates sharpness from an OpenCV image (handles color or grayscale)."""
    if cv_image is None: return 0.0
    gray_img = cv_image if len(cv_image.shape) == 2 else cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
    return np.var(laplacian)

def calculate_noise_level(cv_image):
    """Estimates noise from an OpenCV image (handles color or grayscale)."""
    if cv_image is None: return float('inf')
    gray_img = cv_image if len(cv_image.shape) == 2 else cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    return np.std(gray_img)

def find_and_highlight_sharpest_regions(cv_image_orig_color, num_regions=1, tile_size_wh=(64, 64), 
                                        sharpness_threshold=10.0, show_pointer=False):
    """
    Finds N sharpest regions, highlights them on a grayscale version.
    Optionally draws a pointer to the primary sharpest region.
    Returns the highlighted image.
    """
    if cv_image_orig_color is None: 
        # This condition should ideally be caught before calling this utility function.
        print("Warning: Input image for find_and_highlight_sharpest_regions is None.")
        return None 
    
    # Prepare grayscale image for analysis and BGR base for drawing highlights
    if len(cv_image_orig_color.shape) == 3 and cv_image_orig_color.shape[2] == 3: # BGR Color input
        gray_image_for_analysis = cv2.cvtColor(cv_image_orig_color, cv2.COLOR_BGR2GRAY)
    elif len(cv_image_orig_color.shape) == 2: # Grayscale input
        gray_image_for_analysis = cv_image_orig_color.copy()
    else: # Unexpected format
        print(f"Warning: Unexpected image format for analysis in find_and_highlight_sharpest_regions: shape {cv_image_orig_color.shape}")
        return cv_image_orig_color.copy() if cv_image_orig_color is not None else None # Return a copy or None

    output_image_to_draw_on = cv2.cvtColor(gray_image_for_analysis, cv2.COLOR_GRAY2BGR) # BGR Grayscale for drawing
    
    img_h, img_w = gray_image_for_analysis.shape[:2]
    tile_w, tile_h = tile_size_wh
    
    if img_h < tile_h or img_w < tile_w: # Image too small for specified tile size
        return output_image_to_draw_on

    tile_data = []
    # Iterate through image by tiles
    for y in range(0, img_h - tile_h + 1, tile_h):
        for x in range(0, img_w - tile_w + 1, tile_w):
            tile = gray_image_for_analysis[y:y+tile_h, x:x+tile_w]
            if tile.size == 0: continue # Should not happen with current loop bounds

            laplacian = cv2.Laplacian(tile, cv2.CV_64F) # Use 64F for precision
            sharpness = np.var(laplacian)
            if np.isnan(sharpness) or np.isinf(sharpness): # Handle potential math errors
                sharpness = 0.0
            tile_data.append({'score': sharpness, 'x': x, 'y': y, 'w': tile_w, 'h': tile_h})
    
    if not tile_data: # No tiles were analyzed
        return output_image_to_draw_on
    
    sorted_tiles = sorted(tile_data, key=lambda item: item['score'], reverse=True)
    
    # Draw highlights and optional pointer
    for i in range(min(num_regions, len(sorted_tiles))):
        top_tile = sorted_tiles[i]
        if top_tile['score'] > sharpness_threshold: # Only highlight if score meets threshold
            tx, ty, tw, th = top_tile['x'], top_tile['y'], top_tile['w'], top_tile['h']
            
            highlight_color = (0, 255, 255)  # Bright Yellow (BGR)
            rectangle_thickness = 3          
            cv2.rectangle(output_image_to_draw_on, (tx, ty), (tx + tw, ty + th), highlight_color, rectangle_thickness)
            
            center_x_tile = tx + tw // 2
            center_y_tile = ty + th // 2
            circle_radius = 5 
            cv2.circle(output_image_to_draw_on, (center_x_tile, center_y_tile), circle_radius, highlight_color, -1) # Filled circle

            # Draw pointer for the primary (i=0) sharpest region if enabled
            if show_pointer and i == 0: 
                pointer_color = (255, 255, 255) # White
                pointer_thickness = 3 
                arrow_tip_length = 0.03 

                img_center_x, img_center_y = img_w // 2, img_h // 2
                # Start arrow further in from the corners to make it shorter
                padding_x = img_w // 4 
                padding_y = img_h // 4 
                start_x, start_y = 0,0

                if center_x_tile < img_center_x: start_x = img_w - padding_x
                else: start_x = padding_x
                
                if center_y_tile < img_center_y: start_y = img_h - padding_y
                else: start_y = padding_y
                
                # Draw arrow if start and end points are sufficiently different
                if abs(start_x - center_x_tile) > 10 or abs(start_y - center_y_tile) > 10 : 
                    cv2.arrowedLine(output_image_to_draw_on, (start_x, start_y), (center_x_tile, center_y_tile), 
                                    pointer_color, pointer_thickness, tipLength=arrow_tip_length)
    return output_image_to_draw_on

def find_image_duplicates(pil_images_with_names, hash_algo_str="pHash", hash_size=8, similarity_threshold=5):
    """
    Finds groups of duplicate or visually similar images from a list of PIL images.
    """
    if not pil_images_with_names or len(pil_images_with_names) < 2:
        return []

    # Select hashing function
    if hash_algo_str == "pHash": hash_func = imagehash.phash
    elif hash_algo_str == "dHash": hash_func = imagehash.dhash
    elif hash_algo_str == "aHash": hash_func = imagehash.average_hash
    elif hash_algo_str == "wHash (Wavelet)": hash_func = imagehash.whash 
    else: hash_func = imagehash.phash # Default
    
    image_details = [] # Store filename, PIL image, and its hash
    for filename, pil_image in pil_images_with_names:
        try:
            img_hash = hash_func(pil_image, hash_size=hash_size)
            image_details.append({'filename': filename, 'pil_image': pil_image, 'hash': img_hash})
        except Exception as e: 
            print(f"Error hashing {filename}: {e}") # Log error for debugging
            # Consider st.warning or st.toast if this needs to be user-visible without stopping execution

    if not image_details or len(image_details) < 2: return [] # Not enough images to compare
    
    groups_of_duplicates = []
    processed_indices = [False] * len(image_details) 

    for i in range(len(image_details)):
        if processed_indices[i]: continue # Skip if already grouped

        current_group = [image_details[i]] # Start new group
        processed_indices[i] = True

        for j in range(i + 1, len(image_details)):
            if processed_indices[j]: continue # Skip if already grouped

            hash_difference = image_details[i]['hash'] - image_details[j]['hash'] # Hamming distance
            if hash_difference <= similarity_threshold:
                current_group.append(image_details[j])
                processed_indices[j] = True
        
        if len(current_group) > 1: # A group must have at least two images
            groups_of_duplicates.append(current_group)
            
    return groups_of_duplicates

# --- Streamlit Application UI and Logic ---
st.set_page_config(layout="wide")
st.title("Image Analyzer")
st.write("Upload image(s) to analyze their quality, compare them, find duplicates, and see highlighted sharp regions.")

# Sidebar for settings
st.sidebar.header("Focus Point Settings")
num_regions_to_highlight = st.sidebar.number_input(
    "Number of Sharpest Regions:", min_value=1, max_value=10, value=1, step=1, key="num_regions_sidebar"
)
tile_size_options = {
    "X-Small (16x16)": 16, "Small (32x32)": 32, "Medium (64x64)": 64,
    "Large (128x128)": 128, "X-Large (256x256)": 256
}
selected_tile_size_label = st.sidebar.selectbox(
    "Tile Size (Focus Point):", options=list(tile_size_options.keys()), index=2, key="tile_size_sidebar"
)
tile_size = tile_size_options[selected_tile_size_label]
sharpness_threshold_for_highlight = st.sidebar.slider(
    "Min Sharpness (Focus Point):", min_value=0.0, max_value=500.0, value=10.0, step=1.0, key="sharp_thresh_sidebar"
)
show_focus_pointer_toggle = st.sidebar.toggle("Show Pointer to Focus Point", value=False, key="pointer_toggle")

st.sidebar.divider()
st.sidebar.header("Duplicate Detection Settings")
hash_algo_options = ["pHash", "dHash", "aHash", "wHash (Wavelet)"]
selected_hash_algo = st.sidebar.selectbox(
    "Hashing Algorithm:", options=hash_algo_options, index=0, key="hash_algo_sidebar"
)
hash_size_duplicate = st.sidebar.slider(
    "Hash Size (Duplicates):", min_value=4, max_value=32, value=8, step=4, key="hash_size_dup_sidebar"
)
similarity_threshold_duplicate = st.sidebar.slider(
    "Similarity Threshold (Max Hash Diff):", min_value=0, max_value=30, value=5, step=1,
    help="Lower means more similar. 0 for identical hashes.", key="sim_thresh_dup_sidebar"
)

# Main application modes
analysis_mode = st.radio(
    "Choose Analysis Mode:",
    ("Analyze Single Image", "Compare Two Images", "Analyze/Rank Multiple Images", "Find Duplicate/Similar Images"),
    horizontal=True, key="analysis_mode_selector"
)
st.divider()

# Single Image Analysis Mode
if analysis_mode == "Analyze Single Image":
    st.subheader("Upload an Image for Analysis")
    uploaded_file_single = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="single_uploader")
    if uploaded_file_single is not None:
        try:
            img_pil = Image.open(uploaded_file_single)
            img_pil_oriented = ImageOps.exif_transpose(img_pil)
            img_pil_rgb = img_pil_oriented.convert('RGB')
            img_cv_bgr = cv2.cvtColor(np.array(img_pil_rgb), cv2.COLOR_RGB2BGR)
            st.divider()
            img_highlighted = find_and_highlight_sharpest_regions(
                img_cv_bgr.copy(), num_regions=num_regions_to_highlight, 
                tile_size_wh=(tile_size, tile_size), sharpness_threshold=sharpness_threshold_for_highlight,
                show_pointer=show_focus_pointer_toggle
            )
            display_col_orig, display_col_highlight = st.columns(2)
            with display_col_orig:
                st.image(img_pil_rgb, caption=f"Original: {uploaded_file_single.name}", use_container_width=True)
            with display_col_highlight:
                if img_highlighted is not None:
                    st.image(img_highlighted, channels="BGR", caption=f"Sharpest Region(s) (Grayscale): {uploaded_file_single.name}", use_container_width=True)
            st.divider()
            st.header("Analysis Results")
            sharpness = calculate_sharpness(img_cv_bgr)
            noise = calculate_noise_level(img_cv_bgr)
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1: st.metric(label="Overall Sharpness", value=f"{sharpness:.2f}")
            with metric_col2: st.metric(label="Overall Noise Level", value=f"{noise:.2f}")
        except Exception as e: st.error(f"An error occurred processing the image: {e}")
    else: st.info("Please upload an image for analysis.")

# Compare Two Images Mode
elif analysis_mode == "Compare Two Images":
    st.subheader("Upload Two Images for Comparison")
    col1_upload, col2_upload = st.columns(2)
    uploaded_file1 = st.file_uploader("Upload Image 1", type=["jpg", "jpeg", "png"], key="file1_compare")
    uploaded_file2 = st.file_uploader("Upload Image 2", type=["jpg", "jpeg", "png"], key="file2_compare") # Changed key for uploader 2

    if uploaded_file1 is not None and uploaded_file2 is not None:
        try:
            img1_pil = Image.open(uploaded_file1); img1_pil_oriented = ImageOps.exif_transpose(img1_pil); img1_pil_rgb = img1_pil_oriented.convert('RGB')
            image1_cv_bgr = cv2.cvtColor(np.array(img1_pil_rgb), cv2.COLOR_RGB2BGR)
            img2_pil = Image.open(uploaded_file2); img2_pil_oriented = ImageOps.exif_transpose(img2_pil); img2_pil_rgb = img2_pil_oriented.convert('RGB')
            image2_cv_bgr = cv2.cvtColor(np.array(img2_pil_rgb), cv2.COLOR_RGB2BGR)
            st.divider()
            image1_with_highlights = find_and_highlight_sharpest_regions(
                image1_cv_bgr.copy(), num_regions=num_regions_to_highlight, 
                tile_size_wh=(tile_size, tile_size), sharpness_threshold=sharpness_threshold_for_highlight,
                show_pointer=show_focus_pointer_toggle 
            )
            image2_with_highlights = find_and_highlight_sharpest_regions(
                image2_cv_bgr.copy(), num_regions=num_regions_to_highlight, 
                tile_size_wh=(tile_size, tile_size), sharpness_threshold=sharpness_threshold_for_highlight,
                show_pointer=show_focus_pointer_toggle 
            )
            display_col1, display_col2 = st.columns(2)
            with display_col1:
                st.image(img1_pil_rgb, caption=f"Original: {uploaded_file1.name}", use_container_width=True)
                if image1_with_highlights is not None: st.image(image1_with_highlights, channels="BGR", caption=f"Sharpest Region(s) (Grayscale): {uploaded_file1.name}", use_container_width=True)
            with display_col2:
                st.image(img2_pil_rgb, caption=f"Original: {uploaded_file2.name}", use_container_width=True)
                if image2_with_highlights is not None: st.image(image2_with_highlights, channels="BGR", caption=f"Sharpest Region(s) (Grayscale): {uploaded_file2.name}", use_container_width=True)
            
            st.divider(); st.header("Overall Analysis Results")
            sharpness1 = calculate_sharpness(image1_cv_bgr); noise1 = calculate_noise_level(image1_cv_bgr)
            sharpness2 = calculate_sharpness(image2_cv_bgr); noise2 = calculate_noise_level(image2_cv_bgr)
            results_col1, results_col2 = st.columns(2)
            with results_col1: st.subheader(f"Image 1: {uploaded_file1.name}"); st.metric(label="Overall Sharpness", value=f"{sharpness1:.2f}"); st.metric(label="Overall Noise Level", value=f"{noise1:.2f}")
            with results_col2: st.subheader(f"Image 2: {uploaded_file2.name}"); st.metric(label="Overall Sharpness", value=f"{sharpness2:.2f}"); st.metric(label="Overall Noise Level", value=f"{noise2:.2f}")
            
            st.divider(); st.subheader("Comparison Summary")
            # (Comparison logic for sharpness and noise)
            if sharpness1 > sharpness2: st.write(f"**Overall Sharpness**: Image 1 ('{uploaded_file1.name}') appears sharper.")
            elif sharpness2 > sharpness1: st.write(f"**Overall Sharpness**: Image 2 ('{uploaded_file2.name}') appears sharper.")
            else: st.write("**Overall Sharpness**: Both images have similar overall sharpness scores.")
            if noise1 < noise2: st.write(f"**Overall Noise Level**: Image 1 ('{uploaded_file1.name}') appears to have less noise/texture.")
            elif noise2 < noise1: st.write(f"**Overall Noise Level**: Image 2 ('{uploaded_file2.name}') appears to have less noise/texture.")
            else: st.write("**Overall Noise Level**: Both images have similar overall noise/texture levels.")

            st.divider(); st.subheader("Conclusion")
            # (Conclusion logic)
            if sharpness1 > sharpness2 and noise1 < noise2: st.success(f"**Image 1 ('{uploaded_file1.name}') is likely better overall.**")
            elif sharpness2 > sharpness1 and noise2 < noise1: st.success(f"**Image 2 ('{uploaded_file2.name}') is likely better overall.**")
            elif sharpness1 > sharpness2:
                if noise1 <= noise2 * 1.1: st.info(f"**Image 1 ('{uploaded_file1.name}') is preferred (significantly sharper, comparable noise).**")
                else: st.warning(f"**Image 1 ('{uploaded_file1.name}') is sharper, but Image 2 ('{uploaded_file2.name}') has less noise. Preference depends on specific needs.**")
            elif sharpness2 > sharpness1:
                if noise2 <= noise1 * 1.1: st.info(f"**Image 2 ('{uploaded_file2.name}') is preferred (significantly sharper, comparable noise).**")
                else: st.warning(f"**Image 2 ('{uploaded_file2.name}') is sharper, but Image 1 ('{uploaded_file1.name}') has less noise. Preference depends on specific needs.**")
            elif noise1 < noise2: st.info(f"**Sharpness is similar. Image 1 ('{uploaded_file1.name}') is preferred due to lower noise.**")
            elif noise2 < noise1: st.info(f"**Sharpness is similar. Image 2 ('{uploaded_file2.name}') is preferred due to lower noise.**")
            else: st.info("**Both images have very similar quality characteristics based on these metrics.**")
        except Exception as e: st.error(f"An error occurred processing the images: {e}")
    else:
        if analysis_mode == "Compare Two Images": st.info("Please upload two images for comparison.")

# Analyze/Rank Multiple Images Mode
elif analysis_mode == "Analyze/Rank Multiple Images":
    st.subheader("Upload Multiple Images for Analysis and Ranking")
    uploaded_files_multiple = st.file_uploader(
        "Choose images (select multiple)...", type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True, key="multiple_uploader_rank" # Changed key
    )
    if 'multi_image_cache_rank' not in st.session_state: 
        st.session_state.multi_image_cache_rank = {}
    
    current_uploaded_filenames_rank = {f.name for f in uploaded_files_multiple} if uploaded_files_multiple else set()
    cached_filenames_rank = set(st.session_state.multi_image_cache_rank.keys())

    if uploaded_files_multiple and (current_uploaded_filenames_rank != cached_filenames_rank):
        st.session_state.multi_image_cache_rank = {} 
        progress_bar = st.progress(0); total_files = len(uploaded_files_multiple)
        for i, uploaded_file in enumerate(uploaded_files_multiple):
            try:
                img_pil = Image.open(uploaded_file); img_pil_oriented = ImageOps.exif_transpose(img_pil); img_pil_rgb = img_pil_oriented.convert('RGB')
                img_cv_bgr = cv2.cvtColor(np.array(img_pil_rgb), cv2.COLOR_RGB2BGR)
                sharpness = calculate_sharpness(img_cv_bgr); noise = calculate_noise_level(img_cv_bgr)
                st.session_state.multi_image_cache_rank[uploaded_file.name] = {
                    "filename": uploaded_file.name, "sharpness": sharpness, "noise": noise,
                    "pil_image_rgb": img_pil_rgb, "cv_image_bgr": img_cv_bgr 
                }
            except Exception as e: st.error(f"Error processing {uploaded_file.name}: {e}")
            progress_bar.progress((i + 1) / total_files)
        progress_bar.empty()

    if st.session_state.multi_image_cache_rank:
        st.divider(); st.header("Image Analysis Summary & Ranking")
        df_list_rank = [{'Filename': fname, 'Sharpness Score': round(data['sharpness'],2), 'Noise Level': round(data['noise'],2)} 
                   for fname, data in st.session_state.multi_image_cache_rank.items()]
        if not df_list_rank: st.info("No image data to display.")
        else:
            df_display_rank = pd.DataFrame(df_list_rank)
            st.write("Click on column headers to sort by that metric:")
            st.dataframe(df_display_rank.sort_values(by="Sharpness Score", ascending=False), use_container_width=True, hide_index=True)
            
            st.divider(); st.subheader("View Image Details")
            filenames_for_selection_rank = list(st.session_state.multi_image_cache_rank.keys())
            if filenames_for_selection_rank:
                selected_filename_rank = st.selectbox("Select an image to see details:", options=filenames_for_selection_rank, index=0, key="multi_image_detail_selector_rank")
                if selected_filename_rank and selected_filename_rank in st.session_state.multi_image_cache_rank:
                    selected_image_data_rank = st.session_state.multi_image_cache_rank[selected_filename_rank]
                    img_cv_bgr_selected_rank = selected_image_data_rank['cv_image_bgr']
                    img_pil_rgb_selected_rank = selected_image_data_rank['pil_image_rgb']
                    img_highlighted_selected_rank = find_and_highlight_sharpest_regions(
                        img_cv_bgr_selected_rank.copy(), num_regions=num_regions_to_highlight, tile_size_wh=(tile_size, tile_size),
                        sharpness_threshold=sharpness_threshold_for_highlight, show_pointer=show_focus_pointer_toggle
                    )
                    st.markdown(f"#### Details for: {selected_image_data_rank['filename']}")
                    col_orig_detail_rank, col_high_detail_rank = st.columns(2)
                    with col_orig_detail_rank: st.image(img_pil_rgb_selected_rank, caption="Original", use_container_width=True)
                    with col_high_detail_rank:
                        if img_highlighted_selected_rank is not None: st.image(img_highlighted_selected_rank, channels="BGR", caption="Sharpest Region(s)", use_container_width=True)
                    score_col1_detail_rank, score_col2_detail_rank = st.columns(2)
                    with score_col1_detail_rank: st.metric("Sharpness", f"{selected_image_data_rank['sharpness']:.2f}")
                    with score_col2_detail_rank: st.metric("Noise", f"{selected_image_data_rank['noise']:.2f}")
            else: st.info("No images processed yet in this session for detailed view.")
    elif uploaded_files_multiple is None or not uploaded_files_multiple : # Check the correct variable
        st.info("Please upload multiple images for analysis and ranking.")


# Find Duplicate/Similar Images Mode
elif analysis_mode == "Find Duplicate/Similar Images":
    st.subheader("Upload Images to Find Duplicates or Similarities")
    uploaded_files_duplicates = st.file_uploader(
        "Choose images to check for duplicates...", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True, key="duplicate_uploader"
    )

    if uploaded_files_duplicates:
        if len(uploaded_files_duplicates) < 2:
            st.warning("Please upload at least two images to find duplicates.")
        else:
            pil_images_with_names_for_dup = []
            with st.spinner(f"Preparing {len(uploaded_files_duplicates)} images..."):
                for up_file in uploaded_files_duplicates:
                    try:
                        img_pil = Image.open(up_file)
                        img_pil_oriented = ImageOps.exif_transpose(img_pil)
                        img_pil_rgb = img_pil_oriented.convert('RGB') 
                        pil_images_with_names_for_dup.append((up_file.name, img_pil_rgb))
                    except Exception as e: st.error(f"Error preparing {up_file.name}: {e}")
            
            if pil_images_with_names_for_dup and len(pil_images_with_names_for_dup) >=2:
                hash_algo = selected_hash_algo 
                hash_s = hash_size_duplicate    
                sim_thresh = similarity_threshold_duplicate
                st.info(f"Checking {len(pil_images_with_names_for_dup)} images using '{hash_algo}' (hash size: {hash_s}, threshold: {sim_thresh})...")
                
                with st.spinner("Finding duplicates... this might take a moment for many images."):
                    duplicate_groups = find_image_duplicates(
                        pil_images_with_names_for_dup, hash_algo_str=hash_algo,
                        hash_size=hash_s, similarity_threshold=sim_thresh
                    )

                st.divider()
                if not duplicate_groups:
                    st.success("No significant duplicates or similarities found with the current settings.")
                else:
                    st.header(f"Found {len(duplicate_groups)} Group(s) of Similar/Duplicate Images:")
                    for i, group in enumerate(duplicate_groups):
                        st.subheader(f"Group {i+1}:")
                        max_cols_dup = 4 
                        cols_dup = st.columns(min(len(group), max_cols_dup)) 
                        filenames_in_group_text = []
                        for j, item_detail in enumerate(group):
                            filenames_in_group_text.append(item_detail['filename'])
                            with cols_dup[j % max_cols_dup]:
                                st.image(item_detail['pil_image'], 
                                         caption=f"{item_detail['filename']}\nHash: {item_detail['hash']}", 
                                         use_container_width=True)
                        st.write("Files in this group:", ", ".join(filenames_in_group_text))
                        st.caption("") 
                        st.divider()
    else:
        st.info("Please upload a batch of images to check for duplicates.")