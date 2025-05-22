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
        print("Warning (find_and_highlight_sharpest_regions): Input image is None.")
        return None 
    
    # Prepare grayscale image for analysis and BGR base for drawing highlights
    if len(cv_image_orig_color.shape) == 3 and cv_image_orig_color.shape[2] == 3: # BGR Color input
        gray_image_for_analysis = cv2.cvtColor(cv_image_orig_color, cv2.COLOR_BGR2GRAY)
    elif len(cv_image_orig_color.shape) == 2: # Grayscale input
        gray_image_for_analysis = cv_image_orig_color.copy()
    else: 
        print(f"Warning (find_and_highlight_sharpest_regions): Unexpected image format for analysis: shape {cv_image_orig_color.shape}")
        return cv_image_orig_color.copy() if cv_image_orig_color is not None else None

    output_image_to_draw_on = cv2.cvtColor(gray_image_for_analysis, cv2.COLOR_GRAY2BGR)
    
    img_h, img_w = gray_image_for_analysis.shape[:2]
    tile_w, tile_h = tile_size_wh
    
    if img_h < tile_h or img_w < tile_w: 
        return output_image_to_draw_on

    tile_data = []
    for y in range(0, img_h - tile_h + 1, tile_h):
        for x in range(0, img_w - tile_w + 1, tile_w):
            tile = gray_image_for_analysis[y:y+tile_h, x:x+tile_w]
            if tile.size == 0: continue

            laplacian = cv2.Laplacian(tile, cv2.CV_64F) 
            sharpness = np.var(laplacian)
            if np.isnan(sharpness) or np.isinf(sharpness): 
                sharpness = 0.0
            tile_data.append({'score': sharpness, 'x': x, 'y': y, 'w': tile_w, 'h': tile_h})
    
    if not tile_data: 
        return output_image_to_draw_on
    
    sorted_tiles = sorted(tile_data, key=lambda item: item['score'], reverse=True)
    
    for i in range(min(num_regions, len(sorted_tiles))):
        top_tile = sorted_tiles[i]
        if top_tile['score'] > sharpness_threshold: 
            tx, ty, tw, th = top_tile['x'], top_tile['y'], top_tile['w'], top_tile['h']
            
            highlight_color = (0, 255, 255)
            rectangle_thickness = 3          
            cv2.rectangle(output_image_to_draw_on, (tx, ty), (tx + tw, ty + th), highlight_color, rectangle_thickness)
            
            center_x_tile = tx + tw // 2
            center_y_tile = ty + th // 2
            circle_radius = 5 
            cv2.circle(output_image_to_draw_on, (center_x_tile, center_y_tile), circle_radius, highlight_color, -1)

            if show_pointer and i == 0: 
                pointer_color = (255, 255, 255); pointer_thickness = 3; arrow_tip_length = 0.03 
                img_center_x, img_center_y = img_w // 2, img_h // 2
                padding_x = img_w // 4; padding_y = img_h // 4 
                start_x, start_y = 0,0

                if center_x_tile < img_center_x: start_x = img_w - padding_x
                else: start_x = padding_x
                if center_y_tile < img_center_y: start_y = img_h - padding_y
                else: start_y = padding_y
                
                if abs(start_x - center_x_tile) > 10 or abs(start_y - center_y_tile) > 10 : 
                    cv2.arrowedLine(output_image_to_draw_on, (start_x, start_y), (center_x_tile, center_y_tile), 
                                    pointer_color, pointer_thickness, tipLength=arrow_tip_length)
    return output_image_to_draw_on

def find_image_duplicates(pil_images_with_names, hash_algo_str="pHash", hash_size=8, similarity_threshold=5):
    """Finds groups of duplicate or visually similar images from a list of PIL images."""
    if not pil_images_with_names or len(pil_images_with_names) < 2: return []

    hash_func_map = {
        "pHash": imagehash.phash, "dHash": imagehash.dhash,
        "aHash": imagehash.average_hash, "wHash (Wavelet)": imagehash.whash
    }
    hash_func = hash_func_map.get(hash_algo_str, imagehash.phash) # Default to pHash
    if hash_algo_str not in hash_func_map:
        print(f"Warning (find_image_duplicates): Unknown hash_algo_str '{hash_algo_str}', defaulting to pHash.")
        
    image_details = []
    for filename, pil_image in pil_images_with_names:
        try:
            img_hash = hash_func(pil_image, hash_size=hash_size)
            image_details.append({'filename': filename, 'pil_image': pil_image, 'hash': img_hash})
        except Exception as e: 
            print(f"Error hashing {filename}: {e}")
            # Optionally, you could add this error to a list to display in Streamlit
            # st.toast(f"Could not hash {filename}", icon="⚠️")

    if not image_details or len(image_details) < 2: return []
    
    groups_of_duplicates = []; processed_indices = [False] * len(image_details)
    for i in range(len(image_details)):
        if processed_indices[i]: continue
        current_group = [image_details[i]]; processed_indices[i] = True
        for j in range(i + 1, len(image_details)):
            if processed_indices[j]: continue
            try:
                hash_difference = image_details[i]['hash'] - image_details[j]['hash']
                if hash_difference <= similarity_threshold:
                    current_group.append(image_details[j]); processed_indices[j] = True
            except Exception as e: # Catch potential errors during hash comparison
                print(f"Error comparing hashes for {image_details[i]['filename']} and {image_details[j]['filename']}: {e}")

        if len(current_group) > 1: groups_of_duplicates.append(current_group)
    return groups_of_duplicates

# --- Streamlit Application UI and Logic ---
st.set_page_config(layout="wide", page_title="Image Analyzer", page_icon="🖼️") # Added page_icon
st.title("Image Analyzer")
st.write("Upload image(s) to analyze quality, compare, find duplicates, and see highlighted sharp regions.")

# Sidebar for settings
st.sidebar.header("Focus Point Settings")
num_regions_to_highlight = st.sidebar.number_input(
    "Highlight Top N Sharp Areas:", min_value=1, max_value=10, value=1, step=1, 
    help="Choose how many sharpest spots to mark.", key="num_regions_sidebar"
)
tile_size_options = {
    "X-Small (16x16) - More Detail": 16, "Small (32x32) - Good Detail": 32, 
    "Medium (64x64) - Balanced": 64, "Large (128x128) - Broader Areas": 128,
    "X-Large (256x256) - Very Broad": 256
}
selected_tile_size_label = st.sidebar.selectbox(
    "Sharpness Detail Level:", options=list(tile_size_options.keys()), index=2, 
    help="Controls how finely the image is scanned for sharpness.", key="tile_size_sidebar"
)
tile_size = tile_size_options[selected_tile_size_label]
sharpness_threshold_for_highlight = st.sidebar.slider(
    "Minimum Clarity to Highlight:", min_value=0.0, max_value=500.0, value=10.0, step=1.0, 
    help="Only areas sharper than this will be marked.", key="sharp_thresh_sidebar"
)
show_focus_pointer_toggle = st.sidebar.toggle(
    "Show Arrow to Main Focus Point", value=False, key="pointer_toggle",
    help="If on, an arrow points to the most important sharp area."
)

st.sidebar.divider() 
st.sidebar.header("Duplicate Detection Settings")
hash_algo_friendly_names = {
    "Standard Visual Match (Recommended)": "pHash", "Quick Difference Check": "dHash",
    "Average Tone Match": "aHash", "Detailed Structural Match (Slower)": "wHash (Wavelet)" 
}
selected_friendly_hash_name = st.sidebar.selectbox(
    "Similarity Check Method:", options=list(hash_algo_friendly_names.keys()), index=0, 
    help="Method to compare images. 'Standard Visual Match' is a good balance.", key="hash_algo_sidebar"
)
selected_hash_algo = hash_algo_friendly_names[selected_friendly_hash_name] 
hash_size_duplicate = st.sidebar.slider(
    "Fingerprint Detail (Duplicates):", min_value=4, max_value=32, value=8, step=4, 
    help="Controls detail of the image 'fingerprint'. Higher can be more accurate but slower.", 
    key="hash_size_dup_sidebar"
)
similarity_threshold_duplicate = st.sidebar.slider(
    "How Similar to Count as Duplicate?:", min_value=0, max_value=30, value=5, step=1,
    help="Max difference between fingerprints. '0' for nearly identical. 1-5 for close copies.", 
    key="sim_thresh_dup_sidebar"
)

# Main application modes
analysis_mode = st.radio(
    "Choose Analysis Mode:",
    ("Analyze Single Image", "Compare Two Images", "Analyze/Rank Multiple Images", "Find Duplicate/Similar Images"),
    horizontal=True, key="analysis_mode_selector"
)
st.divider()

# --- Single Image Analysis Mode ---
if analysis_mode == "Analyze Single Image":
    st.subheader("Single Image Analysis")
    uploaded_file_single = st.file_uploader("Upload your image here:", type=["jpg", "jpeg", "png"], key="single_uploader")
    if uploaded_file_single is not None:
        try:
            img_pil = Image.open(uploaded_file_single); img_pil_oriented = ImageOps.exif_transpose(img_pil); img_pil_rgb = img_pil_oriented.convert('RGB')
            img_cv_bgr = cv2.cvtColor(np.array(img_pil_rgb), cv2.COLOR_RGB2BGR)
            st.divider()
            img_highlighted = find_and_highlight_sharpest_regions(
                img_cv_bgr.copy(), num_regions=num_regions_to_highlight, 
                tile_size_wh=(tile_size, tile_size), sharpness_threshold=sharpness_threshold_for_highlight,
                show_pointer=show_focus_pointer_toggle
            )
            display_col_orig, display_col_highlight = st.columns(2)
            with display_col_orig: st.image(img_pil_rgb, caption=f"Original: {uploaded_file_single.name}", use_container_width=True)
            with display_col_highlight:
                if img_highlighted is not None: st.image(img_highlighted, channels="BGR", caption=f"Sharpest Region(s): {uploaded_file_single.name}", use_container_width=True)
            st.divider(); st.header("Analysis Results")
            sharpness = calculate_sharpness(img_cv_bgr); noise = calculate_noise_level(img_cv_bgr)
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1: st.metric(label="Overall Sharpness", value=f"{sharpness:.2f}")
            with metric_col2: st.metric(label="Overall Noise Level", value=f"{noise:.2f}")
        except Exception as e: st.error(f"An error occurred processing the image: {e}")
    else: st.info("Please upload an image for analysis.")

# --- Compare Two Images Mode ---
elif analysis_mode == "Compare Two Images":
    st.subheader("Compare Two Images")
    col1_upload, col2_upload = st.columns(2)
    with col1_upload: uploaded_file1 = st.file_uploader("Upload Image 1:", type=["jpg", "jpeg", "png"], key="file1_compare") # Simplified label
    with col2_upload: uploaded_file2 = st.file_uploader("Upload Image 2:", type=["jpg", "jpeg", "png"], key="file2_compare") # Simplified label

    if uploaded_file1 is not None and uploaded_file2 is not None:
        try:
            img1_pil = Image.open(uploaded_file1); img1_pil_oriented = ImageOps.exif_transpose(img1_pil); img1_pil_rgb = img1_pil_oriented.convert('RGB')
            image1_cv_bgr = cv2.cvtColor(np.array(img1_pil_rgb), cv2.COLOR_RGB2BGR)
            img2_pil = Image.open(uploaded_file2); img2_pil_oriented = ImageOps.exif_transpose(img2_pil); img2_pil_rgb = img2_pil_oriented.convert('RGB')
            image2_cv_bgr = cv2.cvtColor(np.array(img2_pil_rgb), cv2.COLOR_RGB2BGR)
            st.divider()
            image1_with_highlights = find_and_highlight_sharpest_regions(
                image1_cv_bgr.copy(), num_regions=num_regions_to_highlight, tile_size_wh=(tile_size, tile_size), 
                sharpness_threshold=sharpness_threshold_for_highlight, show_pointer=show_focus_pointer_toggle 
            )
            image2_with_highlights = find_and_highlight_sharpest_regions(
                image2_cv_bgr.copy(), num_regions=num_regions_to_highlight, tile_size_wh=(tile_size, tile_size), 
                sharpness_threshold=sharpness_threshold_for_highlight, show_pointer=show_focus_pointer_toggle 
            )
            display_col1, display_col2 = st.columns(2)
            with display_col1:
                st.image(img1_pil_rgb, caption=f"Original: {uploaded_file1.name}", use_container_width=True)
                if image1_with_highlights is not None: st.image(image1_with_highlights, channels="BGR", caption=f"Sharpest Region(s): {uploaded_file1.name}", use_container_width=True)
            with display_col2:
                st.image(img2_pil_rgb, caption=f"Original: {uploaded_file2.name}", use_container_width=True)
                if image2_with_highlights is not None: st.image(image2_with_highlights, channels="BGR", caption=f"Sharpest Region(s): {uploaded_file2.name}", use_container_width=True)
            
            st.divider(); st.header("Overall Analysis Results")
            s1=calculate_sharpness(image1_cv_bgr); n1=calculate_noise_level(image1_cv_bgr)
            s2=calculate_sharpness(image2_cv_bgr); n2=calculate_noise_level(image2_cv_bgr)
            res_col1,res_col2=st.columns(2)
            with res_col1:st.subheader(f"Image 1: {uploaded_file1.name}");st.metric("Sharpness",f"{s1:.2f}");st.metric("Noise",f"{n1:.2f}")
            with res_col2:st.subheader(f"Image 2: {uploaded_file2.name}");st.metric("Sharpness",f"{s2:.2f}");st.metric("Noise",f"{n2:.2f}")
            
            st.divider();st.subheader("Comparison Summary")
            if s1 > s2: st.write(f"**Sharpness**: Image 1 is sharper.")
            elif s2 > s1: st.write(f"**Sharpness**: Image 2 is sharper.")
            else: st.write("**Sharpness**: Similar.")
            if n1 < n2: st.write(f"**Noise**: Image 1 has less noise.")
            elif n2 < n1: st.write(f"**Noise**: Image 2 has less noise.")
            else: st.write("**Noise**: Similar.")

            st.divider();st.subheader("Conclusion")
            if s1 > s2 and n1 < n2: st.success(f"**Image 1 ('{uploaded_file1.name}') is likely better overall.**")
            elif s2 > s1 and n2 < n1: st.success(f"**Image 2 ('{uploaded_file2.name}') is likely better overall.**")
            elif s1 > s2:
                if n1 <= n2*1.1: st.info(f"**Image 1 ('{uploaded_file1.name}') is preferred (sharper, comparable noise).**")
                else: st.warning(f"**Image 1 ('{uploaded_file1.name}') is sharper, but Image 2 ('{uploaded_file2.name}') has less noise.**")
            elif s2 > s1:
                if n2 <= n1*1.1: st.info(f"**Image 2 ('{uploaded_file2.name}') is preferred (sharper, comparable noise).**")
                else: st.warning(f"**Image 2 ('{uploaded_file2.name}') is sharper, but Image 1 ('{uploaded_file1.name}') has less noise.**")
            elif n1 < n2: st.info(f"**Sharpness similar. Image 1 ('{uploaded_file1.name}') is preferred (less noise).**")
            elif n2 < n1: st.info(f"**Sharpness similar. Image 2 ('{uploaded_file2.name}') is preferred (less noise).**")
            else: st.info("**Both images appear to have similar quality.**")
        except Exception as e:st.error(f"An error occurred processing images: {e}")
    else:
        if analysis_mode=="Compare Two Images":st.info("Please upload two images for comparison.")

# Analyze/Rank Multiple Images Mode
elif analysis_mode == "Analyze/Rank Multiple Images":
    st.subheader("Analyze and Rank Multiple Images")
    uploaded_files_multiple = st.file_uploader(
        "Upload your images here (select multiple):", type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True, key="multiple_uploader_rank"
    )
    if 'multi_image_cache_rank' not in st.session_state: 
        st.session_state.multi_image_cache_rank = {}
    
    current_uploaded_filenames_rank = {f.name for f in uploaded_files_multiple} if uploaded_files_multiple else set()
    
    # Reprocess if uploaded files change
    if uploaded_files_multiple and (current_uploaded_filenames_rank != st.session_state.get('processed_filenames_rank', set())):
        st.session_state.multi_image_cache_rank = {} 
        st.session_state.processed_filenames_rank = current_uploaded_filenames_rank # Update processed set
        
        progress_bar = st.progress(0); total_files = len(uploaded_files_multiple)
        temp_cache = {} # Process into a temporary cache first
        for i, uploaded_file in enumerate(uploaded_files_multiple):
            try:
                img_pil = Image.open(uploaded_file); img_pil_oriented = ImageOps.exif_transpose(img_pil); img_pil_rgb = img_pil_oriented.convert('RGB')
                img_cv_bgr = cv2.cvtColor(np.array(img_pil_rgb), cv2.COLOR_RGB2BGR)
                sharpness = calculate_sharpness(img_cv_bgr); noise = calculate_noise_level(img_cv_bgr)
                temp_cache[uploaded_file.name] = {
                    "filename": uploaded_file.name, "sharpness": sharpness, "noise": noise,
                    "pil_image_rgb": img_pil_rgb, "cv_image_bgr": img_cv_bgr 
                }
            except Exception as e: st.error(f"Error processing {uploaded_file.name}: {e}")
            progress_bar.progress((i + 1) / total_files)
        st.session_state.multi_image_cache_rank = temp_cache # Assign to session state once all processed
        progress_bar.empty()

    if st.session_state.multi_image_cache_rank:
        st.divider(); st.header("Image Analysis Summary & Ranking")
        
        # Create DataFrame from cached data for summary table
        df_list_rank = [{'Filename': data['filename'], 
                         'Sharpness Score': round(data['sharpness'],2), 
                         'Noise Level': round(data['noise'],2)} 
                       for data in st.session_state.multi_image_cache_rank.values()]
                       
        if not df_list_rank: st.info("No image data to display.")
        else:
            df_display_rank = pd.DataFrame(df_list_rank)
            st.write("Click on column headers to sort by that metric:")
            st.dataframe(df_display_rank.sort_values(by="Sharpness Score", ascending=False), 
                         use_container_width=True, hide_index=True)
            
            st.divider(); st.subheader("View Image Details")
            # Use filenames from the DataFrame for selection to match table order if user sorts it
            # However, selectbox options don't reorder with DataFrame sort. So, use sorted list of filenames.
            sorted_filenames_for_selection = df_display_rank.sort_values(by="Sharpness Score", ascending=False)['Filename'].tolist()

            if sorted_filenames_for_selection:
                selected_filename_rank = st.selectbox(
                    "Select an image to see details:", 
                    options=sorted_filenames_for_selection, 
                    index=0, # Default to top ranked
                    key="multi_image_detail_selector_rank"
                )
                if selected_filename_rank and selected_filename_rank in st.session_state.multi_image_cache_rank:
                    selected_data = st.session_state.multi_image_cache_rank[selected_filename_rank]
                    img_cv_bgr_sel = selected_data['cv_image_bgr']
                    img_pil_rgb_sel = selected_data['pil_image_rgb']
                    img_highlighted_sel = find_and_highlight_sharpest_regions(
                        img_cv_bgr_sel.copy(), num_regions=num_regions_to_highlight, 
                        tile_size_wh=(tile_size, tile_size),
                        sharpness_threshold=sharpness_threshold_for_highlight, 
                        show_pointer=show_focus_pointer_toggle
                    )
                    st.markdown(f"#### Details for: {selected_data['filename']}")
                    col_orig_det, col_high_det = st.columns(2)
                    with col_orig_det: st.image(img_pil_rgb_sel, caption="Original", use_container_width=True)
                    with col_high_det:
                        if img_highlighted_sel is not None: st.image(img_highlighted_sel, channels="BGR", caption="Sharpest Region(s)", use_container_width=True)
                    
                    score_col1, score_col2 = st.columns(2)
                    with score_col1: st.metric("Sharpness", f"{selected_data['sharpness']:.2f}")
                    with score_col2: st.metric("Noise", f"{selected_data['noise']:.2f}")
            else: 
                st.info("No images processed in this session for detailed view.") 
    elif not uploaded_files_multiple: # No files uploaded yet in this mode
        st.info("Please upload multiple images for analysis and ranking.")


# Find Duplicate/Similar Images Mode
elif analysis_mode == "Find Duplicate/Similar Images":
    st.subheader("Find Duplicate or Similar Images")
    uploaded_files_duplicates = st.file_uploader(
        "Upload images to check for duplicates:", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True, key="duplicate_uploader"
    )
    if uploaded_files_duplicates:
        if len(uploaded_files_duplicates) < 2: st.warning("Please upload at least two images to find duplicates.")
        else:
            pil_images_for_dup = []
            with st.spinner(f"Preparing {len(uploaded_files_duplicates)} images for duplicate check..."):
                for up_file in uploaded_files_duplicates:
                    try:
                        img_pil = Image.open(up_file); img_pil_oriented = ImageOps.exif_transpose(img_pil); img_pil_rgb = img_pil_oriented.convert('RGB') 
                        pil_images_for_dup.append((up_file.name, img_pil_rgb))
                    except Exception as e: st.error(f"Error preparing {up_file.name}: {e}")
            
            if pil_images_for_dup and len(pil_images_for_dup) >=2:
                hash_algo_to_use = selected_hash_algo 
                hash_size_to_use = hash_size_duplicate  
                sim_thresh_to_use = similarity_threshold_duplicate
                st.info(f"Checking {len(pil_images_for_dup)} images using '{hash_algo_to_use}' (size: {hash_size_to_use}, threshold: {sim_thresh_to_use})...")
                
                with st.spinner("Finding duplicates... This might take a moment for many images or large images."):
                    duplicate_groups = find_image_duplicates(
                        pil_images_for_dup, hash_algo_str=hash_algo_to_use,
                        hash_size=hash_size_to_use, similarity_threshold=sim_thresh_to_use
                    )
                st.divider()
                if not duplicate_groups: st.success("No significant duplicates or similarities found with the current settings.")
                else:
                    st.header(f"Found {len(duplicate_groups)} Group(s) of Similar/Duplicate Images:")
                    for i, group_items in enumerate(duplicate_groups):
                        st.subheader(f"Group {i+1}")
                        
                        group_items_with_scores_display = []
                        for item_detail in group_items:
                            # Convert PIL image from group to OpenCV BGR for analysis
                            cv_bgr_img_dup = cv2.cvtColor(np.array(item_detail['pil_image']), cv2.COLOR_RGB2BGR)
                            sharp = calculate_sharpness(cv_bgr_img_dup)
                            noise_val = calculate_noise_level(cv_bgr_img_dup)
                            group_items_with_scores_display.append({**item_detail, 'sharpness': sharp, 'noise': noise_val})
                        
                        best_in_group_disp = None
                        if group_items_with_scores_display:
                            best_in_group_disp = sorted(group_items_with_scores_display, key=lambda x: (-x['sharpness'], x['noise']))[0]

                        max_cols_disp = min(len(group_items_with_scores_display), 4)
                        cols_disp = st.columns(max_cols_disp) 
                        
                        filenames_text = [item['filename'] for item in group_items_with_scores_display]
                        st.write("Files in this group:", ", ".join(filenames_text))

                        for j, item_data_scored in enumerate(group_items_with_scores_display):
                            with cols_disp[j % max_cols_disp]:
                                caption_str = f"{item_data_scored['filename']}\nS: {item_data_scored['sharpness']:.2f}, N: {item_data_scored['noise']:.2f}\nHash: {item_data_scored['hash']}"
                                if best_in_group_disp and item_data_scored['filename'] == best_in_group_disp['filename']:
                                    st.success("🏆 Recommended Best in Group") 
                                st.image(item_data_scored['pil_image'], caption=caption_str, use_container_width=True)
                        st.caption("") 
                        st.divider()
    else:
        st.info("Please upload a batch of images to check for duplicates.")