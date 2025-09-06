import streamlit as st
import numpy as np
from PIL import Image
import io
import json
import pyvista as pv
from stpyvista import stpyvista
from stpyvista.utils import start_xvfb
import trimesh
from streamlit_drawable_canvas import st_canvas
from streamlit_dimensions import st_dimensions
import cv2

# Import refactored logic
from heightmap import process_image
from mesh import (
    generate_mesh,
    get_transforms,
    scale_and_center_mesh
)

def modify_contours(pil_image, pixels):
    """
    Erodes or dilates the contours of an image.
    - A positive 'pixels' value dilates the contours.
    - A negative 'pixels' value erodes the contours.
    """
    gray = np.array(pil_image.convert('L'))

    # Inverted Otsu to get white contours on black background
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Kernel size must be a positive integer. The absolute value of 'pixels'
    # determines the magnitude of the operation.
    kernel_size = abs(pixels)
    if kernel_size == 0:
        return pil_image

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # The sign of 'pixels' determines the operation (dilate or erode).
    if pixels > 0: # Dilate
        modified_thresh = cv2.dilate(thresh, kernel, iterations=1)
    elif pixels < 0: # Erode
        modified_thresh = cv2.erode(thresh, kernel, iterations=1)
    else: # No change
        return pil_image

    # Reverse the inverse to get black contours on white background
    final_image_np = cv2.bitwise_not(modified_thresh)

    return Image.fromarray(final_image_np)

# --- Setup ---
try:
    start_xvfb()
except Exception as e:
    print("unable to start xvfb, when you run local this is fine!")#st.warning(f"(this is ok for local runs) Could not start virtual framebuffer: {e}")    
st.set_page_config(layout="wide")
st.title("🍪🔪 Cookie Cutter Generator")
st.write("""You have a cool design and want to turn it into a cookie cutter? 
         This tool helps you create a 3D printable model. 
         Prerequisites: Dark contours on bright background, one enclosed shape. 
         Adjust parameters in the sidebar as needed. 
         Some shapes are difficult to get the dough out. You can use the insert to push it out reliably.""")

# --- Sidebar UI ---
st.sidebar.header("Processing Parameters")
params = {
    "target_max": st.sidebar.number_input("Target Size [mm]", min_value=10.0, value=100.0, step=1.0),
    "min_wall": st.sidebar.number_input("Cutting Edge Width [mm]", min_value=0.1, value=1.0, step=0.1),
    "h_max": st.sidebar.number_input("Total Height [mm]", min_value=1.0, value=15.0, step=0.5),
    "h_rim": st.sidebar.number_input("Base/Rim Height [mm]", min_value=0.1, value=2.0, step=0.1),
    "w_rim": st.sidebar.number_input("Rim Width [mm]", min_value=0.1, value=5.0, step=0.1),
    "height_dough_thickness": st.sidebar.number_input("Dough Thickness [mm]", min_value=0.1, value=3.0, step=0.1),
    "small_fill": st.sidebar.number_input("Small Area Fill Threshold [mm^2]", min_value=0.0, value=10.0, step=0.1),
    "dpi": st.sidebar.number_input("DPI", min_value=50, value=200, step=10),
}
# # Create a copy of params for JSON export, excluding Streamlit UI elements
# params_for_export = {k: v for k, v in params.items() if not hasattr(v, 'get')}
# config_str = json.dumps(params_for_export, indent=4)
# st.sidebar.download_button(
#     label="Download Config", data=config_str, file_name="cookie_config.json", mime="application/json"
# )


# --- Main Page UI ---
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "bmp"])

if 'heightmap_array' not in st.session_state:
    st.session_state.heightmap_array = None
if 'insert_map_array' not in st.session_state:
    st.session_state.insert_map_array = None


if uploaded_file is not None:
    # Initialize session state for image
    if "current_file_name" not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
        st.session_state.current_file_name = uploaded_file.name
        st.session_state.original_image = Image.open(uploaded_file)
        st.session_state.active_image = st.session_state.original_image
        st.session_state.canvas_json_data = None
        
    st.sidebar.header("Contour Adjustment")
    st.info("""Note: The contours get automatically thickened to make them printable (see 'Cutting Edge Width').
            If the contours look broken or have gaps, try increasing the DPI first.(Increases processing time)""")
    col1, col2, col3 = st.sidebar.columns(3)
    adjustment_pixels = st.sidebar.number_input("Adjustment in Pixels", min_value=0, value=2, step=1)

    
    erode_button = col1.button("Erode")
    dilate_button = col2.button("Dilate")
    reset_button = col3.button("Reset")
    # Handle button clicks to apply cumulative adjustments
    if erode_button:
        st.session_state.active_image = modify_contours(st.session_state.active_image, -adjustment_pixels)
    if dilate_button:
        st.session_state.active_image = modify_contours(st.session_state.active_image, adjustment_pixels)
    if reset_button:
        st.session_state.active_image = st.session_state.original_image

    image = st.session_state.active_image


    # Resize image based on DPI
    dpi = params.get("dpi", 200)
    target_max_mm = params.get("target_max", 100.0)
    ppmm = dpi / 25.4
    params['ppmm'] = ppmm  # Add ppmm to params
    new_size_px = int(target_max_mm * ppmm)

    width, height = image.size
    if width > height:
        new_width = new_size_px
        new_height = int(new_width * height / width)
    else:
        new_height = new_size_px
        new_width = int(new_height * width / height)

    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS).convert("RGBA")
    
    # --- Drawable Canvas ---
    st.header("Drawing Tools")

    # Pipette color picker
    if 'picked_color' not in st.session_state:
        st.session_state.picked_color = '#000000'
    if 'pipette_active' not in st.session_state:
        st.session_state.pipette_active = False
    if 'canvas_json_data' not in st.session_state:
        st.session_state.canvas_json_data = None

    # Drawing controls
    
    # Color picker with pipette
    #st.sidebar.markdown("##### Color Picker")
    col1, col2,col3 = st.columns(3)
    stroke_width = col1.slider("Stroke Width", min_value=1, max_value=50, value=5, step=1, help="Width of the drawing stroke in pixels")
    with col3:
        if st.button("💧 Pick color from last stroke", help="Draw on the image to pick color from there and get the last stroke deleted"):
            st.session_state.pipette_active = not st.session_state.pipette_active
    with col2:
        st.session_state.picked_color = st.color_picker("Color", st.session_state.picked_color, label_visibility="collapsed")

    if st.session_state.pipette_active:
        st.info("Pipette is active. Draw on the image to pick the average color of your stroke.")

    # Get screen dimensions with a fallback
    screen_size = st_dimensions(key="screen_size")
    max_w_dynamic = screen_size['width'] * 0.9 if screen_size and 'width' in screen_size else 1100
    max_h_dynamic = screen_size['height'] * 0.7 if screen_size and 'height' in screen_size else 700

    w, h = image.size

    # Calculate display scaling
    scale = min(1.0, min(max_w_dynamic / w, max_h_dynamic / h)) if w > 0 and h > 0 else 1.0
    display_w = int(w * scale)
    display_h = int(h * scale)

    st.markdown(f"**Interactive Canvas (Resolution: {w} × {h}px, Displayed as: {display_w} x {display_h}px)**")

    # Scale stroke width for display on the scaled canvas
    stroke_width_px_display = max(1, int(stroke_width * scale))

    # Resize background for display
    pil_bg_for_canvas = image
    if scale < 1.0:
        pil_bg_for_canvas = image.resize((display_w, display_h), Image.Resampling.LANCZOS)

    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=stroke_width_px_display,
        stroke_color=st.session_state.picked_color,
        background_image=pil_bg_for_canvas,
        update_streamlit=True,
        height=display_h,
        width=display_w,
        drawing_mode="freedraw",
        key="canvas",
        display_toolbar=True,
        initial_drawing=st.session_state.canvas_json_data,
    )

    if canvas_result.json_data and st.session_state.pipette_active:
        try:
            # Get the last drawn path
            path = canvas_result.json_data['objects'][-1]['path']
            
            # Collect colors along the path
            colors = []
            for point in path:
                # Scale the coordinates back to the original image size
                original_x = int(point[1] / scale)
                original_y = int(point[2] / scale)
                if 0 <= original_x < w and 0 <= original_y < h:
                    colors.append(image.getpixel((original_x, original_y)))
            
            if colors:
                # Calculate the average color
                avg_color = np.mean(colors, axis=0).astype(int)
                
                # Convert to hex, handling both RGB and Grayscale results
                if isinstance(avg_color, (int, float, np.number)):
                    hex_color = f"#{int(avg_color):02x}{int(avg_color):02x}{int(avg_color):02x}"
                elif len(avg_color) >= 3: # RGB/RGBA
                    hex_color = '#%02x%02x%02x' % tuple(avg_color[:3])
                else: # Grayscale
                    val = int(avg_color[0])
                    hex_color = f"#{val:02x}{val:02x}{val:02x}"
                
                st.session_state.picked_color = hex_color
            
            # Remove the last stroke
            if canvas_result.json_data['objects']:
                canvas_result.json_data['objects'].pop()
                st.session_state.canvas_json_data = canvas_result.json_data

            st.session_state.pipette_active = False
            st.rerun()

        except (IndexError, TypeError, KeyError) as e:
            st.warning(f"Could not pick color: {e}. Please try again.")

    if st.button("Process Image and Generate Heightmap"):
        with st.spinner('Processing image...'):
            # Composite the drawing on the background image
            drawing_layer_rgba = canvas_result.image_data
            background_gray = np.array(image.convert('L'))

            if drawing_layer_rgba is not None:
                # The drawing layer is at the display resolution, scale it up to the original image resolution
                # Ensure the data type is uint8 before resizing
                drawing_layer_rgba_uint8 = drawing_layer_rgba.astype(np.uint8)
                drawing_layer_rgba = cv2.resize(
                    drawing_layer_rgba_uint8,
                    (w, h), # (width, height) of the original image
                    interpolation=cv2.INTER_LINEAR
                )
                background_rgb = cv2.cvtColor(background_gray, cv2.COLOR_GRAY2RGB)
                alpha = drawing_layer_rgba[:, :, 3] / 255.0
                alpha_mask = np.stack([alpha, alpha, alpha], axis=-1)
                drawing_layer_rgb = drawing_layer_rgba[:, :, :3]
                composite_rgb = (drawing_layer_rgb * alpha_mask + background_rgb * (1.0 - alpha_mask)).astype(np.uint8)
                final_image_np = cv2.cvtColor(composite_rgb, cv2.COLOR_RGB2GRAY)
                final_image = Image.fromarray(final_image_np)
            else:
                st.warning("No canvas data received. Proceeding with the unmodified image.")
                final_image = image

            heightmap_array, insert_map_array= process_image(final_image, params)
            st.session_state.heightmap_array = heightmap_array
            st.session_state.insert_map_array = insert_map_array


            # Get heightmap dimensions for centering
            h, w = heightmap_array.shape
            st.session_state.center_point = np.array([w / 2, h / 2, 0])

            st.success("Image processing complete!")

    if st.session_state.heightmap_array is not None:
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.header("Heightmap")
            st.image(st.session_state.heightmap_array, caption='Generated Heightmap', use_container_width=True)
            buf = io.BytesIO()
            Image.fromarray(st.session_state.heightmap_array).save(buf, format="PNG")
            st.download_button("Download Heightmap", buf.getvalue(), "heightmap.png", "image/png", key="dl_heightmap")

        with col2:
            st.header("Insert Map")
            st.image(st.session_state.insert_map_array, caption='Generated Insert Map', use_container_width=True)
            buf = io.BytesIO()
            Image.fromarray(st.session_state.insert_map_array).save(buf, format="PNG")
            st.download_button("Download Insert Map", buf.getvalue(), "insert_map.png", "image/png", key="dl_insertmap")

        col1, col2 = st.columns(2)
        with col1:
            generate_cutter = st.button("Generate 3D Cutter", use_container_width=True)
        with col2:
            generate_insert = st.button("Generate Insert", use_container_width=True)

        if 'cutter_mesh' not in st.session_state:
            st.session_state.cutter_mesh = None
        if 'insert_mesh' not in st.session_state:
            st.session_state.insert_mesh = None

        if generate_cutter:
            st.header("3D Model Preview")

            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Step 1/3: Generating cutter mesh...")
            mesh = generate_mesh(st.session_state.heightmap_array, params)
            progress_bar.progress(33)

            status_text.text("Step 2/3: Scaling and centering mesh...")
            sct,ct= get_transforms(st.session_state.heightmap_array, params)
            st.session_state.scale_transform = sct
            st.session_state.center_transform = ct
            cutter_mesh = scale_and_center_mesh(mesh, st.session_state.scale_transform, st.session_state.center_transform)
            st.session_state.cutter_mesh = cutter_mesh
            progress_bar.progress(66)

            original_faces = len(cutter_mesh.faces)
            st.write(f"Original face count: {original_faces}")

            status_text.text("Step 3/3: Finalizing mesh...")
            progress_bar.progress(100)
            status_text.text("Done!")
            progress_bar.progress(100)

        if generate_insert:
            st.header("3D Model Preview")

            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Step 1/3: Generating insert mesh...")
            insert_params = params.copy()
            insert_params['h_max'] = params.get('h_max', 15.0) + 1.0 +params.get('h_rim', 2.0) # Add extra height for insert            
            insert_mesh_raw = generate_mesh(st.session_state.insert_map_array, insert_params)

            progress_bar.progress(33)

            status_text.text("Step 2/3: Scaling and centering mesh...")
            sct,ct= get_transforms(st.session_state.heightmap_array, insert_params)
            st.session_state.scale_transform = sct
            st.session_state.center_transform = ct            
            insert_mesh = scale_and_center_mesh(insert_mesh_raw, st.session_state.scale_transform, st.session_state.center_transform)
            st.session_state.insert_mesh = insert_mesh
            progress_bar.progress(66)

            original_faces = len(insert_mesh.faces)
            st.write(f"Original face count: {original_faces}")

            status_text.text("Step 3/3: Finalizing mesh...")
            progress_bar.progress(100)
            status_text.text("Done!")
            progress_bar.progress(100)

        # --- 3D Preview and Download ---
        if st.session_state.cutter_mesh or st.session_state.insert_mesh:
            st.subheader("Interactive 3D Preview")

            if "output_filename" not in st.session_state:
                st.session_state.output_filename = "cookie_cutter.stl"
            st.session_state.output_filename = st.text_input("Filename", value=st.session_state.output_filename)

            # Define columns for the previews
            col1, col2, col3 = st.columns(3)

            # Cutter Preview
            with col1:
                if st.session_state.cutter_mesh:
                    st.subheader("Cutter")
                    plotter_cutter = pv.Plotter(window_size=[400, 400], border=False)
                    plotter_cutter.add_mesh(pv.wrap(st.session_state.cutter_mesh), name='cutter', color='lightblue', smooth_shading=True, specular=0.5, ambient=0.3)
                    plotter_cutter.view_isometric()
                    plotter_cutter.background_color = 'white'
                    stpyvista(plotter_cutter, key="pv_cutter")

                    with io.BytesIO() as f:
                        st.session_state.cutter_mesh.export(f, file_type='stl'); f.seek(0)
                        stl_data = f.read()
                    st.download_button(label="📥 Download Cutter STL", data=stl_data, file_name=f"cutter_{st.session_state.output_filename}", mime="model/stl", use_container_width=True)

            # Insert Preview
            with col2:
                if st.session_state.insert_mesh:
                    st.subheader("Insert")
                    plotter_insert = pv.Plotter(window_size=[400, 400], border=False)
                    plotter_insert.add_mesh(pv.wrap(st.session_state.insert_mesh), name='insert', color='lightgreen', smooth_shading=True, specular=0.5, ambient=0.3)
                    plotter_insert.view_isometric()
                    plotter_insert.background_color = 'white'
                    stpyvista(plotter_insert, key="pv_insert")

                    with io.BytesIO() as f:
                        st.session_state.insert_mesh.export(f, file_type='stl'); f.seek(0)
                        stl_data = f.read()
                    st.download_button(label="📥 Download Insert STL", data=stl_data, file_name=f"insert_{st.session_state.output_filename}", mime="model/stl", use_container_width=True)

            # Both Preview
            with col3:
                if st.session_state.cutter_mesh and st.session_state.insert_mesh:
                    st.subheader("Both")
                    plotter_both = pv.Plotter(window_size=[400, 400], border=False)
                    plotter_both.add_mesh(pv.wrap(st.session_state.cutter_mesh), name='cutter', color='lightblue', smooth_shading=True, specular=0.5, ambient=0.3)
                    plotter_both.add_mesh(pv.wrap(st.session_state.insert_mesh), name='insert', color='lightgreen', smooth_shading=True, specular=0.5, ambient=0.3)
                    plotter_both.view_isometric()
                    plotter_both.background_color = 'white'
                    stpyvista(plotter_both, key="pv_both")

                    with io.BytesIO() as f:
                        combined_mesh = trimesh.util.concatenate(st.session_state.cutter_mesh, st.session_state.insert_mesh)
                        combined_mesh.export(f, file_type='stl'); f.seek(0)
                        stl_data = f.read()
                    st.download_button(label="📥 Download Combined STL", data=stl_data, file_name=f"combined_{st.session_state.output_filename}", mime="model/stl", use_container_width=True)

        elif generate_cutter or generate_insert:
            st.error("Could not generate a 3D model. This can happen if the image is empty or too simple. Try a different image or adjust the processing parameters.")
else:
    st.info("Upload an image to get started.")
