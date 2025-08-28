import streamlit as st
import numpy as np
from PIL import Image
import io
import json
import pyvista as pv
from stpyvista import stpyvista
from stpyvista.utils import start_xvfb
import trimesh

# Import refactored logic
from heightmap import process_image
from mesh import (
    generate_mesh,
    generate_insert_mesh,
    scale_and_center_mesh
)

# --- Setup ---
try:
    start_xvfb()
except Exception as e:
    st.warning(f"(this is ok for local runs) Could not start virtual framebuffer: {e}")    
st.set_page_config(layout="wide")
st.title("🍪 Advanced Cookie Cutter Generator")
st.write("Upload an image, adjust parameters, and generate a 3D model and insert for your cookie cutter.")


# --- Sidebar UI ---
st.sidebar.header("Processing Parameters")
params = {
    "target_max": st.sidebar.number_input("Target Size [mm]", min_value=10.0, value=100.0, step=1.0),
    "min_wall": st.sidebar.number_input("Cutting Edge Width [mm]", min_value=0.1, value=1.0, step=0.1),
    "h_max": st.sidebar.number_input("Total Height [mm]", min_value=1.0, value=15.0, step=0.5),
    "h_rim": st.sidebar.number_input("Base/Rim Height [mm]", min_value=0.1, value=2.0, step=0.1),
    "w_rim": st.sidebar.number_input("Rim Width [mm]", min_value=0.1, value=5.0, step=0.1),
    "height_dough_thickness": st.sidebar.number_input("Dough Thickness [mm]", min_value=0.1, value=2.0, step=0.1),
    "h_inner": st.sidebar.number_input("Inner Wall Height [mm]", min_value=0.1, value=3.0, step=0.1),
    "small_fill": st.sidebar.number_input("Small Area Fill Threshold [mm^2]", min_value=0.0, value=10.0, step=0.1),
    "dpi": st.sidebar.number_input("DPI", min_value=50, value=200, step=10),
}
# Create a copy of params for JSON export, excluding Streamlit UI elements
params_for_export = {k: v for k, v in params.items() if not hasattr(v, 'get')}
config_str = json.dumps(params_for_export, indent=4)
st.sidebar.download_button(
    label="Download Config", data=config_str, file_name="cookie_config.json", mime="application/json"
)

# --- Main Page UI ---
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "bmp"])

if 'heightmap_array' not in st.session_state:
    st.session_state.heightmap_array = None
if 'insert_map_array' not in st.session_state:
    st.session_state.insert_map_array = None
if 'outside_mask' not in st.session_state:
    st.session_state.outside_mask = None
if 'view_selection' not in st.session_state:
    st.session_state.view_selection = "Cutter"
if 'plotter' not in st.session_state:
    st.session_state.plotter = pv.Plotter(window_size=[800, 600], border=False)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

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

    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    st.image(image, caption='Uploaded and Resized Image', use_container_width=True)

    with st.spinner('Processing image...'):
        heightmap_array, insert_map_array, outside_mask = process_image(image, params)
        st.session_state.heightmap_array = heightmap_array
        st.session_state.insert_map_array = insert_map_array
        st.session_state.outside_mask = outside_mask

        # Get heightmap dimensions for centering
        h, w = heightmap_array.shape
        st.session_state.center_point = np.array([w / 2, h / 2, 0])

        st.success("Image processing complete!")

    # Display results
    col1, col2 = st.columns(2)
    with col1:
        st.header("Heightmap")
        st.image(heightmap_array, caption='Generated Heightmap', use_container_width=True)
        buf = io.BytesIO()
        Image.fromarray(heightmap_array).save(buf, format="PNG")
        st.download_button("Download Heightmap", buf.getvalue(), "heightmap.png", "image/png", key="dl_heightmap")

    with col2:
        st.header("Insert Map")
        st.image(insert_map_array, caption='Generated Insert Map', use_container_width=True)
        buf = io.BytesIO()
        Image.fromarray(insert_map_array).save(buf, format="PNG")
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
        cutter_mesh,scale_transform,center_transform = scale_and_center_mesh(mesh, params)
        st.session_state.scale_transform = scale_transform
        st.session_state.center_transform = center_transform
        st.session_state.cutter_mesh = cutter_mesh
        progress_bar.progress(66)

        original_faces = len(cutter_mesh.faces)
        st.write(f"Original face count: {original_faces}")

        status_text.text("Step 3/3: Finalizing mesh...")
        progress_bar.progress(100)
        st.session_state.view_selection = "Cutter"
        status_text.text("Done!")
        progress_bar.progress(100)

    if generate_insert:
        st.header("3D Model Preview")

        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Step 1/3: Generating insert mesh...")
        insert_mesh_raw = generate_insert_mesh(st.session_state.insert_map_array, st.session_state.outside_mask, params)

        progress_bar.progress(33)

        status_text.text("Step 2/3: Scaling and centering mesh...")
        insert_params = params.copy()
        insert_params['h_max'] = params.get('h_max', 15.0) + 1.0  # Add extra height for insert
        insert_mesh,_,_ = scale_and_center_mesh(insert_mesh_raw, insert_params, scale_transform=st.session_state.scale_transform, center_transform=st.session_state.center_transform)
        st.session_state.insert_mesh = insert_mesh
        progress_bar.progress(66)

        original_faces = len(insert_mesh.faces)
        st.write(f"Original face count: {original_faces}")

        status_text.text("Step 3/3: Finalizing mesh...")
        progress_bar.progress(100)
        st.session_state.view_selection = "Insert"
        status_text.text("Done!")
        progress_bar.progress(100)

    # --- 3D Preview and Download ---
    if st.session_state.cutter_mesh or st.session_state.insert_mesh:
        st.subheader("Interactive 3D Preview")

        st.radio(
            "Select view:",
            ("Cutter", "Insert", "Both"),
            horizontal=True,
            key="view_selection"
        )

        plotter = st.session_state.plotter
        plotter.clear()

        show_cutter = ("Cutter" in st.session_state.view_selection or "Both" in st.session_state.view_selection) and st.session_state.cutter_mesh is not None
        show_insert = ("Insert" in st.session_state.view_selection or "Both" in st.session_state.view_selection) and st.session_state.insert_mesh is not None

        if show_cutter:
            plotter.add_mesh(pv.wrap(st.session_state.cutter_mesh), name='cutter', color='lightblue', smooth_shading=True, specular=0.5, ambient=0.3)
        if show_insert:
            plotter.add_mesh(pv.wrap(st.session_state.insert_mesh), name='insert', color='lightgreen', smooth_shading=True, specular=0.5, ambient=0.3)

        if show_cutter or show_insert:
            plotter.view_isometric()
            plotter.background_color = 'white'
            stpyvista(plotter, key="pv_viewer")

        st.subheader("Download")
        if "output_filename" not in st.session_state:
            st.session_state.output_filename = "cookie_cutter.stl"
        st.session_state.output_filename = st.text_input("Filename", value=st.session_state.output_filename)

        # Placeholder for download logic
        if st.session_state.cutter_mesh:
            with io.BytesIO() as f:
                st.session_state.cutter_mesh.export(f, file_type='stl'); f.seek(0)
                stl_data = f.read()
            st.download_button(label="📥 Download Cutter STL", data=stl_data, file_name=f"cutter_{st.session_state.output_filename}", mime="model/stl", use_container_width=True)

        if st.session_state.insert_mesh:
            with io.BytesIO() as f:
                st.session_state.insert_mesh.export(f, file_type='stl'); f.seek(0)
                stl_data = f.read()
            st.download_button(label="📥 Download Insert STL", data=stl_data, file_name=f"insert_{st.session_state.output_filename}", mime="model/stl", use_container_width=True)

        if st.session_state.cutter_mesh and st.session_state.insert_mesh:
            with io.BytesIO() as f:
                combined_mesh = trimesh.util.concatenate(st.session_state.cutter_mesh, st.session_state.insert_mesh)
                combined_mesh.export(f, file_type='stl'); f.seek(0)
                stl_data = f.read()
            st.download_button(label="📥 Download Combined STL", data=stl_data, file_name=f"combined_{st.session_state.output_filename}", mime="model/stl", use_container_width=True)

    elif generate_cutter or generate_insert:
        st.error("Could not generate a 3D model. This can happen if the image is empty or too simple. Try a different image or adjust the processing parameters.")
else:
    st.info("Upload an image to get started.")
