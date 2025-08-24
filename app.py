import streamlit as st
import numpy as np
from PIL import Image
import io
import json
import pyvista as pv
from stpyvista import stpyvista
from stpyvista.utils import start_xvfb

# Import refactored logic
from heightmap import process_image
from mesh import create_voxel_matrix, create_mesh_from_voxel_matrix, scale_and_center_mesh, decimate_mesh

# --- Setup ---
start_xvfb()
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
    "decimate": st.sidebar.checkbox("Decimate Mesh", value=True),
    "decimate_faces": st.sidebar.number_input("Target Face Count", min_value=100, value=50000, step=100),
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
        heightmap_array, insert_map_array = process_image(image, params)
        st.session_state.heightmap_array = heightmap_array
        st.session_state.insert_map_array = insert_map_array
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

    if st.button("Generate 3D Mesh"):
        st.header("3D Model Preview")

        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Step 1/4: Creating voxel matrix...")
        matrix = create_voxel_matrix(st.session_state.heightmap_array, params)
        progress_bar.progress(25)

        status_text.text("Step 2/4: Creating mesh from voxel matrix...")
        mesh = create_mesh_from_voxel_matrix(matrix)
        progress_bar.progress(50)

        status_text.text("Step 3/4: Scaling and centering mesh...")
        generated_mesh = scale_and_center_mesh(mesh, params)
        progress_bar.progress(75)

        original_faces = len(generated_mesh.faces)

        if params["decimate"] and original_faces > 0:
            target_faces = params['decimate_faces']
            if original_faces > target_faces:
                target_reduction = 1.0 - (target_faces / original_faces)
                status_text.text(f"Step 4/4: Decimating mesh to {target_faces} faces...")
                generated_mesh = decimate_mesh(generated_mesh, target_reduction)
                decimated_faces = len(generated_mesh.faces)
                st.write(f"Mesh decimated from {original_faces} to {decimated_faces} faces.")
            else:
                status_text.text("Step 4/4: Skipping decimation (target faces >= original faces)...")
                st.write(f"Mesh has {original_faces} faces. Decimation was skipped.")
        else:
            status_text.text("Step 4/4: Skipping mesh decimation...")
            st.write(f"Mesh has {original_faces} faces. Decimation was skipped.")

        progress_bar.progress(100)
        status_text.text("Done!")

        if generated_mesh:
            st.subheader("Interactive 3D Preview")
            pv_mesh = pv.wrap(generated_mesh)
            if pv_mesh:
                plotter = pv.Plotter(window_size=[800, 600], border=False)
                plotter.add_mesh(pv_mesh, color='lightblue', smooth_shading=True, specular=0.5, ambient=0.3)
                plotter.view_isometric(); plotter.background_color = 'white'
                stpyvista(plotter, key="pv_viewer")
                st.subheader("Download")
                if "output_filename" not in st.session_state:
                    st.session_state.output_filename = "shadowboard.stl"
                st.session_state.output_filename = st.text_input("Filename", value=st.session_state.output_filename)
                with io.BytesIO() as f:
                    generated_mesh.export(f, file_type='stl'); f.seek(0)
                    stl_data = f.read()
                st.download_button(label="📥 Download STL File", data=stl_data, file_name=st.session_state.output_filename, mime="model/stl", use_container_width=True)
        else:
            st.error("Could not generate a 3D model from the image. This can happen if the image is empty or too simple. Try a different image or adjust the processing parameters.")
else:
    st.info("Upload an image to get started.")
