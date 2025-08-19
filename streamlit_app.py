import streamlit as st
from PIL import Image
import image_processing
import numpy as np
import json
import io
import cv2
from pyvista.trame import st_plotter

st.set_page_config(layout="wide")

st.title("Cookie Cutter Generator")

st.sidebar.header("Processing Parameters")

params = {
    "target_max": st.sidebar.number_input("Target Size [mm]", min_value=10.0, value=100.0, step=1.0),
    "min_wall": st.sidebar.number_input("Min Wall Thickness [mm]", min_value=0.1, value=1.0, step=0.1),
    "h_max": st.sidebar.number_input("Total Height [mm]", min_value=1.0, value=10.0, step=0.5),
    "h_mark": st.sidebar.number_input("Inner Lines Height [mm]", min_value=0.1, value=9.0, step=0.1),
    "h_inner": st.sidebar.number_input("Small Areas Height [mm]", min_value=0.1, value=8.0, step=0.1),
    "h_rim": st.sidebar.number_input("Rim Height [mm]", min_value=0.1, value=1.5, step=0.1),
    "w_rim": st.sidebar.number_input("Rim Width [mm]", min_value=0.1, value=5.0, step=0.1),
    "small_fill": st.sidebar.number_input("Fill Areas smaller than [mm^2]", min_value=0.0, value=50.0, step=1.0)
}

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None:
    if st.button("Process Image"):
        try:
            # To use the file in memory with opencv and pillow
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

            # We need to write it to a temporary file because process_image takes a path
            # Alternatively, we could modify process_image to take a file-like object or array
            temp_file_path = f"/tmp/{uploaded_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(file_bytes)

            with st.spinner('Processing image... this may take a moment.'):
                heightmap_array, insert_map_array = image_processing.process_image(temp_file_path, params)

            st.success("Processing complete!")

            col1, col2 = st.columns(2)

            with col1:
                st.header("Heightmap")
                st.image(heightmap_array, caption='Generated Heightmap', use_column_width=True)

                # Create a download button for the heightmap
                heightmap_img = Image.fromarray(heightmap_array)
                buf = io.BytesIO()
                heightmap_img.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download Heightmap",
                    data=byte_im,
                    file_name="heightmap.png",
                    mime="image/png"
                )

            with col2:
                st.header("Insert Map")
                st.image(insert_map_array, caption='Generated Insert Map', use_column_width=True)

                # Create a download button for the insert map
                insert_map_img = Image.fromarray(insert_map_array)
                buf = io.BytesIO()
                insert_map_img.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download Insert Map",
                    data=byte_im,
                    file_name="insert_map.png",
                    mime="image/png"
                )

            # Save config and provide download
            config_str = json.dumps(params, indent=4)
            st.sidebar.download_button(
                label="Download Config",
                data=config_str,
                file_name="cookie_config.json",
                mime="application/json"
            )

            st.header("3D Visualization")
            with st.spinner("Generating 3D model..."):
                # The generate_3d_model function needs to be adapted for streamlit
                # For now, let's assume it returns a plotter object
                plotter = image_processing.generate_3d_model(heightmap_array)
                if plotter:
                    st_plotter(plotter, key="pv_plotter")

        except Exception as e:
            st.error(f"An error occurred during image processing: {e}")
        finally:
            cv2.destroyAllWindows()
else:
    st.info("Upload an image to get started.")
