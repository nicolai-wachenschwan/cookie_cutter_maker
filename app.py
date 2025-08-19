import streamlit as st
import numpy as np
from PIL import Image
import cv2
import trimesh
import io
import pandas as pd
import pydeck as pdk

def get_binary_image(pil_img):
    """
    Processes the uploaded image to create a clean binary image.
    - Converts to CMYK and uses the K channel.
    - Applies Otsu's thresholding.
    """
    img_np = np.array(pil_img.convert('RGB'))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Denoise if the image is a JPG
    if pil_img.format == 'JPEG':
        img_bgr = cv2.fastNlMeansDenoisingColored(img_bgr, None, 10, 10, 7, 21)

    img_float = img_bgr.astype(np.float32) / 255.
    k_channel = (1 - np.max(img_float, axis=2)) * 255
    k_channel_uint8 = k_channel.astype(np.uint8)

    # Invert the k_channel so that dark lines become bright
    k_channel_inv = cv2.bitwise_not(k_channel_uint8)

    # Apply Otsu's thresholding
    _, binary_image = cv2.threshold(k_channel_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary_image, k_channel_uint8

def generate_heightmap(binary_image, params):
    """
    Generates a heightmap from the binary image based on the logic from the original script.
    """
    h, w = binary_image.shape
    ppmm = max(h, w) / params['target_size']

    # --- Create the stabilizing rim ---
    rim_width_px = int(ppmm * params['rim_width'])
    rim_kernel = np.ones((rim_width_px, rim_width_px), np.uint8)

    # Invert binary image for dilation (dilate white areas)
    dilated_for_rim = cv2.dilate(binary_image, rim_kernel, iterations=1)

    # --- Flood fill to find the outside area ---
    outside_mask = dilated_for_rim.copy()
    cv2.floodFill(outside_mask, None, (0, 0), 128)

    # Rim is the area that was dilated but is not part of the original shape
    rim = cv2.inRange(outside_mask, 127, 129)
    rim = cv2.bitwise_and(rim, cv2.bitwise_not(binary_image))

    # --- Thicken the main contours for the cutting edge ---
    wall_thickness_px = int(ppmm * params['wall_thickness'])
    if wall_thickness_px < 1: wall_thickness_px = 1
    wall_kernel = np.ones((wall_thickness_px, wall_thickness_px), np.uint8)
    cutting_edges = cv2.dilate(binary_image, wall_kernel, iterations=1)

    # --- Composite the heightmap ---
    # Define heights (grayscale values)
    h_max = 255 # Cutting edge
    h_rim = int(params['base_height'] / params['cutter_height'] * 255)
    h_emboss = int(params['emboss_height'] / params['cutter_height'] * 255)

    # Start with a black canvas
    heightmap = np.zeros_like(binary_image, dtype=np.uint8)

    # Add the rim
    heightmap[rim > 0] = h_rim
    # Add the cutting edges (which also contain the inner embossed parts)
    heightmap[cutting_edges > 0] = h_emboss
    # The original binary image defines the highest points (the cutting part of the line)
    heightmap[binary_image > 0] = h_max

    return heightmap, rim, cutting_edges


st.set_page_config(layout="wide")
st.title("🍪 Cookie Cutter Generator")
st.write("Upload an image, adjust the parameters, and generate a 3D model of a cookie cutter, ready for 3D printing.")

st.sidebar.header("Parameters")

uploaded_file = st.sidebar.file_uploader("1. Upload Image", type=["jpg", "jpeg", "png"])

# Group parameters for clarity
st.sidebar.subheader("Sizing")
target_size = st.sidebar.slider("Target Size (mm)", 50, 200, 100)
wall_thickness = st.sidebar.slider("Cutting Edge Width (mm)", 0.5, 2.0, 1.0, 0.1)

st.sidebar.subheader("Heights")
cutter_height = st.sidebar.slider("Total Height (mm)", 5, 25, 15)
emboss_height = st.sidebar.slider("Emboss Depth (mm)", 1, 10, 5)
base_height = st.sidebar.slider("Base/Rim Height (mm)", 1, 5, 2)
st.sidebar.subheader("Structure")
rim_width = st.sidebar.slider("Rim Width (mm)", 2, 10, 5)


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button("Generate Cookie Cutter"):
        params = {
            'target_size': target_size,
            'wall_thickness': wall_thickness,
            'cutter_height': cutter_height,
            'emboss_height': emboss_height,
            'base_height': base_height,
            'rim_width': rim_width
        }
        with st.spinner('Processing image...'):
            binary_image, k_channel = get_binary_image(image)
            heightmap, rim, cutting_edges = generate_heightmap(binary_image, params)

            st.subheader("Image Processing Steps")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.image(k_channel, caption='K-Channel', use_column_width=True)
            with col2:
                st.image(binary_image, caption='1. Binary Shapes', use_column_width=True)
            with col3:
                st.image(cutting_edges, caption='2. Thickened Edges', use_column_width=True)
            with col4:
                st.image(rim, caption='3. Rim Area', use_column_width=True)

            st.subheader("Final Heightmap")
            st.image(heightmap, caption='Grayscale Heightmap for 3D Model', use_column_width=True)

            st.subheader("3D Model Preview")
            with st.spinner('Generating 3D model... This may take a moment.'):
                stl_data, preview_points = generate_3d_model(heightmap, params)

                # Create a PyDeck chart for 3D visualization
                view_state = pdk.ViewState(
                    latitude=preview_points['y'].mean(),
                    longitude=preview_points['x'].mean(),
                    zoom=5, # This will need adjustment depending on model size
                    pitch=50)

                layer = pdk.Layer(
                    'PointCloudLayer',
                    data=preview_points,
                    get_position='[x, y, z]',
                    get_color='[255, 140, 0, 160]', # Amber color
                    get_size=1,
                    size_scale=max(params['target_size'] / 80, 1.0)
                )

                r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "x: {x}, y: {y}, z: {z}"})
                st.pydeck_chart(r)

                st.success("3D model generated successfully!")
                st.download_button(
                    label="Download STL file",
                    data=stl_data,
                    file_name="cookie_cutter.stl",
                    mime="model/stl"
                )

else:
    st.info("Please upload an image to start.")

def generate_3d_model(heightmap, params):
    """
    Generates a 3D model from the heightmap using voxelization with Trimesh.
    Returns the STL data as bytes and a DataFrame of sample points for preview.
    """
    # --- 1. Create a 3D voxel matrix from the 2D heightmap ---
    padded_map = np.pad(heightmap, pad_width=1, mode='constant', constant_values=0)
    matrix = np.zeros((padded_map.shape[0], padded_map.shape[1], 256), dtype=bool)
    for r in range(padded_map.shape[0]):
        for c in range(padded_map.shape[1]):
            height = padded_map[r, c]
            if height > 0:
                matrix[r, c, :height] = True

    # --- 2. Voxel Grid to Mesh ---
    voxel_grid = trimesh.voxel.VoxelGrid(matrix)
    mesh = voxel_grid.marching_cubes

    # --- 3. Scale and Center the Mesh ---
    ppmm = max(heightmap.shape) / params['target_size']
    pixel_width_mm = 1.0 / ppmm
    z_scale_mm = params['cutter_height'] / 255.0
    scale_transform = trimesh.transformations.scale_matrix([pixel_width_mm, pixel_width_mm, z_scale_mm])

    # Center the mesh at the origin
    center_transform = trimesh.transformations.translation_matrix(-mesh.bounds.mean(axis=0))

    mesh.apply_transform(center_transform)
    mesh.apply_transform(scale_transform)
    mesh.process()

    # --- 4. Get Data for Preview and Download ---
    # Export STL to in-memory file
    with io.BytesIO() as f:
        mesh.export(f, file_type='stl')
        f.seek(0)
        stl_data = f.read()

    # Get sample points for preview
    num_points = 5000
    if len(mesh.vertices) > num_points:
        sample_vertices = mesh.vertices[np.random.choice(len(mesh.vertices), num_points, replace=False)]
    else:
        sample_vertices = mesh.vertices

    preview_points = pd.DataFrame(sample_vertices, columns=['x', 'y', 'z'])

    return stl_data, preview_points
