import cv2 as cv
import numpy as np
from PIL import Image
import pyvista as pv
from image_helper_functions import denoise_image

def find_neighbour_contour(image, row, x, prev=True):
    """
    Finds the next non-zero pixel in a row, starting from x.
    Searches backwards if prev is True.
    """
    row_vals = image[row]
    if not prev:
        # Search forward from x
        non_zero = np.where(row_vals[x:] > 0)[0]
        if len(non_zero) > 0:
            return x + non_zero[0]
        return image.shape[1] - 1
    else:
        # Search backward from x
        non_zero = np.where(np.flip(row_vals[:x]) > 0)[0]
        if len(non_zero) > 0:
            return x - non_zero[0] -1
        return 0

def process_image(image_path, parameters):
    """
    Processes an image to generate a heightmap for a cookie cutter.

    Args:
        image_path (str): Path to the input image.
        parameters (dict): A dictionary of processing parameters.

    Returns:
        tuple: A tuple containing the heightmap and insert_map as numpy arrays.
    """
    input_img = Image.open(image_path).convert('L')
    gray = np.asarray(input_img)

    if image_path.lower().endswith((".jpg", ".jpeg")):
        gray = denoise_image(gray.astype(np.uint8))

    _, binary_image = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)
    #cv.imshow("bin", binary_image)

    # Find outside of the cutter by flood-filling from the corner
    outside_comp = binary_image.copy()
    seed_point = (0, 0)
    flood_color = 220
    cv.floodFill(outside_comp, None, seed_point, flood_color)
    _, outside = cv.threshold(outside_comp, 128, 255, cv.THRESH_BINARY)

    # Calculate pixel per mm
    (ox, oy, ow, oh) = cv.boundingRect(cv.bitwise_not(outside))
    ppmm = int(max(ow, oh) / parameters.get("target_max"))
    if ppmm == 0:
        ppmm = 1
    parameters["ppmm"] = ppmm

    # Create the rim
    rim_dilation_distance = int(ppmm * parameters.get("w_rim"))
    if rim_dilation_distance < 1: rim_dilation_distance = 1
    rim_kernel = np.ones((rim_dilation_distance, rim_dilation_distance), np.uint8)
    dil_4_rim = cv.dilate(cv.bitwise_not(binary_image), rim_kernel, iterations=1)
    rim_binary = cv.bitwise_and(dil_4_rim, outside)
    _, rim_binary = cv.threshold(rim_binary, 200, 255, cv.THRESH_BINARY)

    # Main contour thickening
    thresh = cv.bitwise_not(binary_image)
    dilation_distance = int(ppmm * parameters.get("min_wall"))
    if dilation_distance < 1:
        dilation_distance = 1
    kernel = np.ones((dilation_distance, dilation_distance), np.uint8)
    dilated_image = cv.dilate(thresh, kernel, iterations=1)
    _, contours_binary = cv.threshold(dilated_image, 200, 255, cv.THRESH_BINARY)

    # Mask for detection of outer contours vs inner
    outer_dilation_distance = int(dilation_distance * 2)
    if outer_dilation_distance < 1:
        outer_dilation_distance = 1
    outer_kernel = np.ones((outer_dilation_distance, outer_dilation_distance), np.uint8)
    outer_dilated = cv.dilate(rim_binary, outer_kernel, iterations=1)
    outer_mask = cv.bitwise_and(outer_dilated, cv.bitwise_not(outside))
    _, outer_mask = cv.threshold(outer_mask, 200, 255, cv.THRESH_BINARY)

    outer_contours = cv.bitwise_and(contours_binary, outer_mask)
    inner_mask = cv.bitwise_and(cv.bitwise_and(cv.bitwise_not(outer_mask), contours_binary), cv.bitwise_not(outside))

    # Composite the image
    color_outer_contour = 255
    color_inner = int(parameters.get("h_mark") / parameters.get("h_max") * 255)
    color_rim = int(parameters.get("h_rim") / parameters.get("h_max") * 255)
    color_connector = color_rim
    color_small_contour = int(parameters.get("h_inner") / parameters.get("h_max") * 255)
    small_thres = parameters.get("small_fill")

    composite = np.zeros_like(binary_image, np.uint8)
    if color_outer_contour > 0:
        _, colored_outer = cv.threshold(outer_contours, 1, color_outer_contour, cv.THRESH_BINARY)
        composite = cv.add(composite, colored_outer)
    if color_inner > 0:
        _, colored_inner = cv.threshold(inner_mask, 1, color_inner, cv.THRESH_BINARY)
        composite = cv.add(composite, colored_inner)
    if color_rim > 0:
        _, colored_rim = cv.threshold(rim_binary, 1, color_rim, cv.THRESH_BINARY)
        composite = cv.add(composite, colored_rim)

    # Connect loose parts
    small_contour_mask = np.zeros_like(contours_binary)
    connectors = np.zeros_like(contours_binary)
    contours, hierarchy = cv.findContours(contours_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        for i in range(len(contours)):
            area_mm2 = cv.contourArea(contours[i]) / (ppmm**2 if ppmm != 0 else 1)
            if area_mm2 < small_thres:
                cv.drawContours(small_contour_mask, contours, i, color_small_contour, -1)

            parent = hierarchy[0][i][3]
            if parent != -1:
                (x, y, w, h) = cv.boundingRect(contours[i])
                if h > 1:
                    for idr in range(y, y + h):
                        prev_x = find_neighbour_contour(contours_binary, idr, x)
                        next_x = find_neighbour_contour(contours_binary, idr, x + w, prev=False)
                        if next_x > prev_x:
                            cv.line(connectors, (prev_x, idr), (next_x, idr), color_connector, 1)

    small_contour_mask = cv.bitwise_and(small_contour_mask, cv.bitwise_not(contours_binary))
    composite = cv.add(composite, connectors)
    composite = cv.add(composite, small_contour_mask)
    #cv.imshow('All',composite)

    # Add insert
    clearence = int(1 * ppmm)
    if clearence < 1: clearence = 1
    color_insert = int(parameters.get("h_rim") / parameters.get("h_max") * 255)
    color_inner_insert = color_inner - color_insert if color_inner > color_insert else 10

    clearence_kernel = np.ones((clearence,clearence))
    extra_dil_contours = cv.dilate(contours_binary, clearence_kernel, iterations=1)
    inner_insert = cv.bitwise_and(cv.bitwise_not(extra_dil_contours), cv.bitwise_not(outside))

    colored_insert = np.zeros_like(inner_insert)
    if color_insert > 0:
        _, colored_insert = cv.threshold(inner_insert, 1, color_insert, cv.THRESH_BINARY)

    inner_inner_insert = cv.erode(colored_insert, clearence_kernel, iterations=1)

    colored_inner_insert = np.zeros_like(inner_inner_insert)
    if color_inner_insert > 0:
        _, colored_inner_insert = cv.threshold(inner_inner_insert, 1, color_inner_insert, cv.THRESH_BINARY)

    insert_composite = cv.add(colored_insert, colored_inner_insert)
    #cv.imshow("insert",insert_composite)

    return composite, insert_composite


def generate_3d_model(heightmap_array):
    """
    Generates a 3D model from a heightmap array for Streamlit.

    Args:
        heightmap_array (np.ndarray): The heightmap data.

    Returns:
        pyvista.Plotter: The plotter object for embedding in Streamlit, or None.
    """
    if heightmap_array.size == 0:
        print("Heightmap is empty, cannot generate 3D model.")
        return None

    # Create a structured grid
    h, w = heightmap_array.shape
    x = np.arange(w)
    y = np.arange(h)
    x, y = np.meshgrid(x, y)
    z = heightmap_array.astype(float)

    grid = pv.StructuredGrid(x, y, z)

    # Warp by scalar to give it height
    grid = grid.warp_by_scalar(factor=0.2)

    # Create a plotter and add the mesh
    plotter = pv.Plotter(window_size=[600, 400], border=False)
    plotter.add_mesh(grid, cmap='viridis', show_edges=True)
    plotter.show_grid()
    plotter.camera_position = 'xy'

    return plotter
