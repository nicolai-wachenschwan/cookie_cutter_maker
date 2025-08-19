import numpy as np
import cv2
from PIL import Image
from skimage.morphology import remove_small_objects

def find_neighbour_contour(image, row, x, prev=True):
    """Finds the next non-zero pixel in a row, starting from x."""
    row_vals = image[row]
    if not prev:
        non_zero = np.where(row_vals[x:] > 0)[0]
        return x + non_zero[0] if len(non_zero) > 0 else image.shape[1] - 1
    else:
        non_zero = np.where(np.flip(row_vals[:x]) > 0)[0]
        return x - non_zero[0] - 1 if len(non_zero) > 0 else 0

def denoise_image(img):
    """Denoise a grayscale image."""
    return cv2.fastNlMeansDenoising(img.astype(np.uint8))

def process_image(pil_image, parameters):
    """Processes an image to generate a heightmap and an insert map."""
    # Convert to Grayscale for better performance and consistency
    if pil_image.mode != 'L':
        pil_image = pil_image.convert('L')
    img_np = np.array(pil_image)

    # Denoise if it's a JPEG
    if hasattr(pil_image, 'format') and pil_image.format in ['JPEG', 'JPG']:
        img_np = denoise_image(img_np)

    # Apply Otsu's thresholding
    # We use THRESH_BINARY_INV because we want the object to be white (255).
    _, binary_image = cv2.threshold(img_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove small islands of noise - increased threshold
    min_size = int(0.01 * binary_image.shape[0] * binary_image.shape[1])
    bool_image = binary_image.astype(bool)
    cleaned_image = remove_small_objects(bool_image, min_size=min_size)
    binary_image = cleaned_image.astype(np.uint8) * 255

    outside_comp = binary_image.copy()
    cv2.floodFill(outside_comp, None, (0, 0), 220)
    _, outside = cv2.threshold(outside_comp, 128, 255, cv2.THRESH_BINARY)

    ox, oy, ow, oh = cv2.boundingRect(cv2.bitwise_not(outside))
    ppmm = int(max(ow, oh) / parameters["target_max"]) if parameters["target_max"] > 0 else 1
    if ppmm == 0: ppmm = 1
    parameters["ppmm"] = ppmm

    rim_dilation_distance = int(ppmm * parameters["w_rim"])
    if rim_dilation_distance < 1: rim_dilation_distance = 1
    rim_kernel = np.ones((rim_dilation_distance, rim_dilation_distance), np.uint8)
    dil_4_rim = cv2.dilate(cv2.bitwise_not(binary_image), rim_kernel, iterations=1)
    rim_binary = cv2.bitwise_and(dil_4_rim, outside)
    _, rim_binary = cv2.threshold(rim_binary, 200, 255, cv2.THRESH_BINARY)

    thresh = cv2.bitwise_not(binary_image)
    dilation_distance = int(ppmm * parameters["min_wall"])
    if dilation_distance < 1: dilation_distance = 1
    kernel = np.ones((dilation_distance, dilation_distance), np.uint8)
    dilated_image = cv2.dilate(thresh, kernel, iterations=1)
    _, contours_binary = cv2.threshold(dilated_image, 200, 255, cv2.THRESH_BINARY)

    h_max = 255
    color_outer_contour = h_max
    color_inner = int(parameters["h_mark"] / parameters["h_max"] * h_max)
    color_rim = int(parameters["h_rim"] / parameters["h_max"] * h_max)

    composite = np.zeros_like(binary_image, np.uint8)
    if color_rim > 0:
        composite = cv2.add(composite, cv2.threshold(rim_binary, 1, color_rim, cv2.THRESH_BINARY)[1])
    if color_inner > 0:
        inner_mask = cv2.bitwise_and(contours_binary, cv2.bitwise_not(rim_binary))
        composite = cv2.add(composite, cv2.threshold(inner_mask, 1, color_inner, cv2.THRESH_BINARY)[1])
    if color_outer_contour > 0:
        outer_contours = cv2.bitwise_and(contours_binary, rim_binary) # Simplified logic
        composite = cv2.add(composite, cv2.threshold(outer_contours, 1, color_outer_contour, cv2.THRESH_BINARY)[1])

    # --- Insert Map Generation ---
    clearance = int(1 * ppmm)
    if clearance < 1: clearance = 1
    color_insert = int(parameters["h_rim"] / parameters["h_max"] * 255)
    color_inner_insert = color_inner - color_insert if color_inner > color_insert else 10

    clearance_kernel = np.ones((clearance, clearance))
    extra_dil_contours = cv2.dilate(contours_binary, clearance_kernel, iterations=1)
    inner_insert = cv2.bitwise_and(cv2.bitwise_not(extra_dil_contours), cv2.bitwise_not(outside))

    colored_insert = np.zeros_like(inner_insert)
    if color_insert > 0:
        _, colored_insert = cv2.threshold(inner_insert, 1, color_insert, cv2.THRESH_BINARY)

    inner_inner_insert = cv2.erode(colored_insert, clearance_kernel, iterations=1)
    colored_inner_insert = np.zeros_like(inner_inner_insert)
    if color_inner_insert > 0:
        _, colored_inner_insert = cv2.threshold(inner_inner_insert, 1, color_inner_insert, cv2.THRESH_BINARY)

    insert_composite = cv2.add(colored_insert, colored_inner_insert)

    return composite, insert_composite
