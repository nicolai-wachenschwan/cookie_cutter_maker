import cv2
import numpy as np

def delete_small_islands(img, min_island_size=50, connectivity=4):
    """
    Remove small islands from a binary image

    Parameters:
    img= cv2 grayscale image

    min_island_size (int): Minimum number of pixels to keep an island
    connectivity (int): 4 or 8 connected components

    Returns:
    numpy.ndarray: Image with small islands removed
    """
    # Ensure binary image (in case it's not already binary)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Perform connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=connectivity
    )

    # Create a mask to keep only sufficiently large components
    # We start from 1 to skip the background component (index 0)
    mask = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_island_size:
            mask[labels == i] = 255

    return mask

def advanced_island_deletion(img,
                              min_island_size=50,
                              max_island_size=None,
                              connectivity=4):
    """
    Advanced island deletion with more sophisticated filtering

    Parameters:
    img=cv2 grayscale image
    min_island_size (int): Minimum number of pixels to keep an island
    max_island_size (int, optional): Maximum number of pixels to keep an island
    connectivity (int): 4 or 8 connected components

    Returns:
    numpy.ndarray: Image with filtered islands
    """

    # Ensure binary image
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Perform connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=connectivity
    )

    # Create a mask to keep components within size range
    mask = np.zeros_like(binary)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]

        # Check size conditions
        if min_island_size <= area:
            # Optional max size check
            if max_island_size is None or area <= max_island_size:
                mask[labels == i] = 255

    # Optional morphological operations to clean up
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

def denoise_image(img, method='nl_means'):
    """
    Denoise an image using various techniques

    Parameters:
    img = cv2 grayscale image
    method (str): Denoising method to use

    Returns:
    numpy.ndarray: Denoised image
    """

    if method == 'nl_means':
        # Non-local means denoising
        denoised = cv2.fastNlMeansDenoising(img)
    elif method == 'bilateral':
        # Bilateral filtering
        denoised = cv2.bilateralFilter(img, 9, 75, 75)
    else:
        raise ValueError("Invalid denoising method")

    return denoised

def apply_thresholding(img, method='adaptive'):
    """
    Apply thresholding to the image

    Parameters:
    img (numpy.ndarray): Grayscale input image
    method (str): Thresholding method ('adaptive' or 'otsu')

    Returns:
    numpy.ndarray: Thresholded image
    """

    # Apply thresholding
    if method == 'adaptive':
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # Block size
            2    # Constant subtracted from the mean
        )
    else:  # Otsu's method
        _, thresh = cv2.threshold(
            img,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    return thresh

# Example usage
if __name__ == '__main__':
    # This part is for demonstration and won't run when imported as a module.
    # You would need an actual image file named 'original_image.jpg' for this to work.
    try:
        input_image_arr = cv2.imread('original_image.jpg', cv2.IMREAD_GRAYSCALE)
        if input_image_arr is None:
            raise FileNotFoundError("original_image.jpg not found. Please provide an image for the example usage.")

        # Step 1: Denoise
        denoised_img = denoise_image(input_image_arr)
        cv2.imwrite('denoised_image.jpg', denoised_img)

        # Step 2: Threshold
        thresholded_img = apply_thresholding(denoised_img)
        cv2.imwrite('thresholded_image.jpg', thresholded_img)

        # Step 3: Delete small islands
        island_deleted_img = delete_small_islands(
            thresholded_img,
            min_island_size=50,
            connectivity=8
        )
        cv2.imwrite('island_deleted_image.jpg', island_deleted_img)

        # Alternative advanced island deletion
        advanced_island_deleted_img = advanced_island_deletion(
            thresholded_img,
            min_island_size=50,
            max_island_size=1000,
            connectivity=8
        )
        cv2.imwrite('advanced_island_deleted_image.jpg', advanced_island_deleted_img)

        print("Image processing complete!")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")
