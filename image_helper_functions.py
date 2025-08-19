import cv2
import numpy as np
#from skimage import measure

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
    output_path (str): Path to save the processed image
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

# Existing functions from previous script (denoise_image, etc.)
def denoise_image(img, method='nl_means'):
    """
    Denoise an image using various techniques
    
    Parameters:
    img = cv2 grayscale image
    output_path (str): Path to save the denoised image
    method (str): Denoising method to use 
    
    Returns:
    numpy.ndarray: Denoised image
    """
    
    if method == 'nl_means':
        # Non-local means denoising
        denoised = cv2.fastNlMeansDenoising(img)#, None, 10, 10, 7, 21) #cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    elif method == 'bilateral':
        # Bilateral filtering
        denoised = cv2.bilateralFilter(img)#, 9, 75, 75)
    else:
        raise ValueError("Invalid denoising method")
       
    return denoised

def apply_thresholding(input_path, output_path, method='adaptive'):
    """
    Apply thresholding to the image
    
    Parameters:
    input_path (str): Path to the input image
    output_path (str): Path to save the thresholded image
    method (str): Thresholding method ('adaptive' or 'otsu')
    
    Returns:
    numpy.ndarray: Thresholded image
    """
    # Read the image
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
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
    
    # Save the processed image
    cv2.imwrite(output_path, thresh)
    
    return thresh

# Example usage
if __name__ == '_main_':
    input_image = 'original_image.jpg'
    
    # Full processing pipeline
    denoised_image = 'denoised_image.jpg'
    thresholded_image = 'thresholded_image.jpg'
    island_deleted_image = 'island_deleted_image.jpg'
    
    # Step 1: Denoise
    denoise_image(input_image, denoised_image)
    
    # Step 2: Threshold
    apply_thresholding(denoised_image, thresholded_image)
    
    # Step 3: Delete small islands
    delete_small_islands(
        thresholded_image, 
        island_deleted_image, 
        min_island_size=50,  # Adjust based on your image
        connectivity=8
    )
    
    # Alternative advanced island deletion
    advanced_island_deleted_image = 'advanced_island_deleted_image.jpg'
    advanced_island_deletion(
        thresholded_image, 
        advanced_island_deleted_image, 
        min_island_size=50,
        max_island_size=1000,  # Optional maximum size
        connectivity=8
    )
    
    print("Image processing complete!")