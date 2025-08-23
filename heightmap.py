import numpy as np
import cv2
from PIL import Image
from skimage.morphology import skeletonize


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

def process_image(pil_image:Image, parameters:dict):
    gray=np.array(pil_image.convert('L'))  # Convert to grayscale
    blurred = gray.copy()#cv2.GaussianBlur(gray, (7, 7), 0)

    # 3. Adaptives Thresholding
    _, thresh =cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    _, thresh_for_obj =cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) 

    # 4. Morphologische Operationen zum Säubern
    # "Opening" entfernt kleine Störpixel (Pfeffer-Rauschen) im Hintergrund.
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # "Closing" füllt kleine Löcher im Objekt (Salz-Rauschen).
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    #cv2.imwrite('binary_image.png', closing)
    contours, _ = cv2.findContours(thresh_for_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    obj_mask = np.zeros(gray.shape, dtype=np.uint8)
    print(f"Object mask shape: {obj_mask.shape}")

    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        print(f"Main contour area: {cv2.contourArea(main_contour)}")
        
        # 1. & 2. Fülle den inneren Bereich der Kontur auf der schwarzen Maske mit Weiß (255)
        #cv2.drawContours(obj_mask, [main_contour], -1, color=255, thickness=cv2.FILLED)
        cv2.fillPoly(obj_mask, [main_contour], 255)
        #cv2.imwrite('object_mask.png', obj_mask)
        # 3. Invertiere die Maske: Weiß -> Schwarz, Schwarz -> Weiß
        # Jetzt ist der Hintergrund weiß (255) und das Objekt schwarz (0)
        outside_mask = cv2.bitwise_not(obj_mask)
        #cv2.imwrite('outside_mask.png', outside_mask)

    #outercontours, outerhierarchy = cv2.findContours(cv2.bitwise_not(outside), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    (ox,oy,ow,oh)=cv2.boundingRect(obj_mask)
    #ppmm=int(max(ow,oh)/parameters.get("target_max"))
    #parameters["ppmm"]=ppmm #pixel per mm
    ppmm = parameters.get("ppmm", 3.77) # 96dpi as fallback
    binary_image=closing.copy()

    #rim dilation 
    rim_dilation_distance=int(ppmm*parameters.get("w_rim"))
    rim_kernel= np.ones((rim_dilation_distance, rim_dilation_distance), np.uint8)
    dil_4_rim=cv2.dilate(cv2.bitwise_not(binary_image), rim_kernel, iterations=1)
    #cv2.imwrite('dilated_rim.png', dil_4_rim)
    #cv2.imwrite('outside_mask.png', outside_mask)
    _,rim_binary=cv2.threshold(cv2.bitwise_and(dil_4_rim,outside_mask),200,255,cv2.THRESH_BINARY)
    #cv2.imwrite('rim_binary.png', rim_binary)
  
    #Main contour thickening
    skeleton= skeletonize(cv2.bitwise_not(binary_image)/255.0)
    skeleton_image = (skeleton * 255).astype(np.uint8)
    #cv2.imwrite('skeleton_image.png', skeleton_image)
    thresh=skeleton_image#cv2.bitwise_not(binary_image)
    dilation_distance=int(0.5*ppmm*parameters.get("min_wall"))
    kernel = np.ones((dilation_distance, dilation_distance), np.uint8)*255
    dilated_image = cv2.dilate(thresh, kernel, iterations=1)
    _,contours_binary=cv2.threshold(cv2.bitwise_or(dilated_image,cv2.bitwise_not(binary_image)),200,255,cv2.THRESH_BINARY)
    #cv2.imwrite('contours_binary.png', contours_binary)
    
    #mask for detection of outer contours vs inner
    outer_dilation_distance=int(dilation_distance*2)
    outer_kernel= np.ones((outer_dilation_distance, outer_dilation_distance), np.uint8)
    outer_dilated=cv2.dilate(rim_binary, outer_kernel, iterations=1)
    _,outer_mask_dil=cv2.threshold(cv2.bitwise_and(outer_dilated,cv2.bitwise_not(outside_mask)),200,255,cv2.THRESH_BINARY)
    outer_contours=cv2.bitwise_and(contours_binary,outer_mask_dil)
    #cv2.imshow('OUTER', outer_contours)
    #cv2.imshow('CONTOURS', contours_binary)

    inner_mask=cv2.bitwise_and(cv2.bitwise_and(cv2.bitwise_not(outer_mask_dil),contours_binary),cv2.bitwise_not(outer_mask_dil))
    #cv2.imwrite('inner_mask.png', inner_mask)
    
    #LATER add cuting edges
    #blur = cv2.GaussianBlur(gray, (5, 5), 
    #                   cv2.BORDER_DEFAULT)

    #composite the image:
    color_outer_contour=255
    color_inner=int((parameters.get("h_max")-parameters.get("height_dough_thickness"))/parameters.get("h_max")*255)
    color_rim=int(parameters.get("h_rim")/parameters.get("h_max")*255)
    color_connector=color_rim
    color_small_contour=int(parameters.get("h_inner")/parameters.get("h_max")*255)
    small_thres=parameters.get("small_fill")# / (input_img.width/ppmm * input_img.height/ppmm)

    composite=np.zeros_like(binary_image,np.uint8)
    _,colored_outer=cv2.threshold(outer_contours,1,color_outer_contour,cv2.THRESH_BINARY)
    _,colored_inner=cv2.threshold(inner_mask,1,color_inner,cv2.THRESH_BINARY)
    _,colored_rim=cv2.threshold(rim_binary,1,color_rim,cv2.THRESH_BINARY)
    composite = cv2.add(composite, colored_outer)
    composite = cv2.add(composite, colored_inner)
    composite = cv2.add(composite, colored_rim)
    
    #connect loose parts
    obj_mask = np.zeros_like(contours_binary)
    small_contour_mask=np.zeros_like(contours_binary)
    connectors=np.zeros_like(contours_binary)
    contours, hierarchy = cv2.findContours(contours_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        (nextContour, prevContour,child,parent)=hierarchy[0][i] #tree hierarchy indexing: hierarchy[0] is the current hierarchy. i is the entry of the current contour.
        #print(cv2.contourArea(contours[i])/ppmm**2,small_thres)
        if cv2.contourArea(contours[i])/ppmm**2<small_thres:
            cv2.drawContours(small_contour_mask, contours, i, color_small_contour, -1)
        if parent!=0: #0=idx of outerContour hierarchy[0][i][2]==-1 and  
            overlap=np.sum(cv2.bitwise_and(cv2.drawContours(np.zeros_like(contours_binary), [contours[i]], 0, 1),contours_binary))
            if overlap/cv2.contourArea(contours[i])>0.05:#some overlap is enough
                cv2.drawContours(obj_mask, contours, i, 255, -1)# Draw contour on mask
                (x, y, w, h) = cv2.boundingRect(contours[i])
                if h>1:
                    for idr in range(y,y+h):
                        prev=find_neighbour_contour(contours_binary,idr,x)
                        nex=find_neighbour_contour(contours_binary,idr, x+w,prev=False)
                        connectors[idr,prev:nex]=color_connector
    small_contour_mask=cv2.bitwise_and(small_contour_mask,cv2.bitwise_not(contours_binary))                    
    #cv2.imshow('Connectors',connectors)
    composite = cv2.add(composite, connectors)
    composite = cv2.add(composite,small_contour_mask)



    #add insert
    clearence=int(1*ppmm)
    color_insert=int(parameters.get("h_rim")/parameters.get("h_max")*255)
    color_inner_insert=color_inner-color_insert
    clearence_kernel=np.ones((clearence,clearence))
    extra_dil_contours=cv2.dilate(contours_binary,clearence_kernel, iterations=1)
    inner_insert=cv2.bitwise_and(cv2.bitwise_not(extra_dil_contours),cv2.bitwise_not(outside_mask))
    _,colored_insert=cv2.threshold(inner_insert,1,color_insert,cv2.THRESH_BINARY)

    inner_inner_insert=cv2.erode(colored_insert,clearence_kernel,iterations=1)
    _,colored_inner_insert=cv2.threshold(inner_inner_insert,1,color_inner_insert,cv2.THRESH_BINARY)

    insert_composite=cv2.add(colored_insert,colored_inner_insert)

    return composite, insert_composite
    #im_rgba=ImageOps.invert(im_rgba)
    #heightmap.save('heightmap.png')
    #insert_map.save('insert_map.png')
    

if __name__ == "__main__":
    # Example usage
    image_path = 'Cookie_Test_Image.png'  # Path to your image
    params = {
        "target_max": 100.0,
        "min_wall": 1.0,
        "h_max": 10.0,
        "h_rim": 2.0,
        "w_rim": 8.0,
        "height_dough_thickness": 2.0,
        "h_inner": 3.0,
        "small_fill": 10.0
    }
    pil_image = Image.open(image_path)
    heightmap, insert_map = process_image(pil_image, params)
    cv2.imwrite('heightmap.png', heightmap)
    cv2.imwrite('insert_map.png', insert_map)