from PIL import Image, ImageOps,ImageDraw, ImageFilter
import math
import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import filedialog
import json
from image_helper_functions import *

def find_first_last_x(image, row):
    # Finde die erste und letzte x-Position, an der die Zeile 0 ist
    first_x = np.argmax(image[row] == 0)
    last_x = image.shape[1] - np.argmax(image[row, ::-1] == 0) - 1
    
    return first_x, last_x
def find_neighbour_contour(image, row, x,prev=True):
    row_vals=image[row]
    if not prev:
        return x+np.argmax(row_vals[x:]>0)
    else:
        return x-np.argmax(np.flip(row_vals[:x])>0)





def work_on_img():
    path=filedialog.askopenfilename()#r"C:\Users\nicol\Downloads\Cookie_cutter_3_SW.png"
    if not path:
        return    
    parameters=get_parameters_from_GUI()
    input_img=Image.open(path).convert('L')
    gray=np.asarray(input_img)
    if path[-3:].lower()=="jpg" or path[-4:].lower()=="jpeg":
        gray=denoise_image(gray.astype(np.uint8))
        #cv.imshow("denoised",gray)
    #binary_image=np.where(gray>127)  
    _, binary_image = cv.threshold(gray, 100,255, cv.THRESH_BINARY)
    cv.imshow("bin",binary_image)
    seed_point = (0, 0)
    flood_color = 220  
    outside_comp=binary_image.copy()
    tpl=cv.floodFill(outside_comp, None, seed_point, flood_color)
    area=tpl[0]# later: check if area is low enough
    _, outside= cv.threshold(outside_comp,128,255, cv.THRESH_BINARY)

    #outercontours, outerhierarchy = cv.findContours(cv.bitwise_not(outside), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    (ox,oy,ow,oh)=cv.boundingRect(cv.bitwise_not(outside))
    ppmm=int(max(ow,oh)/parameters.get("target_max"))
    parameters["ppmm"]=ppmm #pixel per mm
    rim_dilation_distance=int(ppmm*parameters.get("w_rim"))
    rim_kernel= np.ones((rim_dilation_distance, rim_dilation_distance), np.uint8)
    dil_4_rim=cv.dilate(cv.bitwise_not(binary_image), rim_kernel, iterations=1)
    _,rim_binary=cv.threshold(cv.bitwise_and(dil_4_rim,outside),200,255,cv.THRESH_BINARY)
    
    #Main contour thickening
    thresh=cv.bitwise_not(binary_image)
    dilation_distance=int(ppmm*parameters.get("min_wall"))
    kernel = np.ones((dilation_distance, dilation_distance), np.uint8)*255
    dilated_image = cv.dilate(thresh, kernel, iterations=1)
    _,contours_binary=cv.threshold(dilated_image,200,255,cv.THRESH_BINARY)
    
    #mask for detection of outer contours vs inner
    outer_dilation_distance=int(dilation_distance*2)
    outer_kernel= np.ones((outer_dilation_distance, outer_dilation_distance), np.uint8)
    outer_dilated=cv.dilate(rim_binary, outer_kernel, iterations=1)
    _,outer_mask=cv.threshold(cv.bitwise_and(outer_dilated,cv.bitwise_not(outside)),200,255,cv.THRESH_BINARY)
    outer_contours=cv.bitwise_and(contours_binary,outer_mask)
    #cv.imshow('OUTER', outer_contours)
    #cv.imshow('CONTOURS', contours_binary)

    inner_mask=cv.bitwise_and(cv.bitwise_and(cv.bitwise_not(outer_mask),contours_binary),cv.bitwise_not(outside))
    
    #LATER add cuting edges
    #blur = cv.GaussianBlur(gray, (5, 5), 
    #                   cv.BORDER_DEFAULT)

    #composite the image:
    color_outer_contour=255
    color_inner=int(parameters.get("h_mark")/parameters.get("h_max")*255)
    color_rim=int(parameters.get("h_rim")/parameters.get("h_max")*255)
    color_connector=color_rim
    color_small_contour=int(parameters.get("h_inner")/parameters.get("h_max")*255)
    small_thres=parameters.get("small_fill")# / (input_img.width/ppmm * input_img.height/ppmm)

    composite=np.zeros_like(binary_image,np.uint8)
    _,colored_outer=cv.threshold(outer_contours,1,color_outer_contour,cv.THRESH_BINARY)
    _,colored_inner=cv.threshold(inner_mask,1,color_inner,cv.THRESH_BINARY)
    _,colored_rim=cv.threshold(rim_binary,1,color_rim,cv.THRESH_BINARY)
    composite = cv.add(composite, colored_outer)
    composite = cv.add(composite, colored_inner)
    composite = cv.add(composite, colored_rim)
    
    #connect loose parts
    mask = np.zeros_like(contours_binary)
    small_contour_mask=np.zeros_like(contours_binary)
    connectors=np.zeros_like(contours_binary)
    contours, hierarchy = cv.findContours(contours_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        (nextContour, prevContour,child,parent)=hierarchy[0][i] #tree hierarchy indexing: hierarchy[0] is the current hierarchy. i is the entry of the current contour.
        print(cv.contourArea(contours[i])/ppmm**2,small_thres)
        if cv.contourArea(contours[i])/ppmm**2<small_thres:
            cv.drawContours(small_contour_mask, contours, i, color_small_contour, -1)
        if parent!=0: #0=idx of outerContour hierarchy[0][i][2]==-1 and  
            overlap=np.sum(cv.bitwise_and(cv.drawContours(np.zeros_like(contours_binary), [contours[i]], 0, 1),contours_binary))
            if overlap/cv.contourArea(contours[i])>0.05:#some overlap is enough
                cv.drawContours(mask, contours, i, 255, -1)# Draw contour on mask
                (x, y, w, h) = cv.boundingRect(contours[i])
                if h>1:
                    for idr in range(y,y+h):
                        prev=find_neighbour_contour(contours_binary,idr,x)
                        nex=find_neighbour_contour(contours_binary,idr, x+w,prev=False)
                        connectors[idr,prev:nex]=color_connector
    small_contour_mask=cv.bitwise_and(small_contour_mask,cv.bitwise_not(contours_binary))                    
    #cv.imshow('Connectors',connectors)
    composite = cv.add(composite, connectors)
    composite = cv.add(composite,small_contour_mask)
    cv.imshow('All',composite)


    #add insert
    clearence=int(1*ppmm)
    color_insert=int(parameters.get("h_rim")/parameters.get("h_max")*255)
    color_inner_insert=color_inner-color_insert
    clearence_kernel=np.ones((clearence,clearence))
    extra_dil_contours=cv.dilate(contours_binary,clearence_kernel, iterations=1)
    inner_insert=cv.bitwise_and(cv.bitwise_not(extra_dil_contours),cv.bitwise_not(outside))
    _,colored_insert=cv.threshold(inner_insert,1,color_insert,cv.THRESH_BINARY)

    inner_inner_insert=cv.erode(colored_insert,clearence_kernel,iterations=1)
    _,colored_inner_insert=cv.threshold(inner_inner_insert,1,color_inner_insert,cv.THRESH_BINARY)

    insert_composite=cv.add(colored_insert,colored_inner_insert)
    cv.imshow("insert",insert_composite)


    heightmap=Image.fromarray(composite)
    insert_map=Image.fromarray(insert_composite)

    #im_rgba=ImageOps.invert(im_rgba)
    heightmap.save('heightmap.png')
    insert_map.save('insert_map.png')
    filestream=open("cookie_config.json","w")
    json.dump(parameters,filestream)
    filestream.close()
    #input("saved ausgabe.png")

def get_parameters_from_GUI():
    parameters={}
    parameters["target_max"]=float(target_max_var.get())
    parameters["min_wall"]=float(min_wall_var.get())
    parameters["h_max"]=float(h_max_var.get())
    parameters["h_mark"]=float(h_mark_var.get())
    parameters["h_inner"]=float(h_inner_var.get())
    parameters["h_rim"]=float(h_rim_var.get())
    parameters["w_rim"]=float(w_rim_var.get())
    parameters["small_fill"]=float(small_fill_var.get())
    return parameters

if __name__ == "__main__":
    root=tk.Tk()
    target_max_label=tk.Label(root,text="Zielgröße [mm]:")
    target_max_label.grid(row=2,column=1)
    target_max_var=tk.StringVar(root,"100")
    target_max_inp = tk.Entry(root,width = 5,textvariable=target_max_var)
    target_max_inp.grid(row=2,column=2,padx=10,pady=5)

    min_wall_label=tk.Label(root,text="Mindestwandstärke [mm]:")
    min_wall_label.grid(row=3,column=1)
    min_wall_var=tk.StringVar(root,"1")
    min_wall_inp = tk.Entry(root,width = 5,textvariable=min_wall_var)
    min_wall_inp.grid(row=3,column=2,padx=10,pady=5)

    h_max_label=tk.Label(root,text="Gesamthöhe [mm]:")
    h_max_label.grid(row=4,column=1)
    h_max_var=tk.StringVar(root,"10")
    h_max_inp = tk.Entry(root,width = 5,textvariable=h_max_var)
    h_max_inp.grid(row=4,column=2,padx=10,pady=5)

    h_mark_label=tk.Label(root,text="Innere Linienhöhe [mm]:")
    h_mark_label.grid(row=5,column=1)
    h_mark_var=tk.StringVar(root,"9")
    h_mark_inp = tk.Entry(root,width = 5,textvariable=h_mark_var)
    h_mark_inp.grid(row=5,column=2,padx=10,pady=5)

    h_inner_label=tk.Label(root,text="Kleine Flächen Höhe [mm]:")
    h_inner_label.grid(row=6,column=1)
    h_inner_var=tk.StringVar(root,"8")
    h_inner_inp = tk.Entry(root,width = 5,textvariable=h_inner_var)
    h_inner_inp.grid(row=6,column=2,padx=10,pady=5)   

    h_rim_label=tk.Label(root,text="Bund Höhe [mm]:")
    h_rim_label.grid(row=7,column=1)
    h_rim_var=tk.StringVar(root,"1.5")
    h_rim_inp = tk.Entry(root,width = 5,textvariable=h_rim_var)
    h_rim_inp.grid(row=7,column=2,padx=10,pady=5)   

    w_rim_label=tk.Label(root,text="Bund Breite [mm]:")
    w_rim_label.grid(row=8,column=1)
    w_rim_var=tk.StringVar(root,"5")
    w_rim_inp = tk.Entry(root,width = 5,textvariable=w_rim_var)
    w_rim_inp.grid(row=8,column=2,padx=10,pady=5)     

    small_fill_label=tk.Label(root,text="Flächen unter [mm2] füllen:")
    small_fill_label.grid(row=9,column=1)
    small_fill_var=tk.StringVar(root,"50")
    small_fill_inp = tk.Entry(root,width = 5,textvariable=small_fill_var)
    small_fill_inp.grid(row=9,column=2,padx=10,pady=5)     

    button=tk.Button(root,text="select file", command=work_on_img)
    button.grid(row=20,column=2,padx=10, pady=10)

    root.mainloop()    