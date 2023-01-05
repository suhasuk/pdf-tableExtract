""" 
    Script for helper funtion in Table detectiona on crop tables models. 
    It contain helper function for:
    1. Text detection model (Box dectection i.e. Table)
"""
# Import the required inbuilt pacakages 
import re
import math
import os
import numpy as np
import re
from pathlib import Path
from pdf2image import convert_from_path
import shutil
import base64


# code to check which file shoul pass though recog model drop_duplicate_box(), get_path_boxes() 

def drop_duplicate_box(dic_detect):
    """
    Funtion to drop the duplicate boxes identified from detection model
        Input:
            dic_detect: Dictionary containg the box name and cooridate value with overlapped
        Return:
            dic_detect: Dictionary containing drop duplicate boxes
    """
    if "fb" in dic_detect or "tb" in dic_detect:
        if len(dic_detect)>2 and "fb" in dic_detect and "tb" in dic_detect:
            fb_tb = dic_detect['tb'] + dic_detect['fb']
        elif len(dic_detect) <3 and 'fb' in dic_detect: 
            fb_tb = dic_detect['fb']
        elif len(dic_detect) <3 and 'tb' in dic_detect: 
            fb_tb = dic_detect['tb']
        
        if 'box' not in dic_detect or len(dic_detect['box']) < 1:
            pass
        else:
            for val1 in fb_tb:
                for val2 in dic_detect['box']:
                    if abs(int(val1)-int(val2)) < 50:
                        dic_detect['box'].remove(val2)
    else:
        start = 0
        next_ = 1
        box_val = dic_detect['box']
        sort_box = np.sort(dic_detect['box'])
        while next_ != len(sort_box):
            if abs(int(sort_box[start]) - int(sort_box[next_])) < 50:
                box_val.remove(sort_box[start])
            start+=1
            next_+=1

        dic_detect['box'] = box_val

    return dic_detect
        

def get_path_boxes(image_path):
    """
    Funtion to get the the path of detection box after dropiing the overlapped boxes
    in detection model
        Input:
            image_path: Image directory of detection model boxes
        Return:
            path_list: Path list of detection boxes after droping the duplicates
    """
    # img_list = image_path
    img_list = [os.path.join(image_path, i) for i in os.listdir(image_path)]
    
    dic_detect = {}
    path_dict = {}
    path_list = []
    
    for img in np.sort(img_list):
        i = os.path.basename(img)
        sp = i.split(" ")[0].split("_")
        if sp[1] not in dic_detect:
            dic_detect[sp[1]] = []  
            dic_detect[sp[1]].append(sp[0])      
        else:
            dic_detect[sp[1]].append(sp[0])
        
        path_dict[sp[0]] = img
    
    if len(dic_detect) < 1:
        pass
    else:
        dict_drop = drop_duplicate_box(dic_detect)
    
        for key, value in dict_drop.items():
            for i in value:
                if i in path_dict:
                    path_list.append(path_dict[i])
    return path_list

def create_dir(image_path):
    """
    Function to create recursive directory
        Input:
            image_path:  path of dir to be created
    """
    if not os.path.exists(image_path):
        path = Path(image_path)
        path.mkdir(parents=True)
    else:
        print("valid dir")
    


def convert_pdf2img_file(files):
  """
  Function to convert pdf to jpg
    Args:
        Input: 
            files: PDF file having to convert
        Return:
            convert into jpg file same in folder:
  """
  images = convert_from_path(files)
  for i, image in enumerate(images):
    img = str("page_")+"." +str(i+1)+".jpg"
    image.save("images/"+img,"JPEG")

def check_folder(path_to_save, path_to_bbox):
    """
    Funtion to check if folder exists and contain previous data then delete 
    other wise create the new folder with given path
        Agrs:
            Input:
                path_to_save: Path of input images
                path_to_bbox: Path of cropped images to be saved
            Return:
                check based on condition and perform
    """
    # path_to_save = r'inference/detection/'
    # path_to_bbox = r'inference/bbox/'
    if os.path.exists(path_to_save):
        shutil.rmtree(path_to_save)
    
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    if os.path.exists(path_to_bbox):
        shutil.rmtree(path_to_bbox)
    
    if not os.path.exists(path_to_bbox):
        os.makedirs(path_to_bbox)
    
def response_mod(path_to_save):
    """
    Function to convert crop image into base64 and response structure
        Agrs:
            Input:
                path_to_save: path of crop detected bbox
            Return:
                response: Response containing number of box in each pages
                    with base64 code for cropped images
    """
    page_ind_list = [os.path.join(path_to_save, i) for i in os.listdir(path_to_save)]
    dict_pgind = {}
    response = {}

    for i in page_ind_list:
        bs = os.path.basename(i).split("_")[:2]
        bs_name = bs[0] + "_" + bs[1]
        if bs_name not in dict_pgind:
            dict_pgind[bs_name] = []
            dict_pgind[bs_name].append(i)
        else:
            dict_pgind[bs_name].append(i)
    
    for key, val in dict_pgind.items():
        page_response = {}
        count = 1
        for path in np.sort(val):
            key_name = key + "." + str(count)
            with open(path, "rb") as image2string:
                base_string = base64.b64encode(image2string.read()).decode("utf-8")
                page_response[key_name] = base_string
            count+=1
        page_response['Number of Table'] = len(val)
        response[key] = page_response
    
    return response
