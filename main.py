from custom.pred_detection import run_detection, load_detection_model
from custom.helper import check_folder, response_mod
import os
from pathlib import Path
from pdf2image import convert_from_path
import argparse

import warnings
warnings.filterwarnings('ignore')

#load text detection model
detection_weights = "model_ckpt/detection_ckpt/best_TB.pt"
detection_model_box = load_detection_model(detection_weights)

# create directory for detection results
path_to_save_BTF = r'detectIMG/detection_BTF/'
path_to_bbox_BTF = r'detectIMG/bbox_BTF/'

def infer_image(file_name):
    
    images_pdf = convert_from_path(file_name)
    
    check_folder(path_to_save_BTF, path_to_bbox_BTF)

    for ind, img in enumerate(images_pdf):
        img_name = str("page_") +str(ind+1)+".jpg"
        img.save(img_name,"JPEG")
        page_file = img_name
        print(page_file)
        pred = run_detection(detection_model_box, page_file, path_to_save_BTF, path_to_bbox_BTF)
        os.remove(page_file)

    response = response_mod(path_to_save_BTF)
    
    # return jsonify(response)
    return response


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default="", help='pdf file path')
    args = parser.parse_args()

    response = infer_image(args.file_path)
    