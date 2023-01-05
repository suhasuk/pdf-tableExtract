from custom.pred_detection import run_detection, load_detection_model
from custom.helper import check_folder, response_mod
import os
# import shutil
from pathlib import Path
from flask import Flask, request, jsonify
from pdf2image import convert_from_path


import warnings
warnings.filterwarnings('ignore')

#load text detection model
detection_weights = "model_ckpt/detection_ckpt/best_TB.pt"
detection_model_box = load_detection_model(detection_weights)

# create directory for detection results
path_to_save_BTF = r'detectIMG/detection_BTF/'
path_to_bbox_BTF = r'detectIMG/bbox_BTF/'
# create_dir(path_to_bbox_BTF)
# create_dir(path_to_save_BTF)



app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = './static'

@app.route('/predict', methods=['POST'])
def infer_image():
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"
    
    file = request.files.get('file')
    file.save("test.pdf")
    file_name = "test.pdf"
    images_pdf = convert_from_path(file_name)
    os.remove(file_name)

    # create_dir(path_to_bbox_BTF)
    # create_dir(path_to_save_BTF)
    check_folder(path_to_save_BTF, path_to_bbox_BTF)

    for ind, img in enumerate(images_pdf):
        img_name = str("page_") +str(ind+1)+".jpg"
        img.save(img_name,"JPEG")
        page_file = img_name
        print(page_file)
        pred = run_detection(detection_model_box, page_file, path_to_save_BTF, path_to_bbox_BTF)
        os.remove(page_file)

    response = response_mod(path_to_save_BTF)
    # return jsonify({"TB": "pdf"})

    return jsonify(response)

@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8500)