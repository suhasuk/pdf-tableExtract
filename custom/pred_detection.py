
import os
from pathlib import Path
import numpy as np
import shutil
import torch

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages
from utils.general import (LOGGER, Profile, check_img_size, cv2, non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device, smart_inference_mode


def load_detection_model(weights):
    """
    Funtion to load the detection models
        Inputs:
            weights: CKPT path of detection models
            data:
        Return:
            model:
    """
    device, dnn, half= '', False, False
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, fp16=half)
    return model

@smart_inference_mode()
def run_detection(model, source_image, path_to_save, path_to_bbox):
    """
    Function to the run the detection model
        Input:
            model:
            source_image:
            path_to_save:
            path_to_bbox:
        Return:
            "Prediction Done!" 
    """

    source = str(source_image)

    save_img, save_obj, hide_labels, hide_conf, line_thickness =True, True, False, False, 1
    vid_stride, max_det, iou_thres, conf_thres, imgsz  = 1, 1000, 0.45, 0.2, (640,640)
    
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand

    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size


    # Dataloader
    bs = 1  # batch_size
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im)

        # NMS
        with dt[2]:
            # pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)

        page_name = os.path.basename(source).split(".")[0]
        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(path_to_bbox + "/" + p.name)  # im.jpg
            s += '%gx%g ' % im.shape[2:]  # print string
            imc = im0.copy()
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
        
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in det:    
                    if save_img:  # Add bbox to image
                        label1 = '%s %.6f' % (names[int(cls)], conf)
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.6f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    
                  ## get the crop images based on line detection  
                    if save_obj:
                        for k in range(len(det)):
                            x,y,w,h=int(xyxy[0]), int(xyxy[1]), int(xyxy[2] - xyxy[0]), int(xyxy[3] - xyxy[1])           
                            img_ = imc.astype(np.uint8)
                            crop_img=img_[y:y+ h, x:x + w]                             
                            #!!rescale image !!!
                            filename=str(page_name) + "_" + str(y) + "_" + label1+ '{:}.jpg'.format(+1)
                            filepath=os.path.join(path_to_save, filename)
                            cv2.imwrite(filepath, crop_img)

                                  
                    else:
                        print("There is no detected object")
                        continue
                    # Save results (image with detections)
                    if save_img:
                        if dataset.mode == 'image':
                            cv2.imwrite(save_path, im0)
    return "Prediction done"
                            
             