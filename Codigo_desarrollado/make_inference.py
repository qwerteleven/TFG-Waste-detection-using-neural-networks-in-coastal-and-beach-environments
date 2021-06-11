import multiprocessing
import cv2
import torch
import compress_json
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import os
import json


def make_analysis():
    """

# Faster R-CNN

"COCO-Detection/faster_rcnn_R_50_C4_1x/137257644/model_final_721ade.pkl"
"COCO-Detection/faster_rcnn_R_50_DC5_1x/137847829/model_final_51d356.pkl"
"COCO-Detection/faster_rcnn_R_50_FPN_1x/137257794/model_final_b275ba.pkl"
"COCO-Detection/faster_rcnn_R_50_C4_3x/137849393/model_final_f97cb7.pkl"
"COCO-Detection/faster_rcnn_R_50_DC5_3x/137849425/model_final_68d202.pkl"
"COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
"COCO-Detection/faster_rcnn_R_101_C4_3x/138204752/model_final_298dad.pkl"
"COCO-Detection/faster_rcnn_R_101_DC5_3x/138204841/model_final_3e0943.pkl"
"COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
"COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl"


# RetinaNet


"COCO-Detection/retinanet_R_50_FPN_1x/190397773/model_final_bfca0b.pkl"
"COCO-Detection/retinanet_R_50_FPN_3x/190397829/model_final_5bd44e.pkl"
"COCO-Detection/retinanet_R_101_FPN_3x/190397697/model_final_971ab9.pkl"


# RPN & Fast R-CNN

"COCO-Detection/rpn_R_50_C4_1x/137258005/model_final_450694.pkl"
"COCO-Detection/rpn_R_50_FPN_1x/137258492/model_final_02ce48.pkl"
"COCO-Detection/fast_rcnn_R_50_FPN_1x/137635226/model_final_e5f7ce.pkl"


# COCO Instance Segmentation Baselines with Mask R-CNN

"COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x/137259246/model_final_9243eb.pkl"
"COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x/137260150/model_final_4f86c3.pkl"
"COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl"
"COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x/137849525/model_final_4ce675.pkl"
"COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x/137849551/model_final_84107b.pkl"
"COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
"COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x/138363239/model_final_a2914c.pkl"
"COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x/138363294/model_final_0464b7.pkl"
"COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl"
"COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl"


# COCO Person Keypoint Detection Baselines with Keypoint R-CNN

"COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x/137261548/model_final_04e291.pkl"
"COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl"
"COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x/138363331/model_final_997cc7.pkl"
"COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x/139686956/model_final_5ad38f.pkl"



# COCO Panoptic Segmentation Baselines with Panoptic FPN

"COCO-PanopticSegmentation/panoptic_fpn_R_50_1x/139514544/model_final_dbfeb4.pkl"
"COCO-PanopticSegmentation/panoptic_fpn_R_50_3x/139514569/model_final_c10459.pkl"
"COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl"


    """

    with open("./weight_detectron2.txt", 'r') as reader:
        for line in reader.readlines():

            parts = line.split('/')

            model_inference_test_detectron2("/" + parts[1],
                                            line.split('\n')[0],
                                            "./detectron2/configs/" + parts[0])

    print("done")


def start_thead(target):

    """
    to prevent the models from getting the resources of
    the graph, they are launched as threads

    """
    p = multiprocessing.Process(target=target)
    p.start()
    p.join()


def model_inference_test_detectron2(model_name, url_weigth, config_dir):

    with open('./annotations.json') as f:
        data = json.load(f)

    cfg = get_cfg()
    cfg.merge_from_file(config_dir + model_name + ".yaml")
    cfg.MODEL.WEIGHTS = "detectron2://" + url_weigth
    pred = DefaultPredictor(cfg)
    os.makedirs("./RESULTS/Datos/detectron2" + model_name, exist_ok=True)
    os.makedirs("./Images_inference/detectron2" + model_name, exist_ok=True)


    os.makedirs("./RESULTS/Datos/detectron2" + model_name + "/batch_7", exist_ok=True)
    os.makedirs("./RESULTS/Imagenes/detectron2" + model_name + "/batch_7", exist_ok=True)


    with torch.no_grad():
        for image in data['images']:
            if image['file_name'].split('/')[0] == "batch_7":
                inputs = cv2.imread('./Dataset_test/' + image['file_name'])
                outputs = pred(inputs)
                os.makedirs("./RESULTS/Datos/detectron2" + model_name + "/" + image['file_name'].split("/")[0], exist_ok=True)

                instance = outputs["instances"]

                path = "./RESULTS/Datos/detectron2" + model_name + "/" + image['file_name'].split(".")[0] + ".json"

                compress_json.dump({'pred_boxes': instance.pred_boxes.tensor.tolist(),
                    'scores': instance.scores.tolist(),
                    'pred_classes': instance.pred_classes.tolist(),
                    #'pred_masks': instance.pred_masks.tolist()
                    },
                    path + ".gz")

            # visualizacion de los resultados

                v = Visualizer(inputs[:, :, ::-1])
                out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                cv2.imwrite("./RESULTS/Imagenes/detectron2" + model_name + "/" + image['file_name'], out.get_image()[:, :, ::-1])

                print("Vamos por la imagen numero: ", image['file_name'], "\n")


if __name__ == '__main__':
    make_analysis()