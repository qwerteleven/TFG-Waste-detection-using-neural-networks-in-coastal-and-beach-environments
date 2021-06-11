import multiprocessing
import cv2
import torch
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


def make_analysis():
    """

# COCO Instance Segmentation Baselines with Mask R-CNN

"COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl"

    """

    line = "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x/137259246/model_final_9243eb.pkl"
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


    cfg = get_cfg()
    cfg.merge_from_file(config_dir + model_name + ".yaml")
    cfg.MODEL.WEIGHTS = "detectron2://" + url_weigth
    pred = DefaultPredictor(cfg)


    with torch.no_grad():
        inputs = cv2.imread('./httpswww.istockphoto.comnlsearch2imagemediatype=&phrase=microplastics.jpg')
        outputs = pred(inputs)

        v = Visualizer(inputs[:, :, ::-1])
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite("./RESULTS.jpg", out.get_image()[:, :, ::-1])



if __name__ == '__main__':
    make_analysis()