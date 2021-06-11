import multiprocessing
import compress_json
import os
import json

import mmcv
from mmdet.apis import init_detector, inference_detector


def make_analysis():
    """
    http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r50_caffe_fpn_1x_coco/rpn_r50_caffe_fpn_1x_coco_20200531-5b903a37.pth
    http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r50_fpn_1x_coco/rpn_r50_fpn_1x_coco_20200218-5525fa2e.pth
    http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r50_fpn_2x_coco/rpn_r50_fpn_2x_coco_20200131-0728c9b3.pth
    http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r101_caffe_fpn_1x_coco/rpn_r101_caffe_fpn_1x_coco_20200531-0629a2e2.pth
    http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r101_fpn_1x_coco/rpn_r101_fpn_1x_coco_20200131-2ace2249.pth
    http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_r101_fpn_2x_coco/rpn_r101_fpn_2x_coco_20200131-24e3db1a.pth
    http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_x101_32x4d_fpn_1x_coco/rpn_x101_32x4d_fpn_1x_coco_20200219-b02646c6.pth
    http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_x101_32x4d_fpn_2x_coco/rpn_x101_32x4d_fpn_2x_coco_20200208-d22bd0bb.pth
    http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_x101_64x4d_fpn_1x_coco/rpn_x101_64x4d_fpn_1x_coco_20200208-cde6f7dd.pth
    http://download.openmmlab.com/mmdetection/v2.0/rpn/rpn_x101_64x4d_fpn_2x_coco/rpn_x101_64x4d_fpn_2x_coco_20200208-c65f524f.pth

    """

    with open("./weight_MMdetection.txt", 'r') as reader:
        for line in reader.readlines():
            parts = line.split('/')


            model = "/" + parts[-2]
            config_file = "../framework/mmdetection/configs/" + parts[-3] + model + ".py"
            checkpoint_file = line


            start_thead(model_inference_test_MMdetection(model, checkpoint_file, config_file))

    print("done")


def start_thead(target):
    """
    to prevent the models from getting the resources of
    the graph, they are launched as threads

    """
    p = multiprocessing.Process(target=target)
    p.start()
    p.join()


def model_inference_test_MMdetection(model_name, checkpoint_file, config_file):
    with open('./annotations.json') as f:
        data = json.load(f)

    # download weights
    print("-----------")

    if not './checkpoint/' + checkpoint_file:
        return

    os.makedirs("./RESULTS/Datos/MMdetection" + model_name, exist_ok=True)
    os.makedirs("./Images_inference/MMdetection" + model_name, exist_ok=True)
    os.makedirs("./RESULTS/Datos/MMdetection" + model_name + "/batch_7", exist_ok=True)
    os.makedirs("./RESULTS/Imagenes/MMdetection" + model_name + "/batch_7", exist_ok=True)

    model = init_detector(config_file, "./checkpoint/" + checkpoint_file.split('/')[-1].strip(), device='cuda:0')

    for image in data['images']:
        if image['file_name'].split('/')[0] == "batch_7":
            inputs = './Dataset_test/' + image['file_name']
            img = mmcv.imread(inputs)
            result = inference_detector(model, img)

            pred_boxes = []
            scores = []
            pred_classes = []
            n_class = 0

            for arr in result:
                n_class = n_class + 1
                for detection in arr:
                    print(arr)
                    if
                    if type(detection) != list and detection.shape[0].all() > 4 and detection[-1] > 0.5:
                        pred_boxes.append(detection[:-1].tolist())
                        scores.append(str(detection[-1]))
                        pred_classes.append(n_class)

            path = "./RESULTS/Datos/MMdetection" + model_name + "/" + image['file_name'].split(".")[0] + ".json"

            compress_json.dump({'pred_boxes': pred_boxes,
                    'scores': scores,
                    'pred_classes': pred_classes
                    #'pred_masks': instance.pred_masks.tolist()
                            },
                            path + ".gz")

                # visualizacion de los resultados
            out_path = "RESULTS/Imagenes/MMdetection" + model_name + "/" + image['file_name']
            print("----", out_path, "------")

            model.show_result(inputs, result, out_file=out_path)

            print("Vamos por la imagen numero: ", image['file_name'], "\n")

if __name__ == '__main__':
    make_analysis()