import os
import cv2
import multiprocessing
import json

predict = []
times = []

path_dataset = "./Dataset_test/"
path_labels = "./annotations.json"


def load_labels(folder):

    """

    labels = {instances : segmentation : image}

    Load file with the detections and the name of image

    """



    return labels


def load_images_from_folder(folder):
    """
    Load dataset images and names in two version Gray and BGR

    *see cv2 imread()

    """

    images = []
    filesnames = []
    fold = list(os.walk(folder))
    for i in range(1, len(fold)):
        path = fold[i][0] + "/"
        for filename in fold[i][2]:
            img = cv2.imread(path + filename)
            filesnames.append(filename)
            if img is not None:

                height, width, channels = img.shape

                if height > 900 or width > 900:

                    img = cv2.resize(img, (int(height / 2), int(width / 2)))

                images.append(img)

    return images, filesnames




def output_result(results, model_name):

    """
    create an output file to persist the scan results

    """

    f = open("./RESULTS/result_" + model_name + ".csv", "a+")
    f.write("image, AP, IoU, TP, TN, FP, FN \n")

    for result in results:
        for fact in result:
            f.write(fact + ",")
        f.write("\n")
    f.close()







def Viola_Jones_():
    # Viola-Jones
    face_cascade = cv2.CascadeClassifier("./weights_models/haar/haarcascade_frontalface_alt.xml")
    predict, times = tradicional_methods.predit_haar(grays, face_cascade)
    output_result("viola-jones", filesnames, files, labels, predict, times)


def start_thead(target):
    """
    to prevent the models from getting the resources of
    the graph, they are launched as threads

    """
    p = multiprocessing.Process(target=target)
    p.start()
    p.join()


def make_analysis():

    """
    f = open("./results/face_results.txt", "w")
    f.truncate(0)
    f.close()

    """


    start_thead(Viola_Jones_)
    print("done")



files, labels = load_labels(path_labels)

images, grays, filesnames = load_images_from_folder(path)

if __name__ == '__main__':

    make_analysis()
