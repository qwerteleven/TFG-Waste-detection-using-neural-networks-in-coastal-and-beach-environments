import json

"""

Precision True positive / (true positive + false positive)

Recall   true positive / (true positive + false negative)

F1    2 * ((precision * recall) / (precision + recall))

"""

def accuracy(TP, TN, FP, FN):
    if (TP + TN + FP + FN) == 0:
        return 0

    return (TP + TN) / (TP + TN + FP + FN)


def precision(TP, FP):
    if (TP + FP) == 0:
        return 0

    return TP / (TP + FP)


def recall(TP, FN):
    if (TP + FN) == 0:
        return 0

    return TP / (TP + FN)


def F1(TP, FP, FN):
    if (precision(TP, FP) + recall(TP, FN)) == 0:
        return 0

    return 2 * ((precision(TP, FP) * recall(TP, FN)) / (precision(TP, FP) + recall(TP, FN)))


def TPR(TP, FN):
    if (TP + FN) == 0:
        return 0

    return TP / (TP + FN)


def FPR(FP, TN):
    if (FP + TN) == 0:
        return 0

    return FP / (FP + TN)


def read_data(model):

    map_COCO = []
    with open("./Clases_datasets/classes_COCO_map.txt", 'r') as reader:
        for index, line in enumerate(reader.readlines()):
            map_COCO.append([index, line.split(',')[1].split('\n')[0]])

    data = []
    with open("./RESULTS/Metricas/detectron2/raw" + model + 'data.csv', 'r') as reader:
        for line in reader.readlines():
            classes = line.split('[')[1].split(']')[0].split(', ')

            for map in map_COCO:
                for index, cl in enumerate(classes):
                    if cl == str(map[0]):
                        classes[index] = map[1].strip()

            image = line.split(',')[0]
            IoU = line.split(']')[1].split(',')[1:-1]
            data.append([image, classes, IoU])

    return data


def get_GT_classes():

    with open("./annotations.json") as f:
        TACO = json.load(f)

    map_TACO = []
    images = TACO['images']
    annotations = TACO['annotations']

    GT_classes = []
    for annotation in annotations:
        GT_classes.append([annotation['image_id'], annotation['category_id']])

    for image in images:
        for GT_classe in GT_classes:
            if GT_classe[0] == image['id']:
                GT_classe[0] = image['file_name']


    with open("./Clases_datasets/classes_TACO_map.txt", 'r') as reader:
        for index, line in enumerate(reader.readlines()):
            map_TACO.append([index, line.split(',')[1].split('\n')[0]])

    for GT_classe in GT_classes:
        for map in map_TACO:
            if map[0] == GT_classe[1]:
                GT_classe[1] = map[1].strip()

    data = {}

    for GT_classe in GT_classes:
        if not GT_classe[0] in data:
            data[GT_classe[0]] = [GT_classe[1]]
        else:
            data[GT_classe[0]] = data[GT_classe[0]] + [GT_classe[1]]

    return data

def sum_list(l):
    value = 0
    for data in l:
        value += float(data)

    return value

def mean(data_list):
    value = 0
    for data in data_list:
        value += float(data)

    if value == 0:
        return 0

    return value / len(data_list)


def make_metrics_by_instance(GT_classes, data):

    metrics = []

    for instance in data:
        TP = 0
        FP = 0
        FN = 0
        TN = 0

        for GT_class in GT_classes[instance[0]]:
            if GT_class in instance[1]:
                TP += 1
            if GT_class not in instance[1]:
                FN += 1

        for predicted_class in instance[1]:
            if predicted_class == 'none':
                continue

            if predicted_class not in GT_classes[instance[0]]:
                FP += 1

            if predicted_class in GT_classes[instance[0]]:
                TN += 1

        metrics.append([instance[0], mean(instance[2]), TP, FP, FN, TN,
                        accuracy(TP, TN, FP, FN),
                        precision(TP, FP),
                        recall(TP, FN),
                        F1(TP, FP, FN),
                        TPR(TP, FN),
                        FPR(FP, TN)
                        ])
    return metrics


def make_metrics_by_class(GT_classes, data):

    metrics = []

    name_classes = [
        'Can',
        'Other',
        'Metal',
        'Plastic',
        'Carton',
        'Wrapper',
        'Glass',
        'Paper',
        'String/rope',
        'Cup',
        'Straw',
        'Pop tab'
    ]
    for name_class in name_classes:
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        IoU = 0
        n_IoU = 0.000008

        for instance in data:
            for GT_class in GT_classes[instance[0]]:
                if GT_class != name_class:
                    continue

                if GT_class in instance[1]:
                    TP += 1
                if GT_class not in instance[1]:
                    FN += 1

            for index, predicted_class in enumerate(instance[1]):
                if predicted_class == 'none' or name_class != predicted_class:
                    continue
                IoU += float(instance[2][index])
                n_IoU += 1

                if predicted_class not in GT_classes[instance[0]]:
                    FP += 1
                if predicted_class in GT_classes[instance[0]]:
                    TN += 1

        metrics.append([name_class, IoU / n_IoU, TP, FP, FN, TN,
                        accuracy(TP, TN, FP, FN),
                        precision(TP, FP),
                        recall(TP, FN),
                        F1(TP, FP, FN),
                        TPR(TP, FN),
                        FPR(FP, TN)
                        ])
    return metrics


def to_csv(name, tipe, data, head):
    f = open("./RESULTS/Metricas/detectron2" + tipe + name + '.csv', "w")
    f.write(head)
    f.write('\n')

    for instance in data:
        for index, fact in enumerate(instance):
            if index > 0:
                fact = round(fact, 3)
            f.write(str(fact))
            if index < len(instance):
                f.write(',')

        f.write('\n')
    f.close()

def make_metrics(model, GT_classes, head_image, head_class):
    data = read_data(model)
    metrics_image = make_metrics_by_instance(GT_classes, data)
    metrics_class = make_metrics_by_class(GT_classes, data)
    to_csv(model, "/by-img", metrics_image, head_image)
    to_csv(model, "/by-class", metrics_class, head_class)


if __name__ == '__main__':

    GT_classes = get_GT_classes()
    head_image = "img, IoU_img, TP, FP, FN, TN, accuracy, precision, recall, F1, TPR, FPR,"
    head_class = "Class, IoU_img, TP, FP, FN, TN, accuracy, precision, recall, F1, TPR, FPR,"

    with open("./weight_detectron2.txt", 'r') as reader:
        for line in reader.readlines():
            parts = line.split('/')

            make_metrics("/" + parts[1], GT_classes, head_image, head_class)

    print("done")

