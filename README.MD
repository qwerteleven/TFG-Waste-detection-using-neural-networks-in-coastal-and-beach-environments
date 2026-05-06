# Waste Detection Using Neural Networks in Coastal and Beach Environments

A final-year thesis project (TFG) benchmarking pre-trained [object detection](https://en.wikipedia.org/wiki/Object_detection) models for identifying litter in coastal and beach imagery. Models from [Detectron2](https://github.com/facebookresearch/detectron2) and [MMDetection](https://github.com/open-mmlab/mmdetection) are evaluated against the [TACO](http://tacodataset.org/) dataset using a custom class-mapping pipeline, with results exported as CSV and LaTeX tables.

---

## Goal

Assess how well general-purpose detectors pre-trained on [COCO](https://cocodataset.org/) and [LVIS](https://www.lvisdataset.org/) can detect beach waste without retraining, by mapping their output classes to TACO's waste super-categories (Plastic, Can, Glass, Carton, etc.).

---

## Datasets & Class Mapping (`Clases_datasets/`)

Four datasets are involved, each with a mapping file that translates their classes to a common set of 12 waste super-categories:

| Super-category | Examples |
|---|---|
| Plastic | Clear plastic bottle, plastic film, garbage bag |
| Can | Drink can, food can, aerosol |
| Glass | Glass bottle, glass jar, broken glass |
| Carton | Corrugated carton, drink carton, pizza box |
| Metal | Aluminium foil, scrap metal, metal lid |
| Paper | Normal paper, magazine, tissues |
| Wrapper | Crisp packet, retort pouch |
| Cup | Paper cup, foam cup, disposable plastic cup |
| Straw | Plastic straw, paper straw |
| String/rope | Rope, shoes, handbags |
| Pop tab | Pull tabs |
| Other | Cigarettes, food waste, unlabelled litter |

The `classes_COCO_map.txt` and `classes_TACO_map.txt` files define how each dataset's classes map to these super-categories, enabling cross-dataset evaluation.

---

## Pipeline (`Codigo_desarrollado/`)

### 1. Data Preparation

- **`data_augmentation_script_COCO_to_YOLO.py`** — Applies [data augmentation](https://en.wikipedia.org/wiki/Data_augmentation) to images in YOLO format using the [imgaug](https://imgaug.readthedocs.io/) library: fog, rain, clouds, Gaussian noise, horizontal flip, brightness multiplication, and pixelation. Labels are copied and adjusted for flipped images.
- **`generate_train.py` / `generate_test.py`** — Scans a directory tree and writes `train.txt` / `test.txt` file lists for YOLO-format training.
- **`get_annoted_files.py`** — Copies only images that have a matching label file into a `dataset_final/` folder.
- **`trim_spaces.py`** — Removes spaces from all `.jpg` and `.txt` filenames recursively.

### 2. Inference

- **`make_inference.py`** — Loads Detectron2 models by URL from `weight_detectron2.txt`, runs inference on the test dataset, and saves predictions as compressed JSON (`.json.gz`) alongside visualised output images.
- **`make_inference_MM.py`** — Same pipeline for MMDetection models listed in `weight_MMdetection.txt`. Filters detections with confidence > 0.5.
- **`pruebas.py`** — Quick single-model inference test using Mask R-CNN R50-FPN.
- **`train_detectron2_model.ipynb`** — Fine-tunes Mask R-CNN on the TACO dataset for 500 iterations using Detectron2's `DefaultTrainer`.

### 3. Evaluation (`metricas.py`, `make_analisis.py`)

For each model, detections are matched against TACO ground-truth annotations via [IoU](https://en.wikipedia.org/wiki/Jaccard_index) (bounding-box intersection over union). Class predictions are remapped from COCO to TACO super-categories using the mapping files, then the following metrics are computed both per-image and per-class:

| Metric | Formula |
|---|---|
| [Accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision) | (TP + TN) / (TP + TN + FP + FN) |
| [Precision](https://en.wikipedia.org/wiki/Precision_and_recall) | TP / (TP + FP) |
| [Recall](https://en.wikipedia.org/wiki/Precision_and_recall) | TP / (TP + FN) |
| [F1 score](https://en.wikipedia.org/wiki/F-score) | 2 × (Precision × Recall) / (Precision + Recall) |
| TPR | TP / (TP + FN) |
| FPR | FP / (FP + TN) |

Results are written to CSV files in `RESULTS/Metricas/`.

### 4. LaTeX Export (`make_table_latex.py`)

Converts CSV metric files to LaTeX `tabular` tables. Underscores in filenames are replaced with hyphens for LaTeX compatibility. Pre-generated `.latex` files are in `latex-tables/` (by class) and `metrics_by_img/` (by image).

---

## Models Evaluated

### Detectron2 (30 models)
Faster R-CNN, Mask R-CNN, RetinaNet, Keypoint R-CNN, and Panoptic FPN variants with ResNet-50, ResNet-101, and ResNeXt-101 backbones, across C4, DC5, and FPN feature extractors, at 1× and 3× schedules.

### MMDetection (100+ models)
Mask R-CNN, Cascade R-CNN, RetinaNet, SSD, FCOS, DETR, YOLOv3, CenterNet, and many others with various backbones and training schedules.

> Model weights are not included. URLs are listed in `weight_url/weight_detectron2.txt` and `weight_url/weight_MMdetection.txt`.

---

## Key Finding

General COCO-pretrained detectors show reasonable recall for **Plastic** (the dominant TACO class) but perform poorly on less common categories like **Straw**, **Paper**, and **String/rope**, which are either absent or semantically distant from COCO classes. Keypoint R-CNN models produce zero detections for all waste classes, as expected. RetinaNet achieves high recall but at the cost of very low precision due to many false positives.

---

## Project Structure

```
tfg-waste-detection/
├── Clases_datasets/          # Class lists and COCO/TACO/LVIS/ADE20K mapping files
├── Codigo_desarrollado/
│   ├── data_augmentation_script_COCO_to_YOLO.py
│   ├── generate_train.py / generate_test.py
│   ├── get_annoted_files.py / trim_spaces.py
│   ├── make_inference.py        # Detectron2 inference
│   ├── make_inference_MM.py     # MMDetection inference
│   ├── make_analisis.py         # IoU computation
│   ├── metricas.py              # TP/FP/FN/metric computation
│   ├── make_table_latex.py      # CSV → LaTeX
│   ├── train_detectron2_model.ipynb
│   └── weight_url/              # Model weight URLs
├── latex-tables/                # Per-class metric tables (.csv.latex)
└── metrics_by_img/              # Per-image metric tables (.csv and .csv.latex)
```

---

## Requirements

```bash
pip install torch torchvision detectron2 mmdet mmcv imgaug compress-json pandas
```

Input data: TACO dataset annotations (`annotations.json`) and images in `Dataset_test/`, structured as `batch_N/XXXXXX.jpg`.

---

## License

MIT