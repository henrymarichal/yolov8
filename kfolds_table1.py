import datetime
import shutil
from pathlib import Path
from collections import Counter
from tqdm import tqdm

import yaml
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.model_selection import KFold


def get_data(dataset_dir):
    dataset_path = Path(dataset_dir)  # replace with 'path/to/dataset' for your custom data
    labels = sorted((dataset_path / "labels").rglob("*.txt"))
    print(dataset_path / "labels")
    return dataset_path, labels

def get_class_idx(labels):
    yaml_file = './data/urudendro.yaml'  # your data YAML with data directories and names dictionary
    with open(yaml_file, 'r', encoding="utf8") as y:
        classes = yaml.safe_load(y)['names']
    cls_idx = sorted(classes)
    return cls_idx

def build_pandas_label_dataframe(labels):
    cls_idx = get_class_idx(labels)
    indx = [l.stem for l in labels]  # uses base filename as ID (no extension)
    labels_df = pd.DataFrame([], columns=cls_idx, index=indx)

    for label in labels:
        lbl_counter = Counter()

        with open(label, 'r') as lf:
            lines = lf.readlines()

        for l in lines:
            # classes for YOLO label uses integer at first position of each line
            lbl_counter[int(l.split(' ')[0])] += 1

        labels_df.loc[label.stem] = lbl_counter

    labels_df = labels_df.fillna(0.0)  # replace `nan` values with `0.0`
    return labels_df, cls_idx
def one_fold(labels_df, kfolds=5, train_val_ratio = 0.8):
    folds = [f'split_1']
    folds_df = pd.DataFrame(index=labels_df.index, columns=folds)

    idx = 1
    shape = labels_df.shape[0]
    train_idx = int(shape * train_val_ratio)
    folds_df[f'split_{idx}'].loc[labels_df.iloc[:train_idx].index] = 'train'
    folds_df[f'split_{idx}'].loc[labels_df.iloc[train_idx:].index] = 'val'
    folds_df[f'split_{idx}'].loc[labels_df.iloc[train_idx:].index] = 'test'

    return folds_df

def build_folds(labels_df, ksplit = 5, train_val_ratio = 0.8):

    kf = KFold(n_splits=ksplit, shuffle=True, random_state=20)  # setting random_state for repeatable results

    kfolds = list(kf.split(labels_df))

    folds = [f'split_{n}' for n in range(1, ksplit + 1)]
    folds_df = pd.DataFrame(index=labels_df.index, columns=folds)

    for idx, (train, test) in enumerate(kfolds, start=1):
        shape = train.shape[0]
        train_idx = int(shape * train_val_ratio)
        folds_df[f'split_{idx}'].loc[labels_df.iloc[:train_idx].index] = 'train'
        folds_df[f'split_{idx}'].loc[labels_df.iloc[train_idx:].index] = 'val'
        folds_df[f'split_{idx}'].loc[labels_df.iloc[test].index] = 'test'

    return folds_df

def load_images_path(dataset_path):
    supported_extensions = ['.jpg', '.jpeg', '.png','.JPG']
    dataset_path = Path(dataset_path)
    # Initialize an empty list to store image file paths
    images = []

    # Loop through supported extensions and gather image files
    for ext in supported_extensions:
        images.extend(sorted((dataset_path).rglob(f"*{ext}")))

    return images

def create_directory_structure(dataset_path, folds_df, ksplit, classes, labels ):


    # Initialize an empty list to store image file paths
    images = load_images_path(dataset_path / 'images/segmented')

    # Create the necessary directories and dataset YAML files (unchanged)
    save_path = Path(dataset_path / "cv" / f'{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val')
    save_path.mkdir(parents=True, exist_ok=True)
    ds_yamls = []

    for split in folds_df.columns:
        # Create directories
        split_dir = save_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
        (split_dir / 'val' / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'val' / 'labels').mkdir(parents=True, exist_ok=True)
        (split_dir / 'test' / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'test' / 'labels').mkdir(parents=True, exist_ok=True)

        # Create dataset YAML files
        dataset_yaml = split_dir / f'{split}_dataset.yaml'
        ds_yamls.append(dataset_yaml)

        with open(dataset_yaml, 'w') as ds_y:
            yaml.safe_dump({
                'path': split_dir.as_posix(),
                'train': 'train',
                'val': 'val',
                'test': 'test',  # test set is the same as validation set
                'names': classes
            }, ds_y)


    #copy images and labels to the new directory structure
    for image, label in zip(images, labels):
        for split, k_split in folds_df.loc[image.stem].items():
            # Destination directory
            img_to_path = save_path / split / k_split / 'images'
            lbl_to_path = save_path / split / k_split / 'labels'

            # Copy image and label files to new directory (SamefileError if file already exists)
            shutil.copy(image, img_to_path / image.name)
            shutil.copy(label, lbl_to_path / label.name)

    folds_df.to_csv(save_path / "kfold_datasplit.csv")
    return ds_yamls

def kfolds_cross_validation(dataset_dir, kfolds):
    dataset_path, labels = get_data(dataset_dir)

    labels_df, classes = build_pandas_label_dataframe(labels)

    #folds_df = build_folds(labels_df, kfolds)
    folds_df = one_fold(labels_df, kfolds)

    ds_yamls = create_directory_structure(dataset_path, folds_df, kfolds, classes, labels)

    return ds_yamls

def train_models(results_dir, ds_yamls, ksplit):
    weights_path = "./yolov8n.pt"
    batch = 16
    epochs = 100

    for k in tqdm(range(ksplit)):
        dataset_yaml = ds_yamls[k]
        model = YOLO(weights_path, task='detect')
        model.train(data=dataset_yaml, name=f"train_{k}", epochs=epochs, batch=batch, project=results_dir,imgsz=640,
                    workers=8) # include any train arguments

    return

def evaluate(results_dir, ds_yamls):
    results_dir = Path(results_dir)
    for k, yml in enumerate(ds_yamls):
        fold_dir = Path(f"{results_dir}/predicciones/split_{k + 1}")
        fold_dir.mkdir(parents=True, exist_ok=True)
        data = yaml.safe_load(open(yml))
        dataset_path = Path(data['path'])
        images_path = load_images_path( f'{dataset_path}/test/images')
        model = YOLO(str( results_dir / f"train_{k}/weights/best.pt"), task='detect')
        model(images_path, project = fold_dir, save=True, save_txt=True, imgsz=640, conf=0.01)

    return


def main(results_dir = '/data/maestria/resultados/yolov8', dataset_dir = '/data/maestria/resultados/dnn-pith_detector', kfolds = 5):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    ds_yamls = kfolds_cross_validation(dataset_dir, kfolds)
    train_models(results_dir, ds_yamls, kfolds)
    evaluate( results_dir=results_dir, ds_yamls=ds_yamls)
    return

def create_dataset_yaml(dataset_yaml, path, classes):
    # Create dataset YAML files
    with open(dataset_yaml, 'w') as ds_y:
        yaml.safe_dump({
            'path': path,
            'train': 'train',
            'val': 'val',
            'names': classes
        }, ds_y)


def main_evaluate_models_over_datasets(dataset_dir_for_training, results_dir):

    Path(results_dir).mkdir(parents=True, exist_ok=True)

    weights_path = "./yolov8n.pt"
    batch = 16
    epochs = 100
    dataset_yaml = Path(dataset_dir_for_training) / "dataset.yaml"
    create_dataset_yaml(dataset_yaml, dataset_dir_for_training, ["pith"])
    model = YOLO(weights_path, task='detect')
    model.train(data=dataset_yaml, name= dataset_yaml.parent.name , epochs=epochs, batch=batch, project=results_dir,imgsz=640,
                workers=8) # include any train arguments

    dataset_tree_trace_root = "/clusteruy/home/henry.marichal/dataset_pith/table2"
    #dataset_tree_trace_root = "/data/maestria/datasets/TreeTrace_Douglas_format"
    dataset_list = [dataset_tree_trace_root + "/discs", dataset_tree_trace_root + "/forest", dataset_tree_trace_root + "/logs",
                    dataset_tree_trace_root + "/logyard"]
    for d_directory in dataset_list:
        dataset_yaml = Path(d_directory) / "dataset.yaml"
        create_dataset_yaml(dataset_yaml, d_directory, ["pith"])

    dataset_list += ["/clusteruy/home/henry.marichal/dataset_pith/table2/urudendro"]
    #dataset_list += ["/data/maestria/resultados/dnn-pith_detector"]
    folder_dir = results_dir + "/predic"
    Path(folder_dir).mkdir(parents=True, exist_ok=True)
    for dataset_dir in dataset_list:
        predic_dir = folder_dir + "/" + Path(dataset_dir).name
        Path(predic_dir).mkdir(parents=True, exist_ok=True)
        images_path = load_images_path( f'{dataset_dir}/images/segmented')
        model(images_path, project = predic_dir, save=True, save_txt=True, imgsz=640, conf=0.01)

    return



if __name__ == "__main__":
    #input arguments for the script. results_dir is the directory where the results will be stored
    #dataset_dir is the directory where the dataset is stored
    #kfolds is the number of folds for the cross validation
    import argparse
    parser = argparse.ArgumentParser(description='Kfolds cross validation for YOLOv8')
    parser.add_argument('--results_dir', type=str, default='/data/maestria/resultados/yolov8',
                        help='directory where the results will be stored')
    parser.add_argument('--dataset_dir', type=str, default='/data/maestria/resultados/dnn-pith_detector',
                        help='directory where the dataset is stored')
    parser.add_argument('--kfolds', type=int, default=5,
                        help='number of folds for the cross validation')
    args = parser.parse_args()
    main(args.results_dir, args.dataset_dir, args.kfolds)
    #main_evaluate_models_over_datasets(args.dataset_dir, args.results_dir)