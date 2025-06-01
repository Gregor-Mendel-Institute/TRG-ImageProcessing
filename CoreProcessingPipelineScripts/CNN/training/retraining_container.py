
import os
from ultralytics import YOLO

def create_data_yaml(dataset_path):
    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    print(f'path: {data_yaml_path}')
    with open(data_yaml_path, 'w') as f:
        f.write(f'path: {os.path.abspath(dataset_path)}\n'
                f'train: train\n'
                f'val: val\n'
                f'names:\n'
                f'  0: ring\n'
                f'  1: crack')
    return data_yaml_path

def retraining(model, dataset_path, out_path):
    """

    """
    # find data.yaml file. It should be just under the main dataset path.
    # It has to be prepared by the user for now but may be later i will create it automatically if it will be missing.
    data_yaml_path = create_data_yaml(dataset_path)

    # augmentations are in args.yaml
    # implement resuming training from where it left
    ## first find the last weight
    last_weigth_path = os.path.join(out_path, "train", "weights", "last.pt")
    if os.path.isfile(last_weigth_path):
        #load the last model from path out location
        model = YOLO(last_weigth_path)
        model.train(data=data_yaml_path, epochs=2, imgsz=640, project=out_path, resume=True)

    else:
        # Train from
        model.train(data=data_yaml_path, epochs=2, imgsz=640, project=out_path)

#### experiment
#data_yaml = "/Users/miroslav.polacek/Github/TRG_yolov8/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/training/sample_dataset/data.yaml"
#model = YOLO('yolo11x-seg.pt')