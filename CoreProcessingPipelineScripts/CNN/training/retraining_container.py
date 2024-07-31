
import os.path
from ultralytics import YOLO

def retraining(model, dataset, out_path):
    """

    """
    # find data.yaml file. It should be just under the main dataset path.
    # It has to be prepared by the user for now but may be later i will create it automatically if it will be missing.
    data_yaml = os.path.join(dataset, "data.yaml")
    if not os.path.isfile(data_yaml):
        print("No such file or directory", data_yaml)
        print("Create data.yaml file according to provided example and place it in dataset folder")
        exit()

    # augmentations are in args.yaml
    # implement resuming training from where it left
    ## first find the last weight
    last_weigth_path = os.path.join(out_path, "train", "weights", "last.pt")
    if os.path.isfile(last_weigth_path):
        #load the last model from path out location
        model = YOLO(last_weigth_path)
        model.train(data=data_yaml, epochs=2, imgsz=640, project=out_path, resume=True)

    else:
        # Train from
        model.train(data=data_yaml, epochs=2, imgsz=640, project=out_path)