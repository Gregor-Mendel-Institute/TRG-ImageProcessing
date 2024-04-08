"""
Mask R-CNN
Train on the treering dataset.

------------------------------------------------------------

Usage: run from the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 Train_Rings.py train --dataset=dataset/folder --weights=coco

    # Resume training a model that you had trained earlier
    python3 Train_Rings.py train --dataset=dataset/folder --weights=last

    # Train a new model starting from ImageNet weights
    python3 Train_Rings.py train --dataset=dataset/folder --weights=imagenet

"""

#if __name__ == '__main__':


def retraining(model, dataset, out_path, start_new=True):
    """
    # Validate arguments
    assert dataset, "Argument --dataset is required for training"

    print("Initial weights: ", weights)
    print("Dataset: ", dataset)
    print("Logs: ", logs)
    """
    # implement starting continuing from where it left
    ## eg. if start_new=False:

    # Train
    results = model.train(data='../training/Dataset_sample.yaml', epochs=2, imgsz=640, project=out_path)