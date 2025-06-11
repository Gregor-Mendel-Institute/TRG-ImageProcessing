#!/bin/zsh

python evaluate_weights.py \
--dataset=sample_dataset \
--weight=../weights/best10px1000eAugEnlargedDataset.pt \
--out_path=../output \
--test_name=anotherDebugTest \
--debug=True

