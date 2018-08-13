#!/bin/bash
set -e
set -v

mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle
cd ./data

kaggle competitions download -w -c airbus-ship-detection

unzip test.zip -d ./test
rm test.zip
unzip train.zip -d ./train
rm train.zip
unzip train_ship_segmentations.csv.zip
rm train_ship_segmentations.csv.zip
unzip sample_submission.csv.zip
rm sample_submission.csv.zip
