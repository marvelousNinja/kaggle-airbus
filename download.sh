#!/bin/bash
set -e
set -v

mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle
cd ./data

kaggle competitions download -w -c airbus-ship-detection

unzip test_v2.zip -d ./test
rm test_v2.zip
unzip train_v2.zip -d ./train
rm train_v2.zip
unzip train_ship_segmentations_v2.csv.zip
rm train_ship_segmentations_v2.csv.zip
unzip sample_submission_v2.csv.zip
rm sample_submission_v2.csv.zip
