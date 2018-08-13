#!/bin/bash
set -e
set -v

mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle
cd ./data

kaggle competitions download -w -c airbus-ship-detection

unzip test.zip -d ./test
unzip train.zip -d ./train
rm test.zip
rm train.zip
